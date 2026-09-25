#!/bin/bash
# Create a W&B sweep and submit one Slurm array task per run.
#
#   ./scripts/launch_wandb_sweep.sh scripts/sweep_rigl.yaml               # dry run
#   ./scripts/launch_wandb_sweep.sh scripts/sweep_rigl.yaml --submit
#   ./scripts/launch_wandb_sweep.sh scripts/sweep_rigl.yaml --agents 3 --submit
#   ./scripts/launch_wandb_sweep.sh --resume <sweep_id> --agents 2 --submit
#
# The dry run prints the grid, the run count and the resources, and creates
# nothing. With --submit it runs `wandb sweep` to register the grid, then
# submits an array of agents, each taking one configuration (--count 1).
#
# --agents defaults to the size of the grid, i.e. every run starts as soon as
# the scheduler has room. Fewer agents just means the runs are serialised.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/sweep.conf"

YAML=""
AGENTS=0
SUBMIT=0
RESUME=""
MAX_CONCURRENT="${TRACE_MAX_CONCURRENT:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)    SUBMIT=1; shift ;;
        --agents)    AGENTS="${2:?--agents needs a number}"; shift 2 ;;
        --agents=*)  AGENTS="${1#*=}"; shift ;;
        --resume)    RESUME="${2:?--resume needs a sweep id}"; shift 2 ;;
        --resume=*)  RESUME="${1#*=}"; shift ;;
        --max-concurrent)   MAX_CONCURRENT="${2:?needs a number}"; shift 2 ;;
        --max-concurrent=*) MAX_CONCURRENT="${1#*=}"; shift ;;
        -h|--help)   sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)          echo "ERROR: unknown option: $1" >&2; exit 1 ;;
        *)           YAML="$1"; shift ;;
    esac
done

cd "$TRACE_REPO_ROOT"

# ---- how many runs does this grid contain? ----------------------------------
# Read it from the yaml so the dry run can report it without contacting W&B.
grid_size() {
    python - "$1" <<'PY'
import sys, itertools
try:
    import yaml
except ImportError:
    print("?"); sys.exit(0)
with open(sys.argv[1]) as fh:
    spec = yaml.safe_load(fh)
n = 1
swept = []
for key, body in (spec.get("parameters") or {}).items():
    if isinstance(body, dict) and "values" in body:
        n *= len(body["values"])
        swept.append(f"{key}={body['values']}")
print(n)
print("\n".join(swept))
PY
}

if [[ -n "$RESUME" ]]; then
    SWEEP_ID="$RESUME"
    N_RUNS="${AGENTS:-1}"
    echo "resuming sweep: $SWEEP_ID"
else
    [[ -n "$YAML" ]] || { echo "ERROR: give a sweep yaml (or --resume <id>)" >&2; exit 1; }
    [[ -f "$YAML" ]] || YAML="$TRACE_ROOT/$YAML"
    [[ -f "$YAML" ]] || { echo "ERROR: no such yaml: $YAML" >&2; exit 1; }
    mapfile -t INFO < <(grid_size "$YAML")
    N_RUNS="${INFO[0]}"
    # The yaml's `project:` governs; `wandb sweep` would otherwise put every
    # sweep in whatever is hardcoded here, and the run's own --wandb_project is
    # ignored inside a sweep ("Ignoring project ... when running a sweep").
    PROJECT_DEFAULT=dst_trace_benchmark
    PROJECT="$(sed -n 's/^project:[[:space:]]*//p' "$YAML" | head -1)"
    echo "sweep yaml : $YAML"
    echo "grid       : $N_RUNS run(s)"
    for line in "${INFO[@]:1}"; do [[ -n "$line" ]] && echo "  swept: $line"; done
fi

[[ "$AGENTS" == "0" ]] && AGENTS="$N_RUNS"

PROJECT="${PROJECT:-${PROJECT_DEFAULT:-dst_trace_benchmark}}"
echo "project    : $PROJECT"
echo "agents     : $AGENTS  (one run each, --count 1)"
echo "resources  : --time=$TIME --gpus-per-task=$GPUS_PER_TASK --cpus-per-task=$CPUS_PER_TASK --mem-per-cpu=$MEM_PER_CPU"
echo "account    : ${TRACE_ACCOUNT:-<none>}"
echo "wandb mode : $TRACE_WANDB_MODE"

if [[ "$TRACE_WANDB_MODE" != "online" ]]; then
    echo "ERROR: W&B sweeps need TRACE_WANDB_MODE=online (agents pull configs from the server)." >&2
    echo "       Set it in scripts/local.sh." >&2
    exit 1
fi

if [[ "$SUBMIT" != "1" ]]; then
    echo "(dry run; add --submit to create the sweep and queue the agents)"
    exit 0
fi

# ---- register the grid with W&B ---------------------------------------------
if [[ -z "$RESUME" ]]; then
    trace_load_modules
    trace_activate
    echo
    echo "creating sweep..."
    # `wandb sweep` prints the id on stderr; capture both and pull it out.
    SWEEP_OUT="$(wandb sweep --project "$PROJECT" "$YAML" 2>&1 | tee /dev/stderr)"
    SWEEP_ID="$(grep -oE 'wandb agent [^ ]+' <<<"$SWEEP_OUT" | tail -1 | awk '{print $3}')"
    [[ -n "$SWEEP_ID" ]] || { echo "ERROR: could not parse a sweep id from wandb output" >&2; exit 1; }
    echo "sweep id: $SWEEP_ID"
fi

# ---- queue the agents --------------------------------------------------------
ARRAY="1-$AGENTS"
[[ "$MAX_CONCURRENT" != "0" ]] && ARRAY="$ARRAY%$MAX_CONCURRENT"

mkdir -p "$TRACE_REPO_ROOT/logs/trace"
ACCOUNT_ARG=()
[[ -n "$TRACE_ACCOUNT" ]] && ACCOUNT_ARG=(--account="$TRACE_ACCOUNT")

sbatch "${ACCOUNT_ARG[@]}" \
    --array="$ARRAY" \
    --time="$TIME" \
    --gpus-per-task="$GPUS_PER_TASK" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem-per-cpu="$MEM_PER_CPU" \
    --export=ALL,SWEEP_ID="$SWEEP_ID",TRACE_ROOT="$TRACE_ROOT" \
    "$TRACE_ROOT/scripts/wandb_agent.sh"

echo
echo "sweep : https://wandb.ai/ucalgary/$PROJECT/sweeps/${SWEEP_ID##*/}"
echo "resume: ./scripts/launch_wandb_sweep.sh --resume $SWEEP_ID --agents N --submit"
