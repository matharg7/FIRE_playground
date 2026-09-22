#!/bin/bash
# Build the TRACE sweep from sweep.conf and submit it as a SLURM job array.
#
#   ./sweep.sh                     # list the runs, submit nothing
#   ./sweep.sh --submit            # submit the array
#   ./sweep.sh --filter rigl --submit     # only runs whose name matches
#   ./sweep.sh --dense-only --submit
#
# Each array task reads its own line from the generated run list, so the whole
# sweep is one entry in the queue. Everything site-specific comes from env.sh
# and scripts/local.sh; nothing here hardcodes a path or an account.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/sweep.conf"

SUBMIT=0
FILTER=""
DENSE_ONLY=0
RUN_LIST="$TRACE_REPO_ROOT/logs/trace/sweep_runs.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)     SUBMIT=1; shift ;;
        --filter)     FILTER="${2:?--filter needs an argument}"; shift 2 ;;
        --filter=*)   FILTER="${1#*=}"; shift ;;
        --dense-only) DENSE_ONLY=1; shift ;;
        --out)        RUN_LIST="${2:?--out needs an argument}"; shift 2 ;;
        -h|--help)    sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$RUN_LIST")"
: > "$RUN_LIST"

add_run() {   # name, then args
    local name="$1"; shift
    [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && return 0
    printf '%s|%s\n' "$name" "$*" >> "$RUN_LIST"
    printf '  %-44s %s\n' "$name" "$*"
}

echo "runs (model $MODEL):"
for seed in "${SEEDS[@]}"; do
    add_run "dense_seed${seed}" --sparsifier=dense --seed="$seed"
    [[ "$DENSE_ONLY" == "1" ]] && continue
    for sparsity in "${SPARSITIES[@]}"; do
        for grow in "${GROW_INITS[@]}"; do
            for sched in "${DROP_FRACTION_SCHEDULES[@]}"; do
                add_run "rigl_s${sparsity}_${grow}_${sched}_seed${seed}" \
                    --sparsifier=rigl --sparsity="$sparsity" \
                    --grow_init="$grow" --drop_fraction_schedule="$sched" \
                    --seed="$seed"
            done
        done
    done
done

N=$(wc -l < "$RUN_LIST")
echo
echo "$N run(s) -> $RUN_LIST"
echo "each: --time=$TIME --gpus-per-task=$GPUS_PER_TASK --cpus-per-task=$CPUS_PER_TASK --mem-per-cpu=$MEM_PER_CPU"
[[ -n "$TRACE_ACCOUNT" ]] && echo "account: $TRACE_ACCOUNT" || echo "account: none set (TRACE_ACCOUNT in scripts/local.sh)"

if [[ "$N" -eq 0 ]]; then
    echo "nothing to submit"; exit 0
fi
if [[ "$SUBMIT" != "1" ]]; then
    echo "(dry run; add --submit to queue it)"; exit 0
fi

mkdir -p "$TRACE_REPO_ROOT/logs"
ACCOUNT_ARG=()
[[ -n "$TRACE_ACCOUNT" ]] && ACCOUNT_ARG=(--account="$TRACE_ACCOUNT")
cd "$TRACE_REPO_ROOT"
sbatch "${ACCOUNT_ARG[@]}" \
    --array="1-$N" \
    --time="$TIME" \
    --gpus-per-task="$GPUS_PER_TASK" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem-per-cpu="$MEM_PER_CPU" \
    --export=ALL,SWEEP_RUN_LIST="$RUN_LIST",TRACE_ROOT="$TRACE_ROOT" \
    "$TRACE_ROOT/scripts/run_trace.sh"
