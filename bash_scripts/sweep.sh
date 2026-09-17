#!/bin/bash
# Build the language sweep from sweep.conf and submit it as a SLURM job array.
#
#   ./sweep.sh                 # show the runs and what they will cost
#   ./sweep.sh --submit        # submit the array
#   ./sweep.sh --filter rigl_s0.5 --submit    # only matching runs
#
# Each array task reads its own line from the generated run list, so the whole
# sweep is one job in the queue rather than thirty.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
REPO_ROOT="$FIRE_REPO_ROOT"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/sweep.conf"

SUBMIT=0
FILTER=""
RUN_LIST="$REPO_ROOT/logs/language/sweep_runs.txt"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)    SUBMIT=1; shift ;;
        --filter)    FILTER="${2:?--filter needs an argument}"; shift 2 ;;
        --filter=*)  FILTER="${1#*=}"; shift ;;
        --out)       RUN_LIST="${2:?--out needs an argument}"; shift 2 ;;
        -h|--help)   sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)           echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$RUN_LIST")"
: > "$RUN_LIST"

# tokens a run trains on; full_reset skips chunk 0
chunk1_tokens=$(( OPENWEBTEXT_TOKENS * 75 / 100 ))
chunk0_tokens=$(( 110 * WIKITEXT_TOKENS ))

add_run() {   # name | throughput | skip_chunk0 | args...
    local name="$1" throughput="$2" skip="$3"; shift 3
    [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && return 0
    local tokens=$chunk1_tokens
    [[ "$skip" == "0" ]] && tokens=$(( chunk0_tokens + chunk1_tokens ))
    local hours; hours=$(awk -v t="$tokens" -v r="$throughput" 'BEGIN{printf "%.1f", t/r/3600}')
    printf '%s|%s\n' "$name" "$*" >> "$RUN_LIST"
    printf '%-34s %6sh  %s\n' "$name" "$hours" "$*"
    TOTAL_HOURS=$(awk -v a="${TOTAL_HOURS:-0}" -v b="$hours" 'BEGIN{printf "%.1f", a+b}')
    MAX_HOURS=$(awk -v a="${MAX_HOURS:-0}" -v b="$hours" 'BEGIN{print (b>a)?b:a}')
}

for method in "${DENSE_METHODS[@]}"; do
    skip=0; [[ "$method" == "full_reset" ]] && skip=1
    extra=""; [[ "$method" == "fire" ]] && extra="--fire_iteration=$FIRE_ITERATION"
    # shellcheck disable=SC2086
    add_run "dense_${method}" "$THROUGHPUT_DENSE" "$skip" \
        --method="$method" --sparsifier=dense --compile=True $extra
done

for sparsity in "${SPARSITIES[@]}"; do
    for pruning in "${PRUNING_RATIOS[@]}"; do
        for updates in "${MASK_UPDATES[@]}"; do
            # RigL compiles too. At batch 12 compiling it was 11% slower, but at
            # the sweep's batch 60 it is 6.8% faster (417k vs 391k tok/s), same
            # memory. Measured at the real config, not extrapolated.
            add_run "rigl_s${sparsity}_pr${pruning}_nmu${updates}" "$THROUGHPUT_RIGL" 0 \
                --method=vanilla --sparsifier=rigl --compile=True \
                --sparsity="$sparsity" --pruning_ratio="$pruning" \
                --num_mask_updates="$updates"
        done
    done
done

N=$(wc -l < "$RUN_LIST")
echo
echo "cluster: $FIRE_CLUSTER   W&B: $WANDB_MODE   account: ${FIRE_ACCOUNT:-<none>}"
echo "$N runs · ${TOTAL_HOURS:-0} GPU-hours · longest ${MAX_HOURS:-0}h (walltime $WALLTIME)"
echo "run list: $RUN_LIST"

if [[ "$SUBMIT" != "1" ]]; then
    echo "(dry run; pass --submit to launch the array)"
    exit 0
fi

ARRAY="1-$N"
[[ "$MAX_CONCURRENT" != "0" ]] && ARRAY="$ARRAY%$MAX_CONCURRENT"

# Absolute --chdir/--output and an explicit repo root: sbatch copies the script
# into a spool directory, so relative paths and BASH_SOURCE do not work there.
# GPU and memory flags differ by site, so they come from the cluster profile.
SBATCH_ARGS=(
    --array="$ARRAY"
    --job-name=fire_lang
    --time="$WALLTIME"
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    "${FIRE_SBATCH_GPU[@]}"
    "${FIRE_SBATCH_MEM[@]}"
    "${FIRE_SBATCH_EXTRA[@]}"
    --chdir="$REPO_ROOT"
    --output="$REPO_ROOT/logs/language/%x_%A_%a.out"
    --export=ALL,FIRE_REPO_ROOT="$REPO_ROOT",SWEEP_RUN_LIST="$RUN_LIST",FIRE_WANDB_MODE="$WANDB_MODE"
)
# Sites that bill to an account need one; others reject the flag.
[[ -n "${FIRE_ACCOUNT:-}" ]] && SBATCH_ARGS=(--account="$FIRE_ACCOUNT" "${SBATCH_ARGS[@]}")

echo "submitting on $FIRE_CLUSTER: sbatch ${SBATCH_ARGS[*]}"
sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_language_sparse.sh"
