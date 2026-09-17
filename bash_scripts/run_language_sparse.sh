#!/bin/bash
# Run ONE training run. Submitted by sweep.sh as an array task, or directly:
#
#   sbatch --export=ALL,FIRE_REPO_ROOT=$PWD bash_scripts/run_language_sparse.sh \
#          --method vanilla --sparsifier dense
#
# In array mode each task takes its own "name|args" line from $SWEEP_RUN_LIST.
# Site specifics (modules, venv, paths) come from env.sh + a cluster profile.

set -euo pipefail

# shellcheck disable=SC1091
source "${FIRE_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/bash_scripts/env.sh"
# shellcheck disable=SC1091
source "$FIRE_REPO_ROOT/bash_scripts/sweep.conf"

ARGS=("$@")
RUN_NAME="manual"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -n "${SWEEP_RUN_LIST:-}" ]]; then
    LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SWEEP_RUN_LIST")"
    if [[ -z "$LINE" ]]; then
        echo "ERROR: no run on line $SLURM_ARRAY_TASK_ID of $SWEEP_RUN_LIST" >&2
        exit 1
    fi
    RUN_NAME="${LINE%%|*}"
    read -r -a ARGS <<< "${LINE#*|}"
    echo "array task ${SLURM_ARRAY_TASK_ID}: $RUN_NAME"
fi

fire_load_modules
fire_activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_DIR="$FIRE_WANDB_DIR"
mkdir -p "$FIRE_WANDB_DIR" "$FIRE_OUTPUT_DIR"

echo "run=$RUN_NAME"
fire_show_env
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true

# Warm the page cache: get_batch reads random 2KB windows, which is ~4x slower
# on a cold cache (114k vs 500k tokens/sec). Sequential pre-read is cheap.
for name in wikitext openwebtext; do
    f="$FIRE_DATA_DIR/$name/train.bin"
    [[ -f "$f" ]] && { echo "warming $name"; timeout 900 cat "$f" > /dev/null || true; }
done

cd "$FIRE_REPO_ROOT/language"
echo "args: ${COMMON_ARGS[*]} --wandb_mode=$WANDB_MODE ${ARGS[*]}"
exec python -u train_sparse.py "${COMMON_ARGS[@]}" --wandb_mode="$WANDB_MODE" "${ARGS[@]}"
