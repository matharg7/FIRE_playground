#!/bin/bash
# Run ONE TRACE continual-learning run. Submitted by sweep.sh as an array task,
# or directly:
#
#   sbatch --account=<acct> --gpus-per-task=1 --cpus-per-task=6 \
#          --mem-per-cpu=8G --time=08:00:00 cl_ft_language/scripts/run_trace.sh \
#          --sparsifier rigl --sparsity 0.1
#
# In array mode each task takes its own "name|args" line from $SWEEP_RUN_LIST.
# Site specifics (modules, venv, paths) come from scripts/env.sh.
#SBATCH --job-name=trace-run
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/sweep.conf"

ARGS=("$@")
RUN_NAME=""
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

trace_load_modules
trace_activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # weights come from the local cache
export WANDB_DIR="$TRACE_WANDB_DIR"
mkdir -p "$TRACE_WANDB_DIR" "$TRACE_OUTPUT_DIR"

trace_show_env
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true

NAME_ARG=()
[[ -n "$RUN_NAME" ]] && NAME_ARG=(--run_name="$RUN_NAME")

echo "args: ${COMMON_ARGS[*]} ${NAME_ARG[*]} ${ARGS[*]}"
exec python -u "$TRACE_ROOT/src/train.py" \
    --model="$MODEL" \
    --data_dir="$TRACE_DATA_DIR" --out_root="$TRACE_OUTPUT_DIR" \
    --wandb_mode="$TRACE_WANDB_MODE" --wandb_dir="$TRACE_WANDB_DIR" \
    "${COMMON_ARGS[@]}" "${NAME_ARG[@]}" "${ARGS[@]}"
