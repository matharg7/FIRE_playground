#!/bin/bash
# One W&B sweep agent = one Slurm array task = one training run.
#
# Submitted by launch_wandb_sweep.sh; $SWEEP_ID must be in the environment.
# Not meant to be run by hand.
#
#SBATCH --job-name=trace-sweep
#SBATCH --ntasks=1
#SBATCH --output=logs/trace/%x-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
# shellcheck disable=SC1091
source cl_ft_language/scripts/env.sh
trace_load_modules
trace_activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # weights come from the local cache
export WANDB_DIR="$TRACE_WANDB_DIR"
mkdir -p "$TRACE_WANDB_DIR" "$TRACE_OUTPUT_DIR"

trace_show_env
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true
echo "sweep : ${SWEEP_ID:?SWEEP_ID not set -- launch through launch_wandb_sweep.sh}"
echo "agent : array task ${SLURM_ARRAY_TASK_ID:-<none>}"

# --count 1: take exactly one configuration and exit. One run per job keeps the
# Slurm accounting honest (a job's walltime is one run's walltime) and means a
# crash costs one run rather than the rest of the queue. The server hands out
# each grid point once, so agents never collide.
cd "$TRACE_ROOT"
exec wandb agent --count 1 "$SWEEP_ID"
