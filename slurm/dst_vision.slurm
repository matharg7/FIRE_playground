#!/usr/bin/env bash
#SBATCH --job-name=DST_VISION_SWEEP_AGENT
#SBATCH --account=rrg-yani
#SBATCH --nodes=1
#SBATCH --gpus=h100_10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=8:00:00
#SBATCH -o ../slurm_logs/dst_vision_agent_%j.out
#SBATCH -e ../slurm_logs/dst_vision_agent_%j.err
#SBATCH --mail-user=fredella.pang@ucalgary.ca
#SBATCH --mail-type=ALL
#
# Generic W&B sweep-agent launcher for the DST vision fine-tuning recipe.
#
# A single job starts one `wandb agent` that pulls 3 hyperparameter sets
# consecutively from the sweep server and runs bash_scripts/run_dst_vision.sh
# with those values. The sweep config (sweep.yaml, created separately) owns
# the grid; this file is just the agent template.
#
# Usage (submit from this slurm/ directory so the ../slurm_logs and the sweep's
# ../bash_scripts/run_lora_math.sh relative paths resolve against the repo root):
#   cd slurm && sbatch run_lora_math.slurm <SWEEP_ID>
#
# The sweep's `command:` must invoke run_lora_math.sh, e.g.:
#   command:
#     - ${env}
#     - bash
#     - ../bash_scripts/run_lora_math.sh
#     - ${args}
# so the swept --lora_rank / --lr / --model flags are forwarded to the script.

set -euo pipefail

# --- 0. Validation ---
if [ -z "${1-}" ]; then
  echo "Error: No SWEEP_ID provided."
  echo "Usage: sbatch slurm/run_lora_math.slurm <SWEEP_ID>"
  exit 1
fi
SWEEP_ID="$1"

# --- 1. Environment (mirror run_lora_math.sh so the wandb CLI + venv match) ---
PYTHON_MODULE="python/3.11.5"
SCIPY_MODULE="scipy-stack/2026a"
ARROW_MODULE="arrow/24.0.0"
CUDA_MODULE="cuda/13.2"
VENV="${VENV:-/scratch/${USER}/fire-env}"

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "ERROR: virtual environment not found at: $VENV" >&2 
  echo "       build it first with ./build.sh $VENV" >&2
  exit 1
fi

echo "==> Loading modules (StdEnv/2023 $PYTHON_MODULE $SCIPY_MODULE $ARROW_MODULE $CUDA_MODULE)"
module load StdEnv/2023 "$PYTHON_MODULE" "$SCIPY_MODULE" "$ARROW_MODULE" "$CUDA_MODULE"

echo "==> Activating venv: $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# --- 2. W&B agent configuration ---
if [[ ! -f ~/.wandb_token ]]; then
  echo "Error: ~/.wandb_token not found." >&2
  exit 1
fi
export WANDB_API_KEY="$(cat ~/.wandb_token)"
export WANDB_DIR="/scratch/${USER}/wandb_agents"
mkdir -p "$WANDB_DIR"

echo "[INFO] Starting sweep agent for ID: $SWEEP_ID"

# --- 3. Launch the agent (pulls 3 configs consecutively, runs run_dst_vision.sh) ---
wandb agent --count 3 "$SWEEP_ID"