#!/bin/bash
# T4.1 profiling: data statistics, training throughput and eval cost for each
# model, on one GPU. Submit from the repo root:
#
#   sbatch --account=def-yani cl_ft_language/scripts/profile.sbatch
#   MODELS="Qwen/Qwen2.5-0.5B" PARTS="train" sbatch --account=def-yani cl_ft_language/scripts/profile.sbatch
#
# Results: $TRACE_OUTPUT_DIR/profile/<part>_<model>.json; log in logs/.
#SBATCH --job-name=trace-profile
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck disable=SC1091
source cl_ft_language/scripts/env.sh
trace_load_modules
trace_activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false

MODELS="${MODELS:-HuggingFaceTB/SmolLM2-135M Qwen/Qwen2.5-0.5B}"
PARTS="${PARTS:-data train eval}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for model in $MODELS; do
    for part in $PARTS; do
        python cl_ft_language/src/profile_trace.py --part "$part" --model "$model" ${PROFILE_ARGS:-}
    done
done
