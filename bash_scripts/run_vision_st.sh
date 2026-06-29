#!/bin/bash
# Launch vision/train_st.py for one sparse/dense configuration.
# Reuses module loading, repo-root and venv setup from run_lora_math.sh.
#
# Usage:
#   ./run_vision_st.sh                                  # dense baseline (config defaults)
#   ./run_vision_st.sh --sparsifier rigl                # one sparse run
#   ./run_vision_st.sh --sparsifier set --sparsity 0.95 --pruning-ratio 0.5
#   ./run_vision_st.sh --model TinyViT --task CIFAR100  # different arch/dataset
#   ./run_vision_st.sh --log-subdir sweep1 --gpu 1      # nest log, pick GPU
#   ./run_vision_st.sh --seed 42                        # single seed
#   ./run_vision_st.sh --seed 2,3,8                     # multiple seeds (one run each)
#
# Flags accept `--flag value` or `--flag=value`. Output is captured to
# <repo>/logs/<run_name>.out (or logs/<subdir>/<run_name>.out with --log-subdir).

set -euo pipefail

# Script lives in <repo>/bash_scripts/, so the repo root is one level up.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Defaults mirror vision/config_st.py.
SPARSIFIER="dense"
SPARSITY="0.9"
PRUNING_RATIO="0.3"
NUM_MASK_UPDATES="500"
MODEL="RESNET18"
TASK="CIFAR10"
BENCHMARK="continual"
LOG_SUBDIR="Dense"
GPU="0"
SEEDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sparsifier)            SPARSIFIER="${2:?--sparsifier needs an argument}"; shift 2 ;;
        --sparsifier=*)          SPARSIFIER="${1#*=}"; shift ;;
        --sparsity)              SPARSITY="${2:?--sparsity needs an argument}"; shift 2 ;;
        --sparsity=*)            SPARSITY="${1#*=}"; shift ;;
        --pruning-ratio|--pruning_ratio)
                                 PRUNING_RATIO="${2:?--pruning-ratio needs an argument}"; shift 2 ;;
        --pruning-ratio=*|--pruning_ratio=*)
                                 PRUNING_RATIO="${1#*=}"; shift ;;
        --num-mask-updates|--num_mask_updates)
                                 NUM_MASK_UPDATES="${2:?--num-mask-updates needs an argument}"; shift 2 ;;
        --num-mask-updates=*|--num_mask_updates=*)
                                 NUM_MASK_UPDATES="${1#*=}"; shift ;;
        --model)                 MODEL="${2:?--model needs an argument}"; shift 2 ;;
        --model=*)               MODEL="${1#*=}"; shift ;;
        --task)                  TASK="${2:?--task needs an argument}"; shift 2 ;;
        --task=*)                TASK="${1#*=}"; shift ;;
        --benchmark)             BENCHMARK="${2:?--benchmark needs an argument}"; shift 2 ;;
        --benchmark=*)           BENCHMARK="${1#*=}"; shift ;;
        --log-subdir|--log_subdir)
                                 LOG_SUBDIR="${2:?--log-subdir needs an argument}"; shift 2 ;;
        --log-subdir=*|--log_subdir=*)
                                 LOG_SUBDIR="${1#*=}"; shift ;;
        --gpu)                   GPU="${2:?--gpu needs an argument}"; shift 2 ;;
        --gpu=*)                 GPU="${1#*=}"; shift ;;
        --seed)                  SEEDS="${2:?--seed needs an argument}"; shift 2 ;;
        --seed=*)                SEEDS="${1#*=}"; shift ;;
        -h|--help)               sed -n '4,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)                       echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

VENV="$SCRATCH/fire-env"
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: virtual environment not found at: $VENV" >&2
    exit 1
fi

# HPC modules (StdEnv + the four from run_lora_math.sh).
echo "==> Loading modules"
module load StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/24.0.0 cuda/13.2

echo "==> Activating venv: $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Use exactly the venv's packages, not the host's ~/.local.
export PYTHONNOUSERSITE=1
if [[ -f "$HOME/.wandb_token" ]]; then
    export WANDB_API_KEY="$(cat "$HOME/.wandb_token")"
fi
export CUDA_VISIBLE_DEVICES="$GPU"

# Run-name base for log files (the W&B run name is built inside train_st.py).
case "$SPARSIFIER" in
    dense)  run_base="${MODEL}_${TASK}_dense" ;;
    static) run_base="${MODEL}_${TASK}_static_s${SPARSITY}" ;;
    gmp)    run_base="${MODEL}_${TASK}_gmp_s${SPARSITY}_nmu${NUM_MASK_UPDATES}" ;;
    *)      run_base="${MODEL}_${TASK}_${SPARSIFIER}_s${SPARSITY}_pr${PRUNING_RATIO}_nmu${NUM_MASK_UPDATES}" ;;
esac

LOGDIR="${REPO_ROOT}/logs"
[[ -n "$LOG_SUBDIR" ]] && LOGDIR="${LOGDIR}/${LOG_SUBDIR}"
mkdir -p "$LOGDIR"

# Fall back to config default (seed=5) when no --seed is given.
if [[ -z "$SEEDS" ]]; then
    SEEDS="5"
fi

# Convert comma-separated seeds to an array.
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"

cd "$REPO_ROOT/vision"

for SEED in "${SEED_ARRAY[@]}"; do
    if [[ ${#SEED_ARRAY[@]} -gt 1 ]]; then
        run_name="${run_base}_seed${SEED}"
    else
        run_name="${run_base}"
    fi
    LOG_FILE="${LOGDIR}/${run_name}.out"

    echo "Sparsifier : $SPARSIFIER"
    echo "Model/Task : $MODEL / $TASK ($BENCHMARK)"
    echo "Seed       : $SEED"
    echo "Run name   : $run_name"
    echo "Log file   : $LOG_FILE"
    echo

    python train_st.py \
        --benchmark "$BENCHMARK" \
        --model "$MODEL" \
        --task "$TASK" \
        --sparsifier "$SPARSIFIER" \
        --sparsity "$SPARSITY" \
        --pruning-ratio "$PRUNING_RATIO" \
        --num-mask-updates "$NUM_MASK_UPDATES" \
        --seed "$SEED" \
        > "$LOG_FILE" 2>&1
done
