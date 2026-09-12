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
#   ./run_vision_st.sh --drop-fraction-schedule per_task   # RigL/SET df schedule
#   ./run_vision_st.sh --use-cosine-lr True --cosine-eta-min 0.0
#   ./run_vision_st.sh --use-cosine-lr True --cosine-t-max-epochs 200
#   ./run_vision_st.sh --wandb-project my_proj --log-name my_run
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
DROP_FRACTION_SCHEDULE="global"
USE_COSINE_LR="False"
COSINE_T_MAX_EPOCHS="0"
COSINE_ETA_MIN="0.0"
WANDB_PROJECT=""
LOG_NAME=""
MODEL="RESNET18"
TASK="CIFAR10"
BENCHMARK="continual"
LOG_SUBDIR=""
GPU="0"

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
        --drop-fraction-schedule|--drop_fraction_schedule)
                                 DROP_FRACTION_SCHEDULE="${2:?--drop-fraction-schedule needs an argument}"; shift 2 ;;
        --drop-fraction-schedule=*|--drop_fraction_schedule=*)
                                 DROP_FRACTION_SCHEDULE="${1#*=}"; shift ;;
        --use-cosine-lr|--use_cosine_lr)
                                 USE_COSINE_LR="${2:?--use-cosine-lr needs an argument}"; shift 2 ;;
        --use-cosine-lr=*|--use_cosine_lr=*)
                                 USE_COSINE_LR="${1#*=}"; shift ;;
        --cosine-t-max-epochs|--cosine-T-max-epochs|--cosine_T_max_epochs)
                                 COSINE_T_MAX_EPOCHS="${2:?--cosine-t-max-epochs needs an argument}"; shift 2 ;;
        --cosine-t-max-epochs=*|--cosine-T-max-epochs=*|--cosine_T_max_epochs=*)
                                 COSINE_T_MAX_EPOCHS="${1#*=}"; shift ;;
        --cosine-eta-min|--cosine_eta_min)
                                 COSINE_ETA_MIN="${2:?--cosine-eta-min needs an argument}"; shift 2 ;;
        --cosine-eta-min=*|--cosine_eta_min=*)
                                 COSINE_ETA_MIN="${1#*=}"; shift ;;
        --wandb-project|--wandb_project)
                                 WANDB_PROJECT="${2:?--wandb-project needs an argument}"; shift 2 ;;
        --wandb-project=*|--wandb_project=*)
                                 WANDB_PROJECT="${1#*=}"; shift ;;
        --log-name|--log_name)   LOG_NAME="${2:?--log-name needs an argument}"; shift 2 ;;
        --log-name=*|--log_name=*)
                                 LOG_NAME="${1#*=}"; shift ;;
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
        -h|--help)               sed -n '4,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)                       echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

case "$DROP_FRACTION_SCHEDULE" in
    global|per_task|constant) ;;
    *) echo "ERROR: --drop-fraction-schedule must be global, per_task or constant (got: $DROP_FRACTION_SCHEDULE)" >&2; exit 1 ;;
esac

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

# Run name for the log file (the W&B run name is built inside train_st.py).
# rigl/set carry the drop-fraction schedule, matching the _df_ suffix that
# build_run_name() adds; gmp/static have no drop-fraction scheduler.
case "$SPARSIFIER" in
    dense)  run_name="${MODEL}_${TASK}_dense" ;;
    static) run_name="${MODEL}_${TASK}_static_s${SPARSITY}" ;;
    gmp)    run_name="${MODEL}_${TASK}_gmp_s${SPARSITY}_nmu${NUM_MASK_UPDATES}" ;;
    *)      run_name="${MODEL}_${TASK}_${SPARSIFIER}_s${SPARSITY}_pr${PRUNING_RATIO}_nmu${NUM_MASK_UPDATES}_df${DROP_FRACTION_SCHEDULE}" ;;
esac

# Keep cosine-LR and warmup-LR runs of the same config in separate log files.
case "$USE_COSINE_LR" in
    True|true|t|y|yes|1)
        run_name="${run_name}_coslr"
        [[ "$COSINE_T_MAX_EPOCHS" != "0" ]] && run_name="${run_name}T${COSINE_T_MAX_EPOCHS}"
        ;;
esac

# --log-name overrides the derived log file name (W&B run name is unaffected).
[[ -n "$LOG_NAME" ]] && run_name="$LOG_NAME"

LOGDIR="${REPO_ROOT}/logs"
[[ -n "$LOG_SUBDIR" ]] && LOGDIR="${LOGDIR}/${LOG_SUBDIR}"
mkdir -p "$LOGDIR"
LOG_FILE="${LOGDIR}/${run_name}.out"

echo "Sparsifier : $SPARSIFIER"
echo "Model/Task : $MODEL / $TASK ($BENCHMARK)"
echo "Drop frac. : $DROP_FRACTION_SCHEDULE"
echo "LR sched.  : cosine=$USE_COSINE_LR T_max=$COSINE_T_MAX_EPOCHS eta_min=$COSINE_ETA_MIN"
echo "Run name   : $run_name"
echo "Log file   : $LOG_FILE"
echo

cd "$REPO_ROOT/vision"
python train_st.py \
    --benchmark "$BENCHMARK" \
    --model "$MODEL" \
    --task "$TASK" \
    --sparsifier "$SPARSIFIER" \
    --sparsity "$SPARSITY" \
    --pruning-ratio "$PRUNING_RATIO" \
    --num-mask-updates "$NUM_MASK_UPDATES" \
    --drop-fraction-schedule "$DROP_FRACTION_SCHEDULE" \
    --use-cosine-lr "$USE_COSINE_LR" \
    --cosine-T-max-epochs "$COSINE_T_MAX_EPOCHS" \
    --cosine-eta-min "$COSINE_ETA_MIN" \
    --wandb-project "$WANDB_PROJECT" \
    > "$LOG_FILE" 2>&1
