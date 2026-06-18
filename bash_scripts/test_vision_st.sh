#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.env/bin/activate"
# if [ -f "$SCRIPT_DIR/../.env/bin/activate" ]; then     --> running on personal not alliancecan
#     source "$SCRIPT_DIR/../.env/bin/activate"
# fi
cd "$SCRIPT_DIR/../vision"

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

python train_st.py \
    --benchmark continual \
    --model RESNET18 \
    --task CIFAR10 \
    --sparsifier rigl \
    --sparsity 0.9 \
    --pruning-ratio 0.3 \
    --seed 0 \
    --disable-wandb False
