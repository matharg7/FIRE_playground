#!/bin/bash
# One-off: decode TinyImageNet's 110k JPEGs into the tensor cache that
# vision/task.py (and gpu_data.py, which uploads it to the GPU) expects.
#
#     cd <repo> && sbatch tamia/build_cache.slurm
#
# Why it is a job and not part of the sweep:
#   - Too heavy for a login node.
#   - Left to task.py it happens lazily on first use, which would have every node
#     in a sweep array decode the same dataset at once inside a GPU allocation.
#   run_sweep.slurm refuses to start until this cache exists.
#
# This needs no GPU, so it targets tamIA's CPU-only nodes. The raw datasets must
# already be downloaded -- that needs internet, so it happens on a login node via
# ./build_env.sh before this runs.
#
#SBATCH --job-name=fire-cache
#SBATCH --account=aip-yani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/tamia_build_cache_%j.out
#SBATCH --error=slurm_logs/tamia_build_cache_%j.err

set -euo pipefail

if [[ -n "${FIRE_REPO_ROOT:-}" ]]; then
    :
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    FIRE_REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    FIRE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export FIRE_REPO_ROOT

CACHE_SCRIPT="$FIRE_REPO_ROOT/vision/scripts/build_dataset_cache.py"
if [[ ! -f "$CACHE_SCRIPT" ]]; then
    echo "ERROR: '$FIRE_REPO_ROOT' is not the repo root (no $CACHE_SCRIPT)." >&2
    echo "       Submit from the repo root:  cd <repo> && sbatch tamia/build_cache.slurm" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$FIRE_REPO_ROOT/tamia/env.sh" || exit 1

# The raw data has to be there already: this node has no internet.
if [[ ! -d "$FIRE_DATA_DIR/tiny-imagenet-200/train" ]]; then
    cat >&2 <<MSG
ERROR: raw TinyImageNet not found under $FIRE_DATA_DIR

       Downloading needs internet, which compute nodes do not have. Fetch it
       from a LOGIN node first:
           cd $FIRE_REPO_ROOT && ./build_env.sh --skip-cache
MSG
    exit 1
fi

echo "==> Decoding TinyImageNet into $FIRE_DATA_DIR"
srun python "$CACHE_SCRIPT"

CACHE_VERSION="$(sed -n 's/^[[:space:]]*CACHE_VERSION[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
                     "$FIRE_REPO_ROOT/vision/task.py" | head -1)"
echo "==> Done: $FIRE_DATA_DIR/tiny-imagenet-200-decoded-v${CACHE_VERSION:-1}.pt"
