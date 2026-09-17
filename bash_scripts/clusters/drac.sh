# Digital Research Alliance of Canada (rorqual, narval, beluga, cedar, graham, ...).
# Shared module system and a local wheelhouse; compute nodes have no internet.

# Unversioned cuda picks the site default; versions differ per cluster.
FIRE_MODULES="${FIRE_MODULES:-StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/24.0.0 cuda}"

fire_load_modules() {
    # shellcheck disable=SC2086
    module load $FIRE_MODULES 2>/dev/null || {
        echo "WARNING: could not load modules: $FIRE_MODULES" >&2
        echo "         set FIRE_MODULES for this cluster" >&2
    }
}

fire_create_venv() {
    # --no-download keeps pip/setuptools from reaching the internet.
    virtualenv --no-download "$1" 2>/dev/null || python3 -m venv "$1"
}

fire_pip() {
    # The wheelhouse holds site-built wheels; compute nodes cannot reach PyPI.
    pip install --no-index "$@"
}

# Alliance docs prefer per-task GPU requests, and reject --mem for multi-task jobs.
FIRE_SBATCH_GPU=(--gpus-per-task=1)
FIRE_SBATCH_MEM=(--mem-per-cpu=8G)

# Compute nodes have no route to the internet, so runs log locally and are
# pushed afterwards with sync_wandb.sh from a login node.
FIRE_WANDB_MODE="${FIRE_WANDB_MODE:-offline}"
