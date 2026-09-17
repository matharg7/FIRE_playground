# Generic profile: no module system, packages from PyPI.
# Used when the site is unknown. Copy this to <cluster>.sh and adjust.

fire_load_modules() { :; }

fire_create_venv() {
    python3 -m venv "$1"
}

fire_pip() {
    pip install "$@"
}

# Most schedulers accept these; override in a site profile if not.
FIRE_SBATCH_GPU=(--gres=gpu:1)
FIRE_SBATCH_MEM=(--mem=64G)

# Assume compute nodes can reach wandb.ai; set offline in local.sh if not.
FIRE_WANDB_MODE="${FIRE_WANDB_MODE:-online}"
