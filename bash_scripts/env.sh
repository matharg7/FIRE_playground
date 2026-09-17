# Shared environment for every FIRE script. Source this; do not execute it.
#
# Resolves the repo, picks a cluster profile, and exports the FIRE_* paths that
# the other scripts use. Nothing here hardcodes a path or a site: override any
# value by exporting it before sourcing.
#
#   FIRE_REPO_ROOT   the checkout (auto-detected)
#   FIRE_CLUSTER     profile name (auto-detected; see clusters/)
#   FIRE_WORK        writable scratch space for everything generated
#   FIRE_VENV        python environment
#   FIRE_DATA_DIR    tokenized .bin datasets
#   FIRE_OUTPUT_DIR  run outputs (checkpoints, DONE markers)
#   FIRE_WANDB_DIR   W&B run directories
#   FIRE_CACHE_DIR   HuggingFace / tiktoken caches
#   FIRE_WANDB_MODE  offline (sync later) | online (log live) | disabled

# --- repo root --------------------------------------------------------------
# sbatch copies the submitted script into a spool directory, so BASH_SOURCE is
# only trustworthy outside a batch job.
if [[ -z "${FIRE_REPO_ROOT:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/language/train_sparse.py" ]]; then
        FIRE_REPO_ROOT="$SLURM_SUBMIT_DIR"
    else
        FIRE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi
if [[ ! -f "$FIRE_REPO_ROOT/language/train_sparse.py" ]]; then
    echo "ERROR: FIRE_REPO_ROOT does not look like the repo: $FIRE_REPO_ROOT" >&2
    echo "       export FIRE_REPO_ROOT=/path/to/FIRE_playground" >&2
    return 1 2>/dev/null || exit 1
fi
export FIRE_REPO_ROOT

# --- which cluster ----------------------------------------------------------
# CC_CLUSTER is set on every Digital Research Alliance machine. Elsewhere fall
# back to the hostname with trailing digits stripped (login3 -> login).
if [[ -z "${FIRE_CLUSTER:-}" ]]; then
    if [[ -n "${CC_CLUSTER:-}" ]]; then
        FIRE_CLUSTER="$CC_CLUSTER"
    else
        FIRE_CLUSTER="$(hostname -s 2>/dev/null | sed 's/[0-9]*$//')"
    fi
fi
export FIRE_CLUSTER

_fire_profile="$FIRE_REPO_ROOT/bash_scripts/clusters/${FIRE_CLUSTER}.sh"
if [[ ! -f "$_fire_profile" ]]; then
    # An unknown Alliance cluster still shares the module system and wheelhouse.
    if [[ -n "${CC_CLUSTER:-}" ]]; then
        _fire_profile="$FIRE_REPO_ROOT/bash_scripts/clusters/drac.sh"
    else
        _fire_profile="$FIRE_REPO_ROOT/bash_scripts/clusters/default.sh"
    fi
fi
export FIRE_PROFILE="$_fire_profile"

# --- paths ------------------------------------------------------------------
# Generated data goes somewhere large and writable: $SCRATCH where a site
# provides it, otherwise a .work directory inside the checkout.
FIRE_WORK="${FIRE_WORK:-${SCRATCH:-$FIRE_REPO_ROOT/.work}/fire}"
FIRE_VENV="${FIRE_VENV:-$FIRE_WORK/venv}"
FIRE_DATA_DIR="${FIRE_DATA_DIR:-$FIRE_WORK/data}"
FIRE_OUTPUT_DIR="${FIRE_OUTPUT_DIR:-$FIRE_WORK/output}"
FIRE_WANDB_DIR="${FIRE_WANDB_DIR:-$FIRE_WORK/wandb}"
FIRE_CACHE_DIR="${FIRE_CACHE_DIR:-$FIRE_WORK/cache}"
export FIRE_WORK FIRE_VENV FIRE_DATA_DIR FIRE_OUTPUT_DIR FIRE_WANDB_DIR FIRE_CACHE_DIR

export HF_HOME="${HF_HOME:-$FIRE_CACHE_DIR/huggingface}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$FIRE_CACHE_DIR/tiktoken}"

# --- profile defaults, overridable by the profile ---------------------------
fire_load_modules() { :; }                       # sites without modules need none
fire_create_venv() { python3 -m venv "$1"; }
fire_pip() { pip install "$@"; }
FIRE_ACCOUNT="${FIRE_ACCOUNT:-}"                 # sbatch --account, if the site needs one
FIRE_SBATCH_GPU=(--gres=gpu:1)                   # how to ask for one GPU
FIRE_SBATCH_MEM=(--mem=64G)                      # how to ask for memory
FIRE_SBATCH_EXTRA=()                             # anything else the site requires

# shellcheck disable=SC1090
source "$FIRE_PROFILE"

# Per-user overrides (account, W&B mode, work dir, modules). Gitignored.
if [[ -f "$FIRE_REPO_ROOT/bash_scripts/local.sh" ]]; then
    # shellcheck disable=SC1091
    source "$FIRE_REPO_ROOT/bash_scripts/local.sh"
fi

# offline | online | disabled. The cluster profile picks the sensible default
# (offline where compute nodes have no internet), local.sh overrides it, and an
# exported FIRE_WANDB_MODE beats both.
FIRE_WANDB_MODE="${FIRE_WANDB_MODE:-offline}"
export FIRE_WANDB_MODE

fire_activate() {
    if [[ ! -f "$FIRE_VENV/bin/activate" ]]; then
        echo "ERROR: no environment at $FIRE_VENV -- run bash_scripts/build.sh env" >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$FIRE_VENV/bin/activate"
    export PYTHONNOUSERSITE=1
}

fire_show_env() {
    echo "cluster    : $FIRE_CLUSTER  (profile: $(basename "$FIRE_PROFILE"))"
    echo "repo       : $FIRE_REPO_ROOT"
    echo "work       : $FIRE_WORK"
    echo "venv       : $FIRE_VENV"
    echo "data       : $FIRE_DATA_DIR"
    echo "output     : $FIRE_OUTPUT_DIR"
    echo "wandb      : $FIRE_WANDB_DIR"
    echo "account    : ${FIRE_ACCOUNT:-<none>}"
    echo "wandb      : $FIRE_WANDB_MODE"
}
