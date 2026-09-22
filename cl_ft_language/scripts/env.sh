# Shared environment for every cl_ft_language script. Source this; do not execute it.
#
# Self-contained: does not source anything under bash_scripts/. Exports the
# TRACE_* paths the other scripts use; override any of them by exporting it
# before sourcing, or in scripts/local.sh (gitignored).
#
#   TRACE_ROOT        cl_ft_language/ in the checkout (auto-detected)
#   TRACE_WORK        shared scratch root for generated files ($SCRATCH/fire)
#   TRACE_VENV        python environment ($TRACE_WORK/venv-trace)
#   TRACE_DATA_DIR    TRACE benchmark json data
#   TRACE_OUTPUT_DIR  run outputs (checkpoints, R-matrices, DONE markers)
#   TRACE_WANDB_DIR   W&B run directories (shared with the language runs)
#   HF_HOME           HuggingFace cache (shared with the language runs)
#   TRACE_WANDB_MODE  offline (sync later) | online | disabled

# --- locate cl_ft_language/ ---------------------------------------------------
# sbatch copies the submitted script into a spool directory, so BASH_SOURCE is
# only trustworthy outside a batch job.
if [[ -z "${TRACE_ROOT:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/cl_ft_language/scripts/env.sh" ]]; then
        TRACE_ROOT="$SLURM_SUBMIT_DIR/cl_ft_language"
    elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/env.sh" ]]; then
        TRACE_ROOT="$SLURM_SUBMIT_DIR"
    else
        TRACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi
if [[ ! -f "$TRACE_ROOT/scripts/env.sh" || ! -d "$TRACE_ROOT/trace" ]]; then
    echo "ERROR: TRACE_ROOT does not look like cl_ft_language/: $TRACE_ROOT" >&2
    echo "       export TRACE_ROOT=/path/to/FIRE_playground/cl_ft_language" >&2
    return 1 2>/dev/null || exit 1
fi
export TRACE_ROOT
export TRACE_REPO_ROOT="$(cd "$TRACE_ROOT/.." && pwd)"

# --- site defaults (Digital Research Alliance) --------------------------------
# CC_CLUSTER is set on every Alliance machine. Their compute nodes have no
# internet, so pip uses the local wheelhouse and W&B logs offline.
TRACE_CLUSTER="${CC_CLUSTER:-$(hostname -s 2>/dev/null | sed 's/[0-9]*$//')}"
export TRACE_CLUSTER

case "$TRACE_CLUSTER" in
    rorqual) _trace_cuda="cuda/13.2" ;;   # the version the language venv was built against
    *)       _trace_cuda="cuda" ;;        # unversioned: the site default
esac
TRACE_MODULES="${TRACE_MODULES:-StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/24.0.0 $_trace_cuda}"
TRACE_ACCOUNT="${TRACE_ACCOUNT:-}"
TRACE_WANDB_MODE="${TRACE_WANDB_MODE:-}"

# Per-user overrides (account, W&B mode, work dir, modules). Gitignored.
if [[ -f "$TRACE_ROOT/scripts/local.sh" ]]; then
    # shellcheck disable=SC1091
    source "$TRACE_ROOT/scripts/local.sh"
fi

if [[ -n "${CC_CLUSTER:-}" ]]; then
    trace_load_modules() {
        # shellcheck disable=SC2086
        module load $TRACE_MODULES 2>/dev/null || {
            echo "WARNING: could not load modules: $TRACE_MODULES" >&2
            echo "         set TRACE_MODULES in scripts/local.sh" >&2
        }
    }
    trace_create_venv() { virtualenv --no-download "$1" 2>/dev/null || python3 -m venv "$1"; }
    trace_pip() { pip install --no-index "$@"; }
    TRACE_WANDB_MODE="${TRACE_WANDB_MODE:-offline}"
else
    trace_load_modules() { :; }
    trace_create_venv() { python3 -m venv "$1"; }
    trace_pip() { pip install "$@"; }
    TRACE_WANDB_MODE="${TRACE_WANDB_MODE:-online}"
fi
export TRACE_MODULES TRACE_ACCOUNT TRACE_WANDB_MODE

# --- paths ----------------------------------------------------------------------
TRACE_WORK="${TRACE_WORK:-${SCRATCH:-$TRACE_REPO_ROOT/.work}/fire}"
TRACE_VENV="${TRACE_VENV:-$TRACE_WORK/venv-trace}"
TRACE_DATA_DIR="${TRACE_DATA_DIR:-$TRACE_WORK/data/trace}"
TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-$TRACE_WORK/output/trace}"
TRACE_WANDB_DIR="${TRACE_WANDB_DIR:-$TRACE_WORK/wandb}"
export TRACE_WORK TRACE_VENV TRACE_DATA_DIR TRACE_OUTPUT_DIR TRACE_WANDB_DIR
export HF_HOME="${HF_HOME:-$TRACE_WORK/cache/huggingface}"
export NLTK_DATA="${NLTK_DATA:-$TRACE_WORK/cache/nltk_data}"

trace_activate() {
    if [[ ! -f "$TRACE_VENV/bin/activate" ]]; then
        echo "ERROR: no environment at $TRACE_VENV -- run cl_ft_language/scripts/build_env.sh env" >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$TRACE_VENV/bin/activate"
    export PYTHONNOUSERSITE=1
}

trace_show_env() {
    echo "cluster    : $TRACE_CLUSTER"
    echo "modules    : $TRACE_MODULES"
    echo "root       : $TRACE_ROOT"
    echo "work       : $TRACE_WORK"
    echo "venv       : $TRACE_VENV"
    echo "data       : $TRACE_DATA_DIR"
    echo "output     : $TRACE_OUTPUT_DIR"
    echo "wandb dir  : $TRACE_WANDB_DIR"
    echo "wandb mode : $TRACE_WANDB_MODE"
    echo "hf cache   : $HF_HOME"
    echo "account    : ${TRACE_ACCOUNT:-<none>}"
}
