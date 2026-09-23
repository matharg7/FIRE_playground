#!/bin/bash
# Shared environment for every tamIA script in this directory.
#
# Source it, do not execute it:
#
#     source tamia/env.sh                  # modules + venv only
#     source tamia/env.sh <sweep-name>     # ... and the offline W&B block for that sweep
#
# What it sets up:
#   1. Environment modules and the virtualenv at $FIRE_VENV ($SCRATCH/fire-env).
#   2. Offline W&B, because tamIA's compute nodes have no internet: WANDB_MODE=offline
#      plus per-sweep WANDB_DIR, so `tamia/sync_wandb.sh <sweep>` can upload exactly
#      the runs of one sweep afterwards.
#
# Everything is overridable from the environment, because the module versions below
# are the ones nibi carries and tamIA may not carry the same. Check with:
#     module avail python ; module avail cuda
# and export e.g. FIRE_PYTHON_MODULE=python/3.11.9 before sourcing, or just edit the
# defaults here once you know what tamIA has.

# --- Guard: must be sourced ---------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this file, do not run it:  source tamia/env.sh" >&2
    exit 1
fi

# --- Repo root ----------------------------------------------------------------
# Inside a Slurm job the batch script is spooled to /var/spool/slurmd/..., so
# BASH_SOURCE is useless there and the caller exports FIRE_REPO_ROOT instead.
if [[ -z "${FIRE_REPO_ROOT:-}" ]]; then
    FIRE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export FIRE_REPO_ROOT

# --- Modules ------------------------------------------------------------------
FIRE_STDENV_MODULE="${FIRE_STDENV_MODULE:-StdEnv/2023}"
FIRE_PYTHON_MODULE="${FIRE_PYTHON_MODULE:-python/3.11.5}"
FIRE_SCIPY_MODULE="${FIRE_SCIPY_MODULE:-scipy-stack/2026a}"
FIRE_ARROW_MODULE="${FIRE_ARROW_MODULE:-arrow/24.0.0}"
FIRE_CUDA_MODULE="${FIRE_CUDA_MODULE:-cuda/13.2}"

FIRE_MODULES=(
    "$FIRE_STDENV_MODULE"
    "$FIRE_PYTHON_MODULE"
    "$FIRE_SCIPY_MODULE"
    "$FIRE_ARROW_MODULE"
    "$FIRE_CUDA_MODULE"
)
export FIRE_MODULES

echo "==> Loading modules: ${FIRE_MODULES[*]}"
if ! module load "${FIRE_MODULES[@]}"; then
    cat >&2 <<MSG
ERROR: could not load one of: ${FIRE_MODULES[*]}

       These versions are what nibi carries; tamIA may differ. List what is
       available and override the mismatched one, e.g.

           module avail python
           export FIRE_PYTHON_MODULE=python/3.11.9
           source tamia/env.sh

       StdEnv/2023 is documented as tamIA's standard environment, so it is
       usually one of the other four that needs adjusting.
MSG
    return 1
fi

# --- Virtualenv ---------------------------------------------------------------
# Same path as on nibi, per the porting brief. Built by ./build_env.sh on a
# login node (the wheelhouse and the dataset downloads both need internet).
export FIRE_VENV="${FIRE_VENV:-${SCRATCH:?SCRATCH is not set}/fire-env}"

if [[ ! -f "$FIRE_VENV/bin/activate" ]]; then
    cat >&2 <<MSG
ERROR: virtualenv not found at: $FIRE_VENV

       Build it on a LOGIN node (compute nodes have no internet):
           cd $FIRE_REPO_ROOT && ./build_env.sh
MSG
    return 1
fi

echo "==> Activating venv: $FIRE_VENV"
# shellcheck disable=SC1091
source "$FIRE_VENV/bin/activate"

# Use exactly the venv's packages, not the host's ~/.local.
export PYTHONNOUSERSITE=1

# --- Datasets -----------------------------------------------------------------
# vision/task.py resolves FIRE_DATA_DIR, else $SCRATCH/datasets.
export FIRE_DATA_DIR="${FIRE_DATA_DIR:-$SCRATCH/datasets}"

# --- Scratch layout for tamIA state -------------------------------------------
# Trial claim/done markers live here, never in the repo: a 180-trial sweep adds
# hundreds of tiny files, and /project is inode-limited.
export FIRE_STATE_ROOT="${FIRE_STATE_ROOT:-$SCRATCH/fire-tamia/state}"

# --- Where the offline W&B runs go --------------------------------------------
# Their own top-level directory inside the repo, one sub-directory per sweep, so
# a sweep's logs sit beside the code that produced them and survive the periodic
# $SCRATCH purge. The claim/done markers above deliberately do NOT move here --
# those are thousands of empty files, which is what /project's inode quota
# minds, while a sweep's runs are a few thousand real ones.
#
# Override it to keep them on scratch after all:
#     export FIRE_WANDB_ROOT=$SCRATCH/wandb
export FIRE_WANDB_ROOT="${FIRE_WANDB_ROOT:-$FIRE_REPO_ROOT/wandb_offline}"

# --- Offline W&B --------------------------------------------------------------
# Only configured when a sweep name is given, so that sourcing this file for an
# interactive poke-around does not silently force offline mode on a login node
# (where online logging would work fine).
_fire_sweep="${1:-}"
if [[ -n "$_fire_sweep" ]]; then
    # tamIA compute nodes cannot reach api.wandb.ai. Runs are written to disk and
    # uploaded later with tamia/sync_wandb.sh from a login node.
    export WANDB_MODE="${WANDB_MODE:-offline}"

    # One directory per sweep so sync_wandb.sh can scope an upload to one sweep.
    # wandb appends its own "wandb/" component, so runs land in
    #   $FIRE_WANDB_ROOT/<sweep>/wandb/offline-run-<timestamp>-<id>/
    export WANDB_DIR="${WANDB_DIR:-$FIRE_WANDB_ROOT/$_fire_sweep}"
    # Cache, artifacts and config stay on scratch: regenerable working files
    # rather than run records, and the cache can grow large.
    export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$SCRATCH/wandb/cache}"
    export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$SCRATCH/wandb/artifacts}"
    export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$SCRATCH/wandb/config}"
    mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR"

    # Offline runs are not attached to a W&B sweep object, so there is no sweep
    # page to group them on once synced. WANDB_RUN_GROUP and WANDB_TAGS are read
    # by wandb.init() straight from the environment -- no change to train_st.py --
    # and give back the grouping: filter the project by group == <sweep name>.
    export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$_fire_sweep}"
    export WANDB_TAGS="${WANDB_TAGS:-tamia,$_fire_sweep}"
    export WANDB_ENTITY="${WANDB_ENTITY:-ucalgary}"

    # No WANDB_API_KEY here: offline mode needs none, and the compute nodes could
    # not use it anyway. sync_wandb.sh reads ~/.wandb_token on the login node.
fi
unset _fire_sweep

echo "==> Environment ready (repo: $FIRE_REPO_ROOT)"
