#!/bin/bash
# Build the Python virtual environment used by the vision training scripts,
# then pre-stage the datasets they read.
#
# Recreates $SCRATCH/fire-env, the venv that bash_scripts/run_vision_st.sh
# expects. Scratch is purged periodically, so this script exists to rebuild
# the environment from vision/requirements.txt without guesswork.
#
# That same purge takes the dataset cache with it, so this also downloads
# CIFAR-10, CIFAR-100 and TinyImageNet into $FIRE_DATA_DIR (else
# $SCRATCH/datasets), via vision/scripts/build_dataset_cache.py. Compute nodes
# may have no internet access, which is why the fetch belongs here and not in
# a training job.
#
# Usage:
#   ./build_env.sh                      # build $SCRATCH/fire-env, then datasets
#   ./build_env.sh --force              # delete and rebuild an existing venv
#   ./build_env.sh --skip-datasets      # venv only
#   ./build_env.sh --datasets-only      # datasets only, into an existing venv
#   ./build_env.sh --venv $SCRATCH/foo  # build somewhere else
#   ./build_env.sh --data-dir $SCRATCH/datasets
#   ./build_env.sh --requirements vision/requirements.txt
#
# Flags accept `--flag value` or `--flag=value`.
#
# Run this on a login node (it is just a download + unpack), or inside an
# interactive job if you would rather not touch the login node at all. For
# I/O-heavy workflows the same steps work against $SLURM_TMPDIR inside a job.
#
# One thing is deliberately not done here: decoding TinyImageNet's 110k JPEGs
# into the tensor cache. That is too heavy for a login node, so run it in an
# allocation afterwards:
#   salloc --account=<account> --time=0:30:00 --cpus-per-task=8 --mem=16G
#   python vision/scripts/build_dataset_cache.py

set -euo pipefail

# Script lives at the repo root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${SCRATCH:?SCRATCH is not set}/fire-env"
REQUIREMENTS="$REPO_ROOT/vision/requirements.txt"
DATA_DIR=""
FORCE="false"
SKIP_DATASETS="false"
DATASETS_ONLY="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)           VENV="${2:?--venv needs an argument}"; shift 2 ;;
        --venv=*)         VENV="${1#*=}"; shift ;;
        --requirements|--reqs)
                          REQUIREMENTS="${2:?--requirements needs an argument}"; shift 2 ;;
        --requirements=*|--reqs=*)
                          REQUIREMENTS="${1#*=}"; shift ;;
        --data-dir)       DATA_DIR="${2:?--data-dir needs an argument}"; shift 2 ;;
        --data-dir=*)     DATA_DIR="${1#*=}"; shift ;;
        --force)          FORCE="true"; shift ;;
        --skip-datasets)  SKIP_DATASETS="true"; shift ;;
        --datasets-only)  DATASETS_ONLY="true"; shift ;;
        -h|--help)        sed -n '2,35p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)                echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "$SKIP_DATASETS" == "true" && "$DATASETS_ONLY" == "true" ]]; then
    echo "ERROR: --skip-datasets and --datasets-only are contradictory." >&2
    exit 1
fi

# The dataset cache location is resolved by vision/task.py; exporting the
# override here is what makes --data-dir reach it.
if [[ -n "$DATA_DIR" ]]; then
    export FIRE_DATA_DIR="$DATA_DIR"
fi

if [[ "$DATASETS_ONLY" == "true" ]]; then
    if [[ ! -f "$VENV/bin/activate" ]]; then
        echo "ERROR: no venv at $VENV. Drop --datasets-only to build it first." >&2
        exit 1
    fi
else
    if [[ ! -f "$REQUIREMENTS" ]]; then
        echo "ERROR: requirements file not found: $REQUIREMENTS" >&2
        exit 1
    fi

    if [[ -e "$VENV" ]]; then
        if [[ "$FORCE" != "true" ]]; then
            echo "ERROR: $VENV already exists. Re-run with --force to rebuild it." >&2
            exit 1
        fi
        echo "==> Removing existing venv: $VENV"
        rm -rf "$VENV"
    fi
fi

# The venv holds several GB of wheels (torch alone is large) and the datasets
# add a few more, so check the scratch quota before writing.
if command -v diskusage_report >/dev/null 2>&1; then
    echo "==> Current disk usage"
    diskusage_report
fi

# Same module set that run_vision_st.sh loads, so the venv is built against
# the environment it will run under. python/3.11.5 is the version recorded in
# the original venv's pyvenv.cfg.
echo "==> Loading modules"
module load StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/24.0.0 cuda/13.2

if [[ "$DATASETS_ONLY" != "true" ]]; then
    echo "==> Creating venv: $VENV"
    virtualenv --no-download "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Use exactly the venv's packages, not the host's ~/.local.
export PYTHONNOUSERSITE=1

if [[ "$DATASETS_ONLY" != "true" ]]; then
    echo "==> Upgrading pip from the wheelhouse"
    pip install --no-index --upgrade pip

    # --no-index keeps pip on the Alliance wheelhouse instead of PyPI; every
    # package in the original venv carried a +computecanada tag.
    echo "==> Installing from $REQUIREMENTS"
    pip install --no-index -r "$REQUIREMENTS"

    echo "==> Installed packages"
    pip list
fi

if [[ "$SKIP_DATASETS" != "true" ]]; then
    # Needs the venv's torchvision, hence after the install. Re-running is
    # cheap: every dataset that is already staged is skipped.
    echo "==> Downloading datasets"
    python "$REPO_ROOT/vision/scripts/build_dataset_cache.py" --download
fi

cat <<MSG

Done. Activate it with:

    module load StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/24.0.0 cuda/13.2
    source $VENV/bin/activate

Then build the decoded TinyImageNet cache in an allocation (not on a login node):

    salloc --account=<account> --time=0:30:00 --cpus-per-task=8 --mem=16G
    python vision/scripts/build_dataset_cache.py

MSG
