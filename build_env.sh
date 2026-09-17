#!/bin/bash
# Build the Python virtual environment used by the vision training scripts.
#
# Recreates $SCRATCH/fire-env, the venv that bash_scripts/run_vision_st.sh
# expects. Scratch is purged periodically, so this script exists to rebuild
# the environment from vision/requirements.txt without guesswork.
#
# Usage:
#   ./build_env.sh                      # build $SCRATCH/fire-env
#   ./build_env.sh --force              # delete and rebuild an existing venv
#   ./build_env.sh --venv $SCRATCH/foo  # build somewhere else
#   ./build_env.sh --requirements vision/requirements.txt
#
# Flags accept `--flag value` or `--flag=value`.
#
# Run this on a login node (it is just a download + unpack), or inside an
# interactive job if you would rather not touch the login node at all. For
# I/O-heavy workflows the same steps work against $SLURM_TMPDIR inside a job.

set -euo pipefail

# Script lives at the repo root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${SCRATCH:?SCRATCH is not set}/fire-env"
REQUIREMENTS="$REPO_ROOT/vision/requirements.txt"
FORCE="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)           VENV="${2:?--venv needs an argument}"; shift 2 ;;
        --venv=*)         VENV="${1#*=}"; shift ;;
        --requirements|--reqs)
                          REQUIREMENTS="${2:?--requirements needs an argument}"; shift 2 ;;
        --requirements=*|--reqs=*)
                          REQUIREMENTS="${1#*=}"; shift ;;
        --force)          FORCE="true"; shift ;;
        -h|--help)        sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)                echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

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

# The venv holds several GB of wheels (torch alone is large), so check the
# scratch quota before writing.
if command -v diskusage_report >/dev/null 2>&1; then
    echo "==> Current disk usage"
    diskusage_report
fi

# Same module set that run_vision_st.sh loads, so the venv is built against
# the environment it will run under. python/3.11.5 is the version recorded in
# the original venv's pyvenv.cfg.
echo "==> Loading modules"
module load StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/24.0.0 cuda/13.2

echo "==> Creating venv: $VENV"
virtualenv --no-download "$VENV"

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Use exactly the venv's packages, not the host's ~/.local.
export PYTHONNOUSERSITE=1

echo "==> Upgrading pip from the wheelhouse"
pip install --no-index --upgrade pip

# --no-index keeps pip on the Alliance wheelhouse instead of PyPI; every
# package in the original venv carried a +computecanada tag.
echo "==> Installing from $REQUIREMENTS"
pip install --no-index -r "$REQUIREMENTS"

echo "==> Installed packages"
pip list

cat <<MSG

Done. Activate it with:

    module load StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/24.0.0 cuda/13.2
    source $VENV/bin/activate

MSG
