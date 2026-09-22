#!/bin/bash
# Bring the vision codebase up on a cluster from scratch: virtual environment,
# raw datasets, decoded dataset cache.
#
# Every step is a check first and an action second, so the script is safe to
# re-run and safe to point at a half-finished setup:
#
#   1. venv     -- $SCRATCH/fire-env is created if absent; if it is already
#                  there, its packages are checked against
#                  vision/requirements.txt and anything missing or
#                  out-of-spec is installed from the Alliance wheelhouse.
#   2. datasets -- CIFAR-10, CIFAR-100 and TinyImageNet are looked for in
#                  $FIRE_DATA_DIR (else $SCRATCH/datasets); only the ones that
#                  are absent are downloaded. Compute nodes may have no
#                  internet, which is why the fetch belongs here and not in a
#                  training job.
#   3. cache    -- TinyImageNet's 110k JPEGs are decoded into the tensor cache
#                  that task.py (and gpu_data.py, which uploads it to the GPU)
#                  expects. This one is too heavy for a login node, so it runs
#                  directly only inside an allocation; from a login node pass
#                  --account to submit it as a short batch job, or run the
#                  printed salloc yourself.
#
# Scratch is purged periodically and takes the venv and the datasets with it,
# which is the situation this script exists to recover from.
#
# Usage -- run it from the repo root:
#   ./build_env.sh                          # venv + datasets, cache if possible
#   ./build_env.sh --account rrg-yani       # ... and submit the cache job
#   ./build_env.sh --force                  # delete and rebuild the venv
#   ./build_env.sh --skip-datasets          # venv only
#   ./build_env.sh --datasets-only          # datasets into an existing venv
#   ./build_env.sh --skip-cache             # download, but do not decode
#   ./build_env.sh --venv $SCRATCH/foo      # build somewhere else
#   ./build_env.sh --data-dir $SCRATCH/datasets
#   ./build_env.sh --requirements vision/requirements.txt
#
# Flags accept `--flag value` or `--flag=value`.
#
# Steps 1 and 2 are a download-and-unpack; run them on a login node, or inside
# an interactive job if you would rather not touch the login node at all. For
# I/O-heavy workflows the same steps work against $SLURM_TMPDIR inside a job.

set -euo pipefail

# Script lives at the repo root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${SCRATCH:?SCRATCH is not set}/fire-env"
REQUIREMENTS="$REPO_ROOT/vision/requirements.txt"
DATA_DIR=""
ACCOUNT=""
FORCE="false"
SKIP_DATASETS="false"
DATASETS_ONLY="false"
SKIP_CACHE="false"

MODULES=(StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/24.0.0 cuda/13.2)

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
        --account)        ACCOUNT="${2:?--account needs an argument}"; shift 2 ;;
        --account=*)      ACCOUNT="${1#*=}"; shift ;;
        --force)          FORCE="true"; shift ;;
        --skip-datasets)  SKIP_DATASETS="true"; shift ;;
        --datasets-only)  DATASETS_ONLY="true"; shift ;;
        --skip-cache)     SKIP_CACHE="true"; shift ;;
        -h|--help)        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
                              "${BASH_SOURCE[0]}"; exit 0 ;;
        *)                echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "$SKIP_DATASETS" == "true" && "$DATASETS_ONLY" == "true" ]]; then
    echo "ERROR: --skip-datasets and --datasets-only are contradictory." >&2
    exit 1
fi

# The dataset location is resolved by vision/task.py (FIRE_DATA_DIR, else
# $SCRATCH/datasets); exporting the override here is what makes --data-dir
# reach it, and mirroring the same precedence gives this script the path it
# needs for its own presence checks.
if [[ -n "$DATA_DIR" ]]; then
    export FIRE_DATA_DIR="$DATA_DIR"
fi
DATA_DIR="${FIRE_DATA_DIR:-$SCRATCH/datasets}"

CACHE_SCRIPT="$REPO_ROOT/vision/scripts/build_dataset_cache.py"
if [[ ! -f "$CACHE_SCRIPT" ]]; then
    echo "ERROR: $CACHE_SCRIPT not found -- run this script from the repo root." >&2
    exit 1
fi

# Read the cache version out of task.py rather than hardcoding it, so bumping
# TinyImageNet.CACHE_VERSION does not silently leave this script looking for a
# stale file. Parsed, not imported: the import pulls in torch.
CACHE_VERSION="$(sed -n 's/^[[:space:]]*CACHE_VERSION[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
                     "$REPO_ROOT/vision/task.py" | head -1)"
CACHE_FILE="$DATA_DIR/tiny-imagenet-200-decoded-v${CACHE_VERSION:-1}.pt"

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

    if [[ -e "$VENV" && "$FORCE" == "true" ]]; then
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
echo "==> Loading modules: ${MODULES[*]}"
module load "${MODULES[@]}"

# ---------------------------------------------------------------------------
# 1. Virtual environment
# ---------------------------------------------------------------------------
if [[ -f "$VENV/bin/activate" ]]; then
    echo "==> Reusing existing venv: $VENV"
else
    echo "==> Creating venv: $VENV"
    virtualenv --no-download "$VENV"

    echo "==> Upgrading pip from the wheelhouse"
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    export PYTHONNOUSERSITE=1
    pip install --no-index --upgrade pip
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Use exactly the venv's packages, not the host's ~/.local.
export PYTHONNOUSERSITE=1

# Verify the venv actually satisfies vision/requirements.txt before trusting
# it. A venv that survived a partial purge, or one built for an older
# requirements file, looks fine until a job fails an hour into the queue.
check_requirements() {
    python - "$REQUIREMENTS" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

try:
    from packaging.requirements import Requirement
except ModuleNotFoundError:          # packaging is not guaranteed in a bare venv
    from pip._vendor.packaging.requirements import Requirement

unsatisfied = []
with open(sys.argv[1]) as fh:
    for raw in fh:
        line = raw.split('#', 1)[0].strip()
        if not line or line.startswith('-'):
            continue
        req = Requirement(line)
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            installed = version(req.name)
        except PackageNotFoundError:
            unsatisfied.append((line, 'not installed'))
            continue
        # Alliance wheels carry a +computecanada local tag. PEP 440 ignores a
        # candidate's local label against a specifier that has none, so
        # `torch==2.9.1` still matches 2.9.1+computecanada.
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            unsatisfied.append((line, f'installed {installed}'))
        else:
            print(f"    ok       {req.name} {installed}")

for line, why in unsatisfied:
    print(f"    MISSING  {line}  ({why})")
sys.exit(1 if unsatisfied else 0)
PY
}

echo "==> Checking the venv against $REQUIREMENTS"
if check_requirements; then
    echo "    all requirements satisfied"
else
    echo "==> Installing requirements from the wheelhouse"
    # --no-index keeps pip on the Alliance wheelhouse instead of PyPI; every
    # package in the original venv carried a +computecanada tag.
    pip install --no-index -r "$REQUIREMENTS"

    echo "==> Re-checking"
    if ! check_requirements; then
        cat >&2 <<'MSG'
ERROR: the venv still does not satisfy the requirements after installing.
       The wheelhouse may not carry a wheel at the pinned version -- check
       `pip install --no-index <pkg>==<version>`, and if the wheel is genuinely
       absent, open a support ticket with the Alliance to have it added
       (https://docs.alliancecan.ca/wiki/Technical_support).
MSG
        exit 1
    fi
fi

if [[ "$SKIP_DATASETS" == "true" ]]; then
    echo "==> Skipping datasets (--skip-datasets)"
    echo
    echo "Done. Activate it with:"
    echo
    echo "    module load ${MODULES[*]}"
    echo "    source $VENV/bin/activate"
    echo
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Raw datasets
# ---------------------------------------------------------------------------
# One marker path per dataset, matching what task.py reads. Checking here
# rather than leaning on the script's own skip logic keeps the common case
# (everything already staged) free of a torch import and a network round trip.
echo "==> Checking datasets in $DATA_DIR"
missing_datasets=()
[[ -d "$DATA_DIR/cifar-10-batches-py" ]]  || missing_datasets+=(CIFAR10)
[[ -d "$DATA_DIR/cifar-100-python" ]]     || missing_datasets+=(CIFAR100)
if [[ ! -d "$DATA_DIR/tiny-imagenet-200/train" || ! -d "$DATA_DIR/tiny-imagenet-200/val" ]]; then
    missing_datasets+=(TinyImageNet)
fi

if [[ ${#missing_datasets[@]} -eq 0 ]]; then
    echo "    CIFAR10, CIFAR100, TinyImageNet all present"
else
    echo "    missing: ${missing_datasets[*]}"
    echo "==> Downloading ${missing_datasets[*]}"
    python "$CACHE_SCRIPT" --download --datasets "${missing_datasets[@]}"
fi

# ---------------------------------------------------------------------------
# 3. Decoded TinyImageNet cache
# ---------------------------------------------------------------------------
# Decoding 110k JPEGs is exactly the kind of work that does not belong on a
# login node, so it only runs here when we are already inside an allocation.
cache_job_script() {
    cat <<EOF
#!/bin/bash
#SBATCH --job-name=fire-dataset-cache
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=$REPO_ROOT/slurm_logs/build_dataset_cache_%j.out
set -euo pipefail
module load ${MODULES[*]}
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1
export FIRE_DATA_DIR="$DATA_DIR"
srun python "$CACHE_SCRIPT"
EOF
}

if [[ "$SKIP_CACHE" == "true" ]]; then
    echo "==> Skipping the decoded cache (--skip-cache)"
elif [[ -f "$CACHE_FILE" ]]; then
    echo "==> Decoded TinyImageNet cache already present: $CACHE_FILE"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Already in an allocation (salloc or sbatch): just do it.
    echo "==> Building the decoded TinyImageNet cache (job $SLURM_JOB_ID)"
    python "$CACHE_SCRIPT"
elif [[ -n "$ACCOUNT" ]]; then
    mkdir -p "$REPO_ROOT/slurm_logs"
    echo "==> Submitting the cache build to Slurm (account $ACCOUNT)"
    job_id="$(cache_job_script | sbatch --account="$ACCOUNT" --parsable)"
    cat <<MSG
    Submitted job $job_id (8 cpus, 16G, 30 min).
    Log: $REPO_ROOT/slurm_logs/build_dataset_cache_${job_id}.out
    Check on it with:  squeue -j $job_id   /   sacct -j $job_id
MSG
else
    cat <<MSG
==> Decoded TinyImageNet cache not built: $CACHE_FILE
    Decoding 110k JPEGs is too heavy for a login node, so it needs an
    allocation. Either re-run with --account to submit it as a batch job:

        ./build_env.sh --datasets-only --account <account>

    or build it interactively:

        salloc --account=<account> --time=0:30:00 --cpus-per-task=8 --mem=16G
        module load ${MODULES[*]}
        source $VENV/bin/activate
        python vision/scripts/build_dataset_cache.py

    Training jobs will not start without it (the sweep launcher checks for it).
MSG
fi

cat <<MSG

Done. Activate the environment with:

    module load ${MODULES[*]}
    source $VENV/bin/activate

Datasets: $DATA_DIR
MSG
