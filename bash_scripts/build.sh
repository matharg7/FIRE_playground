#!/bin/bash
# Set up everything needed to run the experiments, on any cluster.
#
#   ./build.sh                 do it all: environment, datasets, verification
#   ./build.sh --force         redo steps that are already done
#   ./build.sh --jobs 16       parallelism for tokenizing
#
# Steps are idempotent, so re-running is cheap: anything already in place is
# skipped. Individual steps can still be run alone if you want the heavy
# tokenization inside a CPU job rather than on a login node:
#
#   ./build.sh env | download | tokenize | check
#
# Paths and site specifics come from env.sh and a profile in clusters/.
# Override anything by exporting it first:  FIRE_WORK=/fast/me ./build.sh

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

PACKAGES=(torch numpy datasets tiktoken transformers wandb tqdm deepspeed pytest triton)
DATASETS=(wikitext openwebtext)
FORCE=0
JOBS=""
STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)      FORCE=1; shift ;;
        --jobs)       JOBS="${2:?--jobs needs a number}"; shift 2 ;;
        --jobs=*)     JOBS="${1#*=}"; shift ;;
        env|download|tokenize|check) STEP="$1"; shift ;;
        -h|--help)    sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)            echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
skip() { printf '    already done: %s (use --force to redo)\n' "$*"; }

# --- 1. python environment --------------------------------------------------
do_env() {
    local marker="$FIRE_VENV/.fire_build_ok"
    if [[ "$FORCE" == "0" && -f "$marker" ]]; then step "environment"; skip "$FIRE_VENV"; return 0; fi
    step "environment: $FIRE_VENV"
    fire_load_modules
    [[ -f "$FIRE_VENV/bin/activate" ]] || { mkdir -p "$(dirname "$FIRE_VENV")"; fire_create_venv "$FIRE_VENV"; }
    # shellcheck disable=SC1091
    source "$FIRE_VENV/bin/activate"
    export PYTHONNOUSERSITE=1
    fire_pip --upgrade pip >/dev/null 2>&1 || true
    #  deepspeed: sparsimony imports it at module load time
    #  triton   : torch.compile's backend; without it compile fails outright
    fire_pip "${PACKAGES[@]}"
    touch "$marker"
}

# --- 2. datasets ------------------------------------------------------------
do_download() {
    local marker="$FIRE_CACHE_DIR/.downloaded"
    if [[ "$FORCE" == "0" && -f "$marker" ]]; then step "download"; skip "$FIRE_CACHE_DIR"; return 0; fi
    step "downloading datasets (needs internet)"
    fire_load_modules; fire_activate
    mkdir -p "$HF_HOME" "$TIKTOKEN_CACHE_DIR"
    python "$FIRE_REPO_ROOT/language/data/prepare.py" --download-only --datasets "${DATASETS[@]}"
    touch "$marker"
}

do_tokenize() {
    fire_load_modules; fire_activate
    mkdir -p "$FIRE_DATA_DIR"
    local todo=()
    for name in "${DATASETS[@]}"; do
        if [[ "$FORCE" == "0" && -s "$FIRE_DATA_DIR/$name/train.bin" ]]; then
            step "tokenize $name"; skip "$FIRE_DATA_DIR/$name/train.bin"
        else
            todo+=("$name")
        fi
    done
    [[ ${#todo[@]} -eq 0 ]] && return 0
    step "tokenizing ${todo[*]} -> $FIRE_DATA_DIR"
    echo "    this is the slow part (openwebtext is ~18GB and takes a while);"
    echo "    on a busy login node consider: ./build.sh tokenize  inside a CPU job"
    local extra=()
    [[ -n "$JOBS" ]] && extra=(--num-proc "$JOBS")
    python "$FIRE_REPO_ROOT/language/data/prepare.py" \
        --datasets "${todo[@]}" --out-dir "$FIRE_DATA_DIR" "${extra[@]}"
}

# --- 3. verify --------------------------------------------------------------
do_check() {
    step "verifying"
    fire_load_modules; fire_activate
    fire_show_env
    echo
    python - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["FIRE_REPO_ROOT"], "language"))
ok = True

import torch
where = (f"{torch.cuda.device_count()} GPU(s)" if torch.cuda.is_available()
         else "no GPU here (expected on a login node; re-check inside a GPU job)")
print(f"torch            {torch.__version__}  {where}")
for name in ("numpy", "datasets", "tiktoken", "transformers", "wandb", "tqdm", "deepspeed"):
    try:
        mod = __import__(name)
        print(f"{name:<16} {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:                       # noqa: BLE001
        print(f"{name:<16} MISSING ({exc})"); ok = False
try:
    import triton  # noqa: F401
    print(f"{'triton':<16} {triton.__version__}  (torch.compile backend)")
except Exception:                                  # noqa: BLE001
    print(f"{'triton':<16} MISSING -- torch.compile will fail"); ok = False
try:
    import sparse_utils  # noqa: F401  (puts the vendored copy on sys.path)
    from sparsimony import rigl  # noqa: F401
    print(f"{'sparsimony':<16} importable (vendored under vision/)")
except Exception as exc:                           # noqa: BLE001
    print(f"{'sparsimony':<16} MISSING ({exc})"); ok = False

import numpy as np
for name in ("wikitext", "openwebtext"):
    path = os.path.join(os.environ["FIRE_DATA_DIR"], name, "train.bin")
    if os.path.exists(path):
        n = len(np.memmap(path, dtype=np.uint16, mode="r"))
        print(f"data/{name:<11} {n:>15,} tokens")
    else:
        print(f"data/{name:<11} MISSING ({path})"); ok = False

print("\nREADY -- next: ./bash_scripts/sweep.sh" if ok else "\nINCOMPLETE -- see above")
sys.exit(0 if ok else 1)
PY
}

case "$STEP" in
    env)      do_env ;;
    download) do_download ;;
    tokenize) do_tokenize ;;
    check)    do_check ;;
    "")       do_env; do_download; do_tokenize; do_check ;;
esac
