#!/bin/bash
# Set up the python environment for the TRACE continual-learning experiments.
#
#   ./build_env.sh             env, download, then check
#   ./build_env.sh env         create the venv and install packages
#   ./build_env.sh download    TRACE data + model weights (needs internet: login node)
#   ./build_env.sh check       report versions and import everything we need
#   ./build_env.sh --force env redo a step that is already done
#
# Paths and site specifics come from env.sh (and the optional, gitignored
# local.sh next to it). Override anything by exporting it first:
#   TRACE_VENV=/elsewhere/venv ./build_env.sh env

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

# transformers: sparsimony pins <5, and 4.57 is the newest 4.x in the wheelhouse.
# deepspeed   : sparsimony imports it at module load time (never used for training).
# triton      : torch.compile's backend.
# rouge, rouge_score, fuzzywuzzy, rapidfuzz, sacrebleu, sacremoses, nltk: TRACE metrics.
PACKAGES=(
    torch numpy "transformers==4.57.6" datasets deepspeed triton
    sentencepiece pandas nltk rouge rouge_score fuzzywuzzy rapidfuzz sacrebleu sacremoses
    wandb tqdm pytest gdown
)
# Google Drive file id from trace/README.md ("Trace Benchmark" link).
TRACE_GDRIVE_ID="1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"
MODELS=(HuggingFaceTB/SmolLM2-135M Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B)
FORCE=0
STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)   FORCE=1; shift ;;
        env|download|check) STEP="$1"; shift ;;
        -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)         echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
skip() { printf '    already done: %s (use --force to redo)\n' "$*"; }

do_env() {
    local marker="$TRACE_VENV/.trace_build_ok"
    if [[ "$FORCE" == "0" && -f "$marker" ]]; then step "environment"; skip "$TRACE_VENV"; return 0; fi
    step "environment: $TRACE_VENV"
    trace_load_modules
    [[ -f "$TRACE_VENV/bin/activate" ]] || { mkdir -p "$(dirname "$TRACE_VENV")"; trace_create_venv "$TRACE_VENV"; }
    # shellcheck disable=SC1091
    source "$TRACE_VENV/bin/activate"
    export PYTHONNOUSERSITE=1
    trace_pip --upgrade pip >/dev/null 2>&1 || true
    trace_pip "${PACKAGES[@]}"
    # The vendored copy under vision/, editable so fixes there apply here too.
    # Its backend is hatchling; install it into the venv and build without
    # isolation, since isolated builds would try to fetch it from PyPI.
    trace_pip hatchling editables
    trace_pip --no-build-isolation -e "$TRACE_REPO_ROOT/vision/sparsimony"
    touch "$marker"
}

do_download() {
    trace_load_modules
    trace_activate

    local marker="$TRACE_DATA_DIR/.downloaded"
    if [[ "$FORCE" == "0" && -f "$marker" ]]; then
        step "TRACE data"; skip "$TRACE_DATA_DIR"
    else
        step "TRACE data -> $TRACE_DATA_DIR (needs internet)"
        mkdir -p "$TRACE_DATA_DIR"
        local archive="$TRACE_DATA_DIR/TRACE-benchmark.zip"
        [[ -s "$archive" && "$FORCE" == "0" ]] || gdown "$TRACE_GDRIVE_ID" -O "$archive"
        python -m zipfile -e "$archive" "$TRACE_DATA_DIR"
        touch "$marker"
    fi

    step "model weights -> $HF_HOME (needs internet)"
    mkdir -p "$HF_HOME"
    python - "${MODELS[@]}" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    path = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print(f"    {repo:<32} {path}")
PY
}

do_check() {
    step "verifying"
    trace_load_modules
    trace_activate
    trace_show_env
    echo
    # Models must load from the local cache: compute nodes have no internet.
    HF_HUB_OFFLINE=1 python - "${MODELS[@]}" <<'PY'
import importlib, sys
ok = True

import torch
where = (f"{torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}"
         if torch.cuda.is_available()
         else "no GPU here (expected on a login node; re-check inside a GPU job)")
print(f"{'torch':<16} {torch.__version__}  {where}")

for name in ("numpy", "transformers", "datasets", "deepspeed", "triton", "sentencepiece",
             "pandas", "nltk", "rouge", "rouge_score", "fuzzywuzzy", "rapidfuzz",
             "sacrebleu", "sacremoses", "wandb", "tqdm", "pytest"):
    try:
        mod = importlib.import_module(name)
        print(f"{name:<16} {getattr(mod, '__version__', 'ok')}")
    except Exception as exc:                       # noqa: BLE001
        print(f"{name:<16} MISSING ({exc})"); ok = False

try:
    import sparsimony
    from sparsimony import rigl  # noqa: F401
    print(f"{'sparsimony':<16} importable from {sparsimony.__path__[0]}")
except Exception as exc:                           # noqa: BLE001
    print(f"{'sparsimony':<16} MISSING ({exc})"); ok = False

# On a GPU node, load each model in bf16 with sdpa attention and generate a few tokens.
from transformers import AutoModelForCausalLM, AutoTokenizer
for repo in sys.argv[1:]:
    try:
        tok = AutoTokenizer.from_pretrained(repo)
        if not torch.cuda.is_available():
            print(f"{repo:<32} tokenizer ok (model load skipped: no GPU)")
            continue
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        ids = tok("The capital of France is", return_tensors="pt").to("cuda")
        out = model.generate(**ids, max_new_tokens=5, do_sample=False)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        text = tok.decode(out[0][ids["input_ids"].shape[1]:]).strip()
        print(f"{repo:<32} {n:.0f}M params, generates: {text!r}")
        del model; torch.cuda.empty_cache()
    except Exception as exc:                       # noqa: BLE001
        print(f"{repo:<32} FAILED ({exc})"); ok = False

print("\nREADY" if ok else "\nINCOMPLETE -- see above")
sys.exit(0 if ok else 1)
PY
}

case "$STEP" in
    env)      do_env ;;
    download) do_download ;;
    check)    do_check ;;
    "")       do_env; do_download; do_check ;;
esac
