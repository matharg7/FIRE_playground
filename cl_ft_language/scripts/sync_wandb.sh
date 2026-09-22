#!/bin/bash
# Push finished offline W&B runs to wandb.ai. Only needed when the runs logged
# offline (the default on clusters whose compute nodes have no internet).
# RUN THIS WHERE THERE IS INTERNET (a login node on Alliance clusters):
# compute nodes cannot sync themselves, and neither can dependent SLURM jobs.
#
#   ./sync_wandb.sh                    # sync every finished, unsynced run
#   ./sync_wandb.sh --dry-run          # list what would be synced
#   ./sync_wandb.sh --root <dir>       # default: $TRACE_OUTPUT_DIR
#
# A run is eligible when train.py has written a DONE marker in its output
# directory and no SYNCED marker exists yet, so this is safe to re-run: each
# run is pushed exactly once.

set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
ROOT="$TRACE_OUTPUT_DIR"
DRY_RUN=0
WANDB_BIN="${WANDB_BIN:-wandb}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)     ROOT="${2:?--root needs an argument}"; shift 2 ;;
        --root=*)   ROOT="${1#*=}"; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          echo "ERROR: unknown option: $1" >&2; exit 1 ;;
    esac
done

# Tests set TRACE_SYNC_NO_ENV=1 and point WANDB_BIN at a stub.
if [[ -z "${TRACE_SYNC_NO_ENV:-}" ]]; then
    trace_load_modules
    trace_activate || true
fi

if [[ ! -d "$ROOT" ]]; then
    echo "Nothing to do: $ROOT does not exist"
    exit 0
fi

synced=0; skipped=0; failed=0; pending=0

while IFS= read -r done_file; do
    run_dir="$(dirname "$done_file")"
    name="$(basename "$run_dir")"

    if [[ -f "$run_dir/SYNCED" ]]; then
        skipped=$((skipped + 1))
        continue
    fi

    wandb_run_dir="$(sed -n 's/^wandb_run_dir=//p' "$done_file" | head -1)"
    if [[ -z "$wandb_run_dir" || ! -d "$wandb_run_dir" ]]; then
        echo "SKIP  $name (no offline run directory recorded)"
        skipped=$((skipped + 1))
        continue
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "WOULD SYNC  $name  <-  $wandb_run_dir"
        pending=$((pending + 1))
        continue
    fi

    echo "SYNC  $name  <-  $wandb_run_dir"
    if "$WANDB_BIN" sync "$wandb_run_dir"; then
        printf 'synced_at=%s\nwandb_run_dir=%s\n' \
            "$(date -Is)" "$wandb_run_dir" > "$run_dir/SYNCED"
        synced=$((synced + 1))
    else
        echo "FAILED  $name" >&2
        failed=$((failed + 1))
    fi
done < <(find "$ROOT" -mindepth 2 -maxdepth 2 -name DONE -type f | sort)

if [[ "$DRY_RUN" == "1" ]]; then
    echo "dry run: $pending to sync, $skipped skipped"
else
    echo "synced $synced, skipped $skipped, failed $failed"
fi
[[ "$failed" -eq 0 ]]
