#!/bin/bash
# Upload a sweep's offline W&B runs. RUN THIS ON A LOGIN NODE -- it is the only
# place with internet.
#
#   tamia/sync_wandb.sh tin_vgg16_class_inc_set_rigl
#   tamia/sync_wandb.sh <sweep> --dry-run          # list what would be uploaded
#   tamia/sync_wandb.sh <sweep> --batch 10         # smaller chunks
#   tamia/sync_wandb.sh --all                      # every sweep on disk
#
# tamIA compute nodes cannot reach api.wandb.ai, so training runs with
# WANDB_MODE=offline and writes complete run records to
#
#     $FIRE_WANDB_ROOT/<sweep>/wandb/offline-run-<timestamp>-<id>/
#
# which env.sh puts inside the repo (wandb_offline/). Sweeps that ran before
# that lived under $SCRATCH/wandb/<sweep>/wandb/, and both are still searched,
# so an older backlog uploads without moving anything.
#
# `wandb sync` replays those records to the server. It is idempotent: a synced
# run gets a *.wandb.synced marker and is skipped here on later passes, so
# re-running after an interrupted upload is safe and cheap.
#
# Uploading is network- and I/O-light but not instant; a 180-run sweep takes a
# while. It is chunked (--batch, default 25) so an interrupted session leaves
# whole runs uploaded rather than one huge half-finished operation, and so you
# can stop it between chunks without losing progress. For a very large backlog,
# run it inside `screen`/`tmux`, or from an interactive allocation -- but note
# that COMPUTE NODES HAVE NO INTERNET, so an allocation will not work. The login
# node is the only option; keep an eye on it and be considerate.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SWEEP=""
ALL="false"
BATCH="25"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)       ALL="true"; shift ;;
        --batch)     BATCH="${2:?--batch needs an argument}"; shift 2 ;;
        --batch=*)   BATCH="${1#*=}"; shift ;;
        --dry-run)   DRY_RUN="true"; shift ;;
        -h|--help)   sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        -*)          echo "ERROR: unknown option: $1" >&2; exit 1 ;;
        *)           SWEEP="$1"; shift ;;
    esac
done

if [[ -z "$SWEEP" && "$ALL" != "true" ]]; then
    echo "ERROR: give a sweep name, or --all." >&2
    echo "Available offline run directories:" >&2
    ls -1 "${FIRE_WANDB_ROOT:-$REPO_ROOT/wandb_offline}" "${SCRATCH:-}/wandb" 2>/dev/null \
        | sed 's/^/  /' >&2 || true
    exit 1
fi

# Warn if this looks like a compute node: no internet there, so the sync will
# hang or fail confusingly.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "WARNING: running inside Slurm job ${SLURM_JOB_ID}. tamIA compute nodes have no"
    echo '         internet -- "wandb sync" will not reach the server. Use a login node.'
    echo
fi

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source tamia/env.sh >/dev/null || { echo "ERROR: could not set up the environment." >&2; exit 1; }

# env.sh only forces offline mode when given a sweep name, but clear it anyway so
# an inherited WANDB_MODE from an earlier shell cannot make this a no-op.
unset WANDB_MODE

if [[ ! -f "$HOME/.wandb_token" ]]; then
    echo "ERROR: ~/.wandb_token not found -- put your W&B API key there." >&2
    echo "       (https://wandb.ai/authorize)" >&2
    exit 1
fi
export WANDB_API_KEY="$(cat "$HOME/.wandb_token")"

# Every directory a sweep's runs may be in: the in-repo one env.sh points
# WANDB_DIR at, and the scratch path used before that. Only those that exist.
sweep_roots() {
    local sweep="$1" r
    for r in "$FIRE_WANDB_ROOT/$sweep/wandb" "${SCRATCH:-}/wandb/$sweep/wandb"; do
        [[ -d "$r" ]] && echo "$r"
    done
    return 0
}

sync_one_sweep() {
    local sweep="$1"
    local roots=() r
    while IFS= read -r r; do roots+=("$r"); done < <(sweep_roots "$sweep")

    if [[ ${#roots[@]} -eq 0 ]]; then
        echo "-- $sweep: no offline runs at $FIRE_WANDB_ROOT/$sweep/wandb (nothing ran yet?)"
        return 0
    fi

    # Collect unsynced offline runs. wandb drops a *.wandb.synced file inside a
    # run directory once it has been uploaded, which is what makes this
    # re-runnable without duplicating runs in the project.
    local pending=() total=0 d
    while IFS= read -r -d '' d; do
        total=$(( total + 1 ))
        if compgen -G "$d/*.wandb.synced" >/dev/null; then
            continue
        fi
        pending+=("$d")
    done < <(find "${roots[@]}" -maxdepth 1 -type d -name 'offline-run-*' -print0 | sort -z)

    local synced=$(( total - ${#pending[@]} ))

    echo "== $sweep: $total offline run(s), $synced already synced, ${#pending[@]} to upload"
    if [[ ${#roots[@]} -gt 1 ]]; then
        printf '   (searched: %s)\n' "${roots[*]}"
    fi
    if [[ ${#pending[@]} -eq 0 ]]; then
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        printf '   would sync: %s\n' "${pending[@]##*/}"
        return 0
    fi

    local i=0 n=${#pending[@]}
    while (( i < n )); do
        local chunk=("${pending[@]:i:BATCH}")
        echo "   uploading $(( i + 1 ))-$(( i + ${#chunk[@]} )) of $n ..."
        # Not `wandb sync --sync-all`: that walks whatever directory it decides
        # is the wandb root, which would drag in other sweeps. Explicit paths keep
        # an upload scoped to the sweep you asked for.
        if ! wandb sync "${chunk[@]}"; then
            echo "   ERROR: chunk failed. Fix the problem and re-run -- already-uploaded" >&2
            echo "          runs are skipped automatically." >&2
            return 1
        fi
        i=$(( i + BATCH ))
    done
    echo "   $sweep done."
}

status=0
if [[ "$ALL" == "true" ]]; then
    shopt -s nullglob
    found="false"
    declare -A seen=()
    for d in "$FIRE_WANDB_ROOT"/*/ "${SCRATCH:-}"/wandb/*/; do
        name="$(basename "$d")"
        # Skip the shared cache/artifact/config directories env.sh creates.
        case "$name" in cache|artifacts|config) continue ;; esac
        # A sweep present in both roots is one sweep; sync_one_sweep sees both.
        [[ -n "${seen[$name]:-}" ]] && continue
        seen["$name"]=1
        found="true"
        sync_one_sweep "$name" || status=1
    done
    [[ "$found" == "true" ]] \
        || echo "No sweep directories under $FIRE_WANDB_ROOT or ${SCRATCH:-}/wandb."
else
    sync_one_sweep "$SWEEP" || status=1
fi

cat <<MSG

Offline runs are not attached to a W&B sweep object, so there is no sweep page.
Find them in the project (DST Continual Learning) by filtering on
  Group == <sweep name>        e.g. Group == $SWEEP
or on the tag 'tamia'. Check the run count there against the number above.
MSG

exit "$status"
