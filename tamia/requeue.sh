#!/bin/bash
# Progress report and recovery for a sweep's trial queue.
#
#   tamia/requeue.sh <sweep>                  # report only (safe, the default)
#   tamia/requeue.sh <sweep> --clear-stale    # release trials abandoned by a killed job
#   tamia/requeue.sh <sweep> --retry-failed   # also re-queue trials that exited non-zero
#   tamia/requeue.sh <sweep> --reset          # forget everything and start the sweep over
#   tamia/requeue.sh --list                   # every sweep with state on disk
#
# The queue lives on scratch, one marker per trial:
#
#   $FIRE_STATE_ROOT/<sweep>/claimed/<tid>/   a worker took it (atomic mkdir)
#   $FIRE_STATE_ROOT/<sweep>/done/<tid>       it exited 0
#   $FIRE_STATE_ROOT/<sweep>/failed/<tid>     it exited non-zero
#
# A trial claimed but neither done nor failed is STALE: its node hit the 24 h wall
# clock, was pre-empted, or crashed. train_st.py has no checkpointing, so there is
# nothing to resume -- the trial has to run again from scratch, and its claim must
# be released first or no future job will ever pick it up. That is what
# --clear-stale does, and it is the normal thing to run between submissions.
#
# --clear-stale refuses to act while jobs for that sweep are still queued or
# running, since their claims are legitimately held. Override with --force only if
# you are certain (releasing a live trial's claim lets a second worker start the
# same trial, producing a duplicate W&B run).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SWEEP=""
LIST="false"
CLEAR_STALE="false"
RETRY_FAILED="false"
RESET="false"
FORCE="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)          LIST="true"; shift ;;
        --clear-stale)   CLEAR_STALE="true"; shift ;;
        --retry-failed)  RETRY_FAILED="true"; CLEAR_STALE="true"; shift ;;
        --reset)         RESET="true"; shift ;;
        --force)         FORCE="true"; shift ;;
        -h|--help)       sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        -*)              echo "ERROR: unknown option: $1" >&2; exit 1 ;;
        *)               SWEEP="${1%.yaml}"; SWEEP="$(basename "$SWEEP")"; shift ;;
    esac
done

FIRE_STATE_ROOT="${FIRE_STATE_ROOT:-${SCRATCH:?SCRATCH is not set}/fire-tamia/state}"

if [[ "$LIST" == "true" ]]; then
    if [[ ! -d "$FIRE_STATE_ROOT" ]]; then
        echo "No sweep state yet under $FIRE_STATE_ROOT"
        exit 0
    fi
    echo "Sweeps with state under $FIRE_STATE_ROOT:"
    for d in "$FIRE_STATE_ROOT"/*/; do
        [[ -d "$d" ]] || continue
        n="$(basename "$d")"
        printf '  %-42s done=%-5s failed=%-5s claimed=%s\n' "$n" \
            "$(find "$d/done"    -maxdepth 1 -type f 2>/dev/null | wc -l)" \
            "$(find "$d/failed"  -maxdepth 1 -type f 2>/dev/null | wc -l)" \
            "$(find "$d/claimed" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)"
    done
    exit 0
fi

[[ -n "$SWEEP" ]] || { echo "ERROR: no sweep given. Try --list or --help." >&2; exit 1; }

STATE="$FIRE_STATE_ROOT/$SWEEP"
TRIALS="$REPO_ROOT/tamia/trials/$SWEEP.tsv"

# --- Reset -------------------------------------------------------------------
if [[ "$RESET" == "true" ]]; then
    if [[ ! -d "$STATE" ]]; then
        echo "No state for '$SWEEP' at $STATE -- nothing to reset."
        exit 0
    fi
    echo "About to delete ALL queue state for '$SWEEP':"
    echo "    $STATE"
    echo "Every trial will be re-run on the next submission. Offline W&B runs"
    echo "already on disk are NOT deleted, so re-running produces duplicates in"
    echo "the project unless you also clear \$SCRATCH/wandb/$SWEEP."
    read -r -p "Type the sweep name to confirm: " reply
    if [[ "$reply" != "$SWEEP" ]]; then
        echo "Not confirmed; nothing was deleted."
        exit 1
    fi
    rm -rf "$STATE"
    echo "Deleted $STATE"
    exit 0
fi

if [[ ! -d "$STATE" ]]; then
    echo "No queue state for '$SWEEP' yet (nothing has run)."
    [[ -f "$TRIALS" ]] && echo "Trial list: $TRIALS ($(grep -vc '^#' "$TRIALS") trials)"
    exit 0
fi

mkdir -p "$STATE/claimed" "$STATE/done" "$STATE/failed"

# --- Stale claims: claimed, but neither done nor failed ----------------------
stale=()
while IFS= read -r -d '' c; do
    tid="$(basename "$c")"
    [[ -e "$STATE/done/$tid" || -e "$STATE/failed/$tid" ]] && continue
    stale+=("$tid")
done < <(find "$STATE/claimed" -maxdepth 1 -mindepth 1 -type d -print0)

TOTAL="?"
[[ -f "$TRIALS" ]] && TOTAL="$(grep -vc '^#' "$TRIALS")"
NDONE="$(find "$STATE/done"   -maxdepth 1 -type f | wc -l)"
NFAIL="$(find "$STATE/failed" -maxdepth 1 -type f | wc -l)"
NCLAIM="$(find "$STATE/claimed" -maxdepth 1 -mindepth 1 -type d | wc -l)"

cat <<REPORT

  sweep     : $SWEEP
  state     : $STATE
  trials    : $TOTAL
  done      : $NDONE
  failed    : $NFAIL
  claimed   : $NCLAIM  (of which ${#stale[@]} in flight or stale)
REPORT

if [[ "$TOTAL" != "?" ]]; then
    echo "  remaining : $(( TOTAL - NDONE ))"
fi
echo

# --- Are any jobs for this sweep still around? -------------------------------
running=0
if command -v squeue >/dev/null 2>&1; then
    running="$(squeue -u "$USER" -h -n "$SWEEP" 2>/dev/null | wc -l)"
    if (( running > 0 )); then
        echo "  NOTE: $running job(s) named '$SWEEP' are queued or running -- their claims"
        echo "        are live, not stale."
        echo
    fi
fi

if [[ ${#stale[@]} -eq 0 && "$RETRY_FAILED" != "true" ]]; then
    echo "Nothing to release."
    [[ "$NFAIL" -gt 0 ]] && echo "($NFAIL failed trial(s); re-queue them with --retry-failed.)"
    exit 0
fi

if [[ "$CLEAR_STALE" != "true" ]]; then
    cat <<MSG
${#stale[@]} trial(s) are claimed but unfinished. If no job is running, those are
stale and will never be retried until released:

    tamia/requeue.sh $SWEEP --clear-stale
MSG
    exit 0
fi

if (( running > 0 )) && [[ "$FORCE" != "true" ]]; then
    cat >&2 <<MSG
REFUSING to release claims: $running job(s) for '$SWEEP' are still queued or running,
so some of those claims belong to trials currently in flight. Releasing them would
let a second worker start the same trial and log a duplicate W&B run.

Wait for the jobs to finish, then re-run. Override with --force if you are sure.
MSG
    exit 1
fi

released=0
for tid in "${stale[@]}"; do
    rmdir "$STATE/claimed/$tid" 2>/dev/null && released=$(( released + 1 ))
done
echo "Released $released stale claim(s)."

if [[ "$RETRY_FAILED" == "true" ]]; then
    refail=0
    while IFS= read -r -d '' f; do
        tid="$(basename "$f")"
        rm -f "$f"
        rmdir "$STATE/claimed/$tid" 2>/dev/null || true
        refail=$(( refail + 1 ))
    done < <(find "$STATE/failed" -maxdepth 1 -type f -print0)
    echo "Re-queued $refail failed trial(s)."
fi

echo
echo "Now re-submit:  tamia/submit.sh $SWEEP"
