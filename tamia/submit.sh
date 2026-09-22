#!/bin/bash
# Expand a sweep, size the array for it, and submit -- the recommended way to
# launch anything on tamIA.
#
# Usage:
#   tamia/submit.sh <sweep-name> [options]
#
#   tamia/submit.sh tin_vgg16_class_inc_dense                    # 3 trials, one node
#   tamia/submit.sh tin_vgg16_class_inc_set_rigl                 # 180 trials, sized for you
#   tamia/submit.sh tin_vgg16_class_inc_set_rigl --gpu-type h200 # 8 GPUs/node
#   tamia/submit.sh <sweep> --nodes 4                            # force the array size
#   tamia/submit.sh <sweep> --dry-run                            # print, do not submit
#   tamia/submit.sh <sweep> --test                               # 1 node, 30 min, smoke test
#
# Options:
#   --gpu-type h100|h200   node flavour                     (default: h100)
#   --trials-per-gpu N     concurrent trials per GPU        (default: 2)
#   --trial-budget MIN     wall clock one trial needs       (default: 150)
#   --time HH:MM:SS        wall clock per array task        (default: 23:00:00)
#   --nodes N              array size; default is computed from the trial count
#   --account NAME         Slurm account                    (default: $FIRE_ACCOUNT or aip-yani)
#   --job-name NAME        Slurm job name                   (default: the sweep name)
#   --no-expand            reuse tamia/trials/<sweep>.tsv as it is
#   --dry-run              print the plan and the sbatch line, submit nothing
#   --test                 --nodes 1 --time 0:30:00  (tamIA allows >=5 min test jobs)
#
# Array sizing
# ------------
# Each node runs `ngpu x trials-per-gpu` trials at once and can get through
# `floor(walltime / trial-budget)` of them per slot, so one array task clears
# about `slots x rounds` trials. The array is sized to cover what is still
# outstanding. Over-provisioning is harmless -- surplus tasks find every trial
# claimed and exit in seconds -- so when in doubt, round up.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SWEEP=""
GPU_TYPE="h100"
TRIALS_PER_GPU="2"
TRIAL_BUDGET_MIN="150"
WALLTIME="23:00:00"
NODES=""
# tamIA RAPs carry the aip- prefix and are separate from nibi's rrg-yani. If this
# is wrong, sbatch rejects it immediately -- find the right one under Resource
# Allocation Projects at https://ccdb.alliancecan.ca/ and pass --account.
ACCOUNT="${FIRE_ACCOUNT:-aip-yani}"
JOB_NAME=""
EXPAND="true"
DRY_RUN="false"
BUDGET_EXPLICIT="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-type)        GPU_TYPE="${2:?--gpu-type needs an argument}"; shift 2 ;;
        --gpu-type=*)      GPU_TYPE="${1#*=}"; shift ;;
        --trials-per-gpu)  TRIALS_PER_GPU="${2:?--trials-per-gpu needs an argument}"; shift 2 ;;
        --trials-per-gpu=*) TRIALS_PER_GPU="${1#*=}"; shift ;;
        --trial-budget)    TRIAL_BUDGET_MIN="${2:?--trial-budget needs an argument}"; BUDGET_EXPLICIT="true"; shift 2 ;;
        --trial-budget=*)  TRIAL_BUDGET_MIN="${1#*=}"; BUDGET_EXPLICIT="true"; shift ;;
        --time)            WALLTIME="${2:?--time needs an argument}"; shift 2 ;;
        --time=*)          WALLTIME="${1#*=}"; shift ;;
        --nodes)           NODES="${2:?--nodes needs an argument}"; shift 2 ;;
        --nodes=*)         NODES="${1#*=}"; shift ;;
        --account)         ACCOUNT="${2:?--account needs an argument}"; shift 2 ;;
        --account=*)       ACCOUNT="${1#*=}"; shift ;;
        --job-name)        JOB_NAME="${2:?--job-name needs an argument}"; shift 2 ;;
        --job-name=*)      JOB_NAME="${1#*=}"; shift ;;
        --no-expand)       EXPAND="false"; shift ;;
        --dry-run)         DRY_RUN="true"; shift ;;
        --test)            NODES="1"; WALLTIME="0:30:00"; shift ;;
        -h|--help)         sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        -*)                echo "ERROR: unknown option: $1" >&2; exit 1 ;;
        *)                 [[ -n "$SWEEP" ]] && { echo "ERROR: more than one sweep given: $SWEEP, $1" >&2; exit 1; }
                           SWEEP="$1"; shift ;;
    esac
done

[[ -n "$SWEEP" ]] || { echo "ERROR: no sweep given. Try: tamia/submit.sh --help" >&2; exit 1; }
SWEEP="${SWEEP%.yaml}"
SWEEP="$(basename "$SWEEP")"
JOB_NAME="${JOB_NAME:-$SWEEP}"

case "$GPU_TYPE" in
    h100) NGPU=4; NCPU=48 ;;
    h200) NGPU=8; NCPU=64 ;;
    *)    echo "ERROR: --gpu-type must be h100 or h200 (got: $GPU_TYPE)" >&2; exit 1 ;;
esac

# --- Wall clock, in minutes, for the sizing arithmetic and the policy checks ---
to_minutes() {
    local t="$1" d=0
    if [[ "$t" == *-* ]]; then d="${t%%-*}"; t="${t#*-}"; fi
    local IFS=':' ; read -r h m s <<<"$t"
    echo $(( d * 1440 + ${h:-0} * 60 + ${m:-0} + ( ${s:-0} > 0 ? 1 : 0 ) ))
}
WALL_MIN="$(to_minutes "$WALLTIME")"

# tamIA policy: jobs run 1 h to 24 h (test jobs may be as short as 5 min).
if (( WALL_MIN > 1440 )); then
    echo "ERROR: --time $WALLTIME exceeds tamIA's 24 h maximum." >&2
    echo "       Use more nodes instead of a longer job: --nodes N" >&2
    exit 1
fi
if (( WALL_MIN < 60 )); then
    echo "WARNING: --time $WALLTIME is under tamIA's 1 h minimum for regular jobs."
    echo "         Only test jobs (>= 5 min) may be this short."
    if (( WALL_MIN < 5 )); then
        echo "ERROR: under the 5 min floor for test jobs." >&2
        exit 1
    fi
fi

# The runner's deadline guard will not start a trial with less than the budget
# left, so a budget at or above the wall clock means no worker ever starts
# anything -- easy to do with --test, and it would look like a mysteriously empty
# job. Auto-shrink the default, but never silently override an explicit value.
if (( TRIAL_BUDGET_MIN >= WALL_MIN )); then
    if [[ "$BUDGET_EXPLICIT" == "true" ]]; then
        cat >&2 <<MSG
ERROR: --trial-budget ${TRIAL_BUDGET_MIN} min is not less than --time $WALLTIME (${WALL_MIN} min).
       The runner refuses to start a trial it cannot finish, so every worker
       would exit immediately. Lower --trial-budget or raise --time.
MSG
        exit 1
    fi
    NEW_BUDGET=$(( WALL_MIN * 9 / 10 )); (( NEW_BUDGET >= 1 )) || NEW_BUDGET=1
    echo "NOTE: trial budget lowered ${TRIAL_BUDGET_MIN} -> ${NEW_BUDGET} min to fit --time $WALLTIME."
    echo "      Trials that outlast the job are killed and re-queued (no checkpointing)."
    TRIAL_BUDGET_MIN="$NEW_BUDGET"
fi

cd "$REPO_ROOT"

# --- Environment (needed for PyYAML in expand_sweep.py) ----------------------
# shellcheck disable=SC1091
source tamia/env.sh >/dev/null || { echo "ERROR: could not set up the environment." >&2; exit 1; }

# --- Expand the grid ---------------------------------------------------------
TRIALS="$REPO_ROOT/tamia/trials/$SWEEP.tsv"
if [[ "$EXPAND" == "true" ]]; then
    python tamia/expand_sweep.py "$SWEEP"
elif [[ ! -f "$TRIALS" ]]; then
    echo "ERROR: --no-expand given but $TRIALS does not exist." >&2
    exit 1
fi

TOTAL="$(grep -vc '^#' "$TRIALS")"

# --- What is left to do ------------------------------------------------------
STATE="$FIRE_STATE_ROOT/$SWEEP"
NDONE=0
NFAIL=0
[[ -d "$STATE/done" ]]   && NDONE="$(find "$STATE/done"   -maxdepth 1 -type f | wc -l)"
[[ -d "$STATE/failed" ]] && NFAIL="$(find "$STATE/failed" -maxdepth 1 -type f | wc -l)"
OUTSTANDING=$(( TOTAL - NDONE ))

# A trial claimed but neither done nor failed was abandoned mid-flight (wall
# clock, crash) or belongs to a job still running. In the first case its claim
# blocks every future worker from retrying it, which is quiet and easy to miss --
# so say so rather than silently submitting an array that cannot finish the sweep.
NSTALE=0
if [[ -d "$STATE/claimed" ]]; then
    while IFS= read -r -d '' c; do
        tid="$(basename "$c")"
        [[ -e "$STATE/done/$tid" || -e "$STATE/failed/$tid" ]] && continue
        NSTALE=$(( NSTALE + 1 ))
    done < <(find "$STATE/claimed" -maxdepth 1 -mindepth 1 -type d -print0)
fi
if (( NSTALE > 0 )); then
    cat <<MSG

NOTE: $NSTALE trial(s) are claimed but unfinished. If a job for this sweep is
      still running they are in flight and fine. If not, they were abandoned and
      no worker will retry them until you release the claims:

          tamia/requeue.sh $SWEEP --clear-stale

MSG
fi

if (( OUTSTANDING <= 0 )); then
    cat <<MSG

Nothing to submit: all $TOTAL trials of '$SWEEP' are marked done.
  Re-run a subset :  tamia/requeue.sh $SWEEP --retry-failed
  Start over      :  tamia/requeue.sh $SWEEP --reset
  Upload to W&B   :  tamia/sync_wandb.sh $SWEEP
MSG
    exit 0
fi

# --- Size the array ----------------------------------------------------------
SLOTS=$(( NGPU * TRIALS_PER_GPU ))
ROUNDS=$(( WALL_MIN / TRIAL_BUDGET_MIN )); (( ROUNDS >= 1 )) || ROUNDS=1
CAPACITY=$(( SLOTS * ROUNDS ))

if [[ -z "$NODES" ]]; then
    NODES=$(( (OUTSTANDING + CAPACITY - 1) / CAPACITY ))
    (( NODES >= 1 )) || NODES=1
fi
# tamIA allows at most 1000 jobs (running + pending); an array task is a job.
if (( NODES > 1000 )); then
    echo "WARNING: array of $NODES capped at 1000 (tamIA's job limit)."
    NODES=1000
fi
ARRAY="0-$(( NODES - 1 ))"

mkdir -p "$REPO_ROOT/slurm_logs"

SBATCH_ARGS=(
    --account="$ACCOUNT"
    --job-name="$JOB_NAME"
    --gpus="${GPU_TYPE}:${NGPU}"
    --cpus-per-task="$NCPU"
    --mem=0
    --time="$WALLTIME"
    --array="$ARRAY"
    --export="ALL,FIRE_SWEEP=$SWEEP,FIRE_REPO_ROOT=$REPO_ROOT,FIRE_TRIALS_PER_GPU=$TRIALS_PER_GPU,FIRE_TRIAL_BUDGET_MIN=$TRIAL_BUDGET_MIN"
)

cat <<PLAN

--------------------------------------------------------------
 Sweep         : $SWEEP
 Trials        : $TOTAL total, $NDONE done, $NFAIL failed -> $OUTSTANDING outstanding
 Node          : ${GPU_TYPE} x ${NGPU} GPUs, ${NCPU} cores, whole node (--mem=0)
 Concurrency   : $SLOTS trials/node ($TRIALS_PER_GPU per GPU)
 Wall clock    : $WALLTIME  (~$ROUNDS trial(s) per slot at ${TRIAL_BUDGET_MIN} min each)
 Capacity      : ~$CAPACITY trials per array task
 Array         : $ARRAY  ($NODES node(s), ~$(( NODES * CAPACITY )) trial capacity)
 Account       : $ACCOUNT
 Logs          : slurm_logs/tamia_${JOB_NAME}_<jobid>_<task>.out
 W&B           : offline -> \$SCRATCH/wandb/$SWEEP  (sync afterwards)
--------------------------------------------------------------

PLAN

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run. Would submit:"
    printf '  sbatch'
    # Quote only what needs it, so the line stays readable and still pastes.
    for a in "${SBATCH_ARGS[@]}" tamia/run_sweep.slurm; do
        if [[ "$a" == *[[:space:]\'\"]* ]]; then printf ' %q' "$a"; else printf ' %s' "$a"; fi
    done
    printf '\n'
    exit 0
fi

JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" tamia/run_sweep.slurm)"

cat <<MSG
Submitted array job $JOB_ID ($NODES task(s)).

  Watch     :  squeue -u \$USER -j $JOB_ID
  Progress  :  tamia/requeue.sh $SWEEP
  Logs      :  tail -f slurm_logs/tamia_${JOB_NAME}_${JOB_ID}_0.out
  After     :  tamia/sync_wandb.sh $SWEEP     (from a LOGIN node)
MSG
