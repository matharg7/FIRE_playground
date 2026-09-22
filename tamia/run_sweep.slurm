#!/bin/bash
# tamIA sweep runner -- one array task fills one whole node with trials.
#
#   SUBMIT FROM THE REPO ROOT, and prefer tamia/submit.sh over raw sbatch:
#
#       tamia/submit.sh tin_vgg16_class_inc_set_rigl
#
#   The raw form, if you want to drive it yourself:
#
#       python tamia/expand_sweep.py tin_vgg16_class_inc_set_rigl
#       mkdir -p slurm_logs
#       sbatch --array=0-5 --export=ALL,FIRE_SWEEP=tin_vgg16_class_inc_set_rigl \
#              tamia/run_sweep.slurm
#
# ---------------------------------------------------------------------------
# Why this does not use `wandb agent`
# ---------------------------------------------------------------------------
# tamIA compute nodes have no internet, so an agent cannot ask the sweep server
# for its next configuration. The grid is therefore expanded up front by
# expand_sweep.py into tamia/trials/<sweep>.tsv, and the trials are handed out by
# a lock-free queue on scratch: a worker owns a trial once `mkdir` of its claim
# directory succeeds, which is atomic on Lustre.
#
#     $FIRE_STATE_ROOT/<sweep>/claimed/<tid>/   the claim
#     $FIRE_STATE_ROOT/<sweep>/done/<tid>       trial exited 0
#     $FIRE_STATE_ROOT/<sweep>/failed/<tid>     trial exited non-zero
#
# This keeps the two properties the agent gave you -- work is balanced across
# the array, and surplus array tasks are harmless (they find nothing to claim and
# exit in seconds) -- and adds resumability: re-submitting picks up exactly what
# never finished. That matters because train_st.py does not checkpoint and tamIA
# caps jobs at 24 h, so a trial killed by the wall clock has to be redone whole.
#
# ---------------------------------------------------------------------------
# Whole-node packing
# ---------------------------------------------------------------------------
# tamIA allocates whole nodes and requires that every GPU be used, so the headers
# below take all 4 H100s (submit.sh rewrites them for --gpu-type h200, which is
# 8 GPUs). The node runs NGPU x FIRE_TRIALS_PER_GPU trials concurrently; 2 per
# GPU by default, matching the two-agents-per-GPU setup on nibi, which pays off
# because a single trial leaves an H100 underused.
#
# ---------------------------------------------------------------------------
#SBATCH --job-name=fire-sweep
#SBATCH --account=aip-yani
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=23:00:00
#SBATCH --array=0-0
#SBATCH --output=slurm_logs/tamia_%x_%A_%a.out
#SBATCH --error=slurm_logs/tamia_%x_%A_%a.err

set -uo pipefail

# --- 0. Inputs ---------------------------------------------------------------
SWEEP="${FIRE_SWEEP:-}"
if [[ -z "$SWEEP" ]]; then
    echo "ERROR: FIRE_SWEEP is not set." >&2
    echo '       sbatch --export=ALL,FIRE_SWEEP=<sweep-name> tamia/run_sweep.slurm' >&2
    echo "       (or just use tamia/submit.sh <sweep-name>)" >&2
    exit 1
fi

TRIALS_PER_GPU="${FIRE_TRIALS_PER_GPU:-2}"
# Wall-clock a single trial is assumed to need, in minutes. A worker will not
# claim a new trial when less than this remains, because an unfinished trial is
# wasted work -- there is no checkpoint to resume from.
TRIAL_BUDGET_MIN="${FIRE_TRIAL_BUDGET_MIN:-150}"
# Debug hook: substitute a no-op (or `echo`) for the training wrapper to exercise
# the queue without burning GPU time. Used by the offline test in the README.
TRIAL_CMD="${FIRE_TRIAL_CMD:-}"

# --- 1. Repo root ------------------------------------------------------------
# Not from BASH_SOURCE: Slurm copies this script to the node's spool directory,
# where it expands to /var/spool/slurmd/job<N>/slurm_script. SLURM_SUBMIT_DIR is
# where sbatch was run, which the relative slurm_logs/ paths above already assume
# is the repo root.
if [[ -n "${FIRE_REPO_ROOT:-}" ]]; then
    :                                   # explicit override wins
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    FIRE_REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    FIRE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export FIRE_REPO_ROOT

WRAPPER="$FIRE_REPO_ROOT/bash_scripts/run_vision_st.sh"
if [[ ! -f "$WRAPPER" || ! -d "$FIRE_REPO_ROOT/sweep_config" ]]; then
    cat >&2 <<MSG
ERROR: '$FIRE_REPO_ROOT' is not the FIRE_playground repo root.
       Submit from the repo root:
           cd <repo> && sbatch --export=ALL,FIRE_SWEEP=$SWEEP tamia/run_sweep.slurm
       or name it explicitly:
           sbatch --export=ALL,FIRE_SWEEP=$SWEEP,FIRE_REPO_ROOT=<repo> tamia/run_sweep.slurm
MSG
    exit 1
fi

# --- 2. Environment (modules, venv, offline W&B for this sweep) ---------------
# shellcheck disable=SC1091
source "$FIRE_REPO_ROOT/tamia/env.sh" "$SWEEP" || exit 1

# --- 3. Preflight ------------------------------------------------------------
TRIALS="${FIRE_TRIALS_FILE:-$FIRE_REPO_ROOT/tamia/trials/$SWEEP.tsv}"
if [[ ! -f "$TRIALS" ]]; then
    cat >&2 <<MSG
ERROR: trial list not found: $TRIALS
       Expand the grid on a login node first:
           python tamia/expand_sweep.py $SWEEP
MSG
    exit 1
fi

# The decoded TinyImageNet cache must already exist. task.py would otherwise
# build it lazily, having every node in the array decode 110k JPEGs at once
# inside a GPU allocation. Build it once with tamia/build_cache.slurm.
if grep -q -- '--task=TinyImageNet' "$TRIALS"; then
    CACHE_VERSION="$(sed -n 's/^[[:space:]]*CACHE_VERSION[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
                         "$FIRE_REPO_ROOT/vision/task.py" | head -1)"
    CACHE="$FIRE_DATA_DIR/tiny-imagenet-200-decoded-v${CACHE_VERSION:-1}.pt"
    if [[ ! -f "$CACHE" ]]; then
        cat >&2 <<MSG
ERROR: TinyImageNet tensor cache not found at: $CACHE
       Build it once, off-GPU:
           sbatch tamia/build_cache.slurm
MSG
        exit 1
    fi
fi

# --- 4. Node shape -----------------------------------------------------------
# How many GPUs this allocation actually got, asked of the allocation rather than
# assumed, so the same script is correct on an h100:4 and an h200:8 node.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NGPU="$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NGPU="$SLURM_GPUS_ON_NODE"
else
    NGPU="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0)"
fi
[[ "${NGPU:-0}" -ge 1 ]] || { echo "ERROR: no GPUs visible in this allocation." >&2; exit 1; }

SLOTS=$(( NGPU * TRIALS_PER_GPU ))
NCPU="${SLURM_CPUS_ON_NODE:-$(nproc)}"
THREADS=$(( NCPU / SLOTS )); (( THREADS >= 1 )) || THREADS=1
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

# --- 5. Queue state ----------------------------------------------------------
STATE="$FIRE_STATE_ROOT/$SWEEP"
CLAIMED="$STATE/claimed"
DONE="$STATE/done"
FAILED="$STATE/failed"
mkdir -p "$CLAIMED" "$DONE" "$FAILED"

# --- 6. Deadline guard -------------------------------------------------------
# Seconds left in this allocation, or -1 when it cannot be determined (in which
# case the guard is skipped rather than silently refusing to start any trial).
JOB_END_EPOCH="${SLURM_JOB_END_TIME:-}"
if [[ -z "$JOB_END_EPOCH" && -n "${SLURM_JOB_ID:-}" ]]; then
    _end="$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null \
            | tr ' ' '\n' | sed -n 's/^EndTime=//p' | head -1)"
    if [[ -n "$_end" && "$_end" != "Unknown" ]]; then
        JOB_END_EPOCH="$(date -d "$_end" +%s 2>/dev/null || echo '')"
    fi
fi

remaining_seconds() {
    if [[ -z "$JOB_END_EPOCH" ]]; then
        echo -1
    else
        echo $(( JOB_END_EPOCH - $(date +%s) ))
    fi
}

TRIAL_BUDGET_S=$(( TRIAL_BUDGET_MIN * 60 ))

# --- 7. Banner ---------------------------------------------------------------
cat <<BANNER
==============================================================
 Sweep         : $SWEEP
 Job / task    : ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-?}} / ${SLURM_ARRAY_TASK_ID:-0}
 Node          : $(hostname)
 GPUs          : $NGPU  ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'n/a'))
 Slots         : $SLOTS  ($TRIALS_PER_GPU per GPU, $THREADS CPU threads each)
 Trials file   : $TRIALS  ($(grep -vc '^#' "$TRIALS") trials)
 Queue state   : $STATE
 W&B           : mode=$WANDB_MODE dir=$WANDB_DIR group=$WANDB_RUN_GROUP
 Trial budget  : ${TRIAL_BUDGET_MIN} min
 Time left     : $(remaining_seconds) s
==============================================================
BANNER

# --- 8. Worker ---------------------------------------------------------------
# Slot s runs on GPU (s % NGPU), so with TRIALS_PER_GPU=2 slots 0 and NGPU share
# GPU 0. Each worker walks the trial list, claims what it can, and stops when the
# list is exhausted or the deadline guard trips.
worker() {
    local slot="$1"
    local gpu=$(( slot % NGPU ))
    local worst=0 ran=0 tid label targs rc left

    while IFS=$'\t' read -r tid label targs; do
        [[ -z "$tid" || "$tid" == \#* ]] && continue
        [[ -e "$DONE/$tid" ]] && continue

        left="$(remaining_seconds)"
        if [[ "$left" -ge 0 && "$left" -lt "$TRIAL_BUDGET_S" ]]; then
            echo "$(( left / 60 )) min left (< ${TRIAL_BUDGET_MIN} min budget) -- not starting another trial"
            break
        fi

        # The claim. mkdir is atomic on Lustre, so exactly one worker anywhere in
        # the array wins; everyone else moves on.
        mkdir "$CLAIMED/$tid" 2>/dev/null || continue

        echo "gpu $gpu  START  $tid  $label"
        local t0 t1
        t0="$(date +%s)"

        # eval: args were shell-quoted by expand_sweep.py, which is what keeps
        # --wandb_project='DST Continual Learning' one argument.
        eval "set -- $targs"

        # </dev/null so the trial cannot consume this worker's copy of the trials
        # file on stdin.
        if [[ -n "$TRIAL_CMD" ]]; then
            $TRIAL_CMD "$@" --gpu "$gpu" </dev/null
        else
            bash "$WRAPPER" "$@" --gpu "$gpu" </dev/null
        fi
        rc=$?

        t1="$(date +%s)"
        if [[ $rc -eq 0 ]]; then
            : > "$DONE/$tid"
            echo "gpu $gpu  DONE   $tid  ($(( (t1 - t0) / 60 )) min)"
        else
            : > "$FAILED/$tid"
            echo "gpu $gpu  FAIL   $tid  rc=$rc  ($(( (t1 - t0) / 60 )) min)  $label"
            [[ $worst -eq 0 ]] && worst=$rc
        fi
        ran=$(( ran + 1 ))
    done < "$TRIALS"

    echo "finished: $ran trial(s), worst rc=$worst"
    return "$worst"
}

# --- 9. Run the slots --------------------------------------------------------
# Plain background subshells, not `srun` steps: this allocation has a single
# task, so a second concurrent srun would block waiting for resources unless
# given --overlap/--exact. Children inherit the allocation's CUDA_VISIBLE_DEVICES
# and the wrapper narrows each to its own GPU via --gpu.
pids=()
for (( s = 0; s < SLOTS; s++ )); do
    # All slots share this task's single .out/.err, so every line is tagged with
    # its slot to keep the interleaved streams readable -- including the lines the
    # wrapper itself prints. Backgrounding the whole subshell (rather than a bare
    # pipeline) keeps $! the pid whose status `wait` reports; pipefail, set above,
    # is inherited, so the worker's status survives the pipe through sed.
    ( worker "$s" 2>&1 | sed -u "s/^/[slot $s] /" ) &
    pids+=("$!")
done

status=0
for i in "${!pids[@]}"; do
    rc=0
    wait "${pids[$i]}" || rc=$?
    [[ $rc -ne 0 && $status -eq 0 ]] && status=$rc
done

# --- 10. Summary -------------------------------------------------------------
total="$(grep -vc '^#' "$TRIALS")"
ndone="$(find "$DONE" -maxdepth 1 -type f | wc -l)"
nfail="$(find "$FAILED" -maxdepth 1 -type f | wc -l)"
cat <<SUMMARY
==============================================================
 Sweep $SWEEP after this task: $ndone/$total done, $nfail failed
 Progress / recovery :  tamia/requeue.sh $SWEEP
 Upload to W&B       :  tamia/sync_wandb.sh $SWEEP     (login node)
==============================================================
SUMMARY

exit "$status"
