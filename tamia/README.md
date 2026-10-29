# Running the vision sweeps on tamIA

Everything in this directory exists because tamIA differs from nibi in two ways that
break the `slurm/` setup outright:

| | nibi | tamIA |
|---|---|---|
| Compute-node internet | yes | **no** |
| Allocation granularity | per GPU (`--gpus=h100:1`) | **whole node**, all GPUs must be used |

No internet means `wandb agent` cannot ask the sweep server for its next
configuration, so the grid is expanded up front on a login node and handed out by a
queue on scratch. Whole nodes mean an array task fills 4 (H100) or 8 (H200) GPUs at
once instead of taking one.

Nothing outside this directory changed. `vision/train_st.py` and
`bash_scripts/run_vision_st.sh` are used exactly as they are on nibi, and the
sweeps still come from `sweep_config/*.yaml` — the same files, no copies.

---

## Quick reference

```sh
cd <repo>                                            # always submit from the repo root

# one-time
./build_env.sh                                       # LOGIN node (needs internet)
sbatch tamia/build_cache.slurm                       # decode TinyImageNet

# run a sweep
tamia/submit.sh tin_vgg16_class_inc_set_rigl         # expands, sizes the array, submits
tamia/submit.sh <sweep> --dry-run                    # see the plan first
tamia/submit.sh <sweep> --gpu-type h200              # 8 GPUs per node instead of 4

# while it runs
squeue -u $USER
tail -f slurm_logs/tamia_<sweep>_<jobid>_0.out
tamia/requeue.sh <sweep>                             # progress: done / failed / remaining

# after it runs
tamia/sync_wandb.sh <sweep>                          # LOGIN node: upload offline runs
```

Sweep names are the `sweep_config/*.yaml` basenames, e.g. `tin_vgg16_class_inc_gmp`.

Three sets of logs, all inside the repo:

| | |
|---|---|
| `wandb_offline/<sweep>/wandb/` | the offline W&B runs, uploaded by `sync_wandb.sh` |
| `logs/<log-subdir>/<run>.out` | one trial's stdout (`--log-subdir` in the sweep YAML) |
| `slurm_logs/tamia_<name>_<jobid>_<task>.out` | one array task, all its slots interleaved |

---

## 1. One-time setup

### 1.1 Account

tamIA needs an RAP whose name starts with `aip-`. It is **not** your nibi
`rrg-yani` / `def-yani`. The scripts default to `aip-yani`; confirm it under
*Resource Allocation Projects* at <https://ccdb.alliancecan.ca/>, and if it differs:

```sh
export FIRE_ACCOUNT=aip-something        # or pass --account to submit.sh
```

Then fix the `#SBATCH --account=` line in `run_sweep.slurm` and `build_cache.slurm`
so raw `sbatch` works too. (Access also requires the tamIA access request in CCDB
and the *General Access to PAICE Systems* form. The cluster is only reachable from
inside Canada.)

### 1.2 Environment and datasets — on a **login node**

Compute nodes have no internet, so the wheelhouse install and the dataset downloads
have to happen here:

```sh
cd <repo>
./build_env.sh --skip-cache          # venv at $SCRATCH/fire-env + raw datasets
echo "<your W&B API key>" > ~/.wandb_token && chmod 600 ~/.wandb_token
```

> **Module versions.** `tamia/env.sh` loads
> `StdEnv/2023 python/3.11.5 scipy-stack/2026a arrow/24.0.0 cuda/13.2`, which is what
> nibi carries. `StdEnv/2023` is tamIA's documented standard environment, but the
> other four may not exist at these versions. Check and override:
>
> ```sh
> module avail python ; module avail cuda ; module avail arrow ; module avail scipy-stack
> export FIRE_PYTHON_MODULE=python/3.11.9      # for example
> ```
>
> Once you know the right set, edit the defaults at the top of `tamia/env.sh` so you
> do not have to export them every session. `build_env.sh` has its own `MODULES=(...)`
> line that needs the same edit.

### 1.3 TinyImageNet cache — a job

`vision/task.py` needs a decoded tensor cache. Building it lazily would have every
node in a sweep decode 110k JPEGs at once inside a GPU allocation, so
`run_sweep.slurm` refuses to start without it:

```sh
sbatch tamia/build_cache.slurm       # CPU-only node, 8 cores, 1 h
```

Skip this if you only run CIFAR sweeps.

---

## 2. Interactive jobs

For debugging, a single run, or anything you want to watch.

```sh
# H100 node: 4 GPUs, 48 cores
salloc --account=aip-yani --nodes=1 --ntasks=1 \
       --gpus=h100:4 --cpus-per-task=48 --mem=0 --time=1:00:00

# H200 node: 8 GPUs, 64 cores
salloc --account=aip-yani --nodes=1 --ntasks=1 \
       --gpus=h200:8 --cpus-per-task=64 --mem=0 --time=1:00:00
```

Then, on the node:

```sh
cd <repo>
source tamia/env.sh                       # modules + venv; leaves W&B online

bash_scripts/run_vision_st.sh \
    --task TinyImageNet --model VGG16 --benchmark class_incremental \
    --sparsifier rigl --sparsity 0.7 --pruning-ratio 0.3 \
    --num-mask-updates 10 --use-cosine-lr True --seed 0 \
    --gpu 0
```

Output goes to `logs/<run_name>.out` as usual.

**Three things to keep in mind:**

- **All 4 (or 8) GPUs are yours** and the policy is to use them. For a single debug
  run that is unavoidable, so keep the job short. Otherwise start one run per GPU in
  the background (`--gpu 0`, `--gpu 1`, … — `train_st.py` is single-GPU, so a run
  uses exactly the one you give it), or just use `tamia/submit.sh --test`.
- **Jobs must be at least 1 h** (5 min for test jobs) and at most 24 h.
- **W&B is online here unless you make it offline.** `source tamia/env.sh` with no
  argument deliberately does *not* set `WANDB_MODE`, and a compute node cannot reach
  api.wandb.ai — so `wandb.init` will stall retrying before dropping to offline mode
  on its own. Set it yourself:

  ```sh
  export WANDB_MODE=offline     # log to disk, sync later
  export WANDB_MODE=disabled    # throw the run away (debugging)
  ```

  Or `source tamia/env.sh <sweep-name>`, which sets up the whole offline block —
  `WANDB_DIR`, group and tags included — exactly as a batch job gets it.

**VSCode:** forbidden on tamIA login nodes — it leaves heavy stale processes that
affect everyone. It is allowed on compute nodes, so `salloc` first and point the
Remote-SSH session at the allocated node (`squeue -u $USER -o '%N'`). Check for
leftovers with `ps -u $USER` on the login node and kill any stragglers.

---

## 3. Batch sweeps

### 3.1 The one-liner

```sh
cd <repo>
tamia/submit.sh tin_vgg16_class_inc_set_rigl
```

`submit.sh` expands the grid, counts what is still outstanding, works out how many
nodes that needs, prints the plan, and submits. Use `--dry-run` first if you want to
see the plan without submitting.

Useful options:

| Option | Default | |
|---|---|---|
| `--gpu-type h100\|h200` | `h100` | 4 GPUs/48 cores, or 8 GPUs/64 cores |
| `--trials-per-gpu N` | `2` | concurrent trials sharing each GPU |
| `--time HH:MM:SS` | `23:00:00` | per array task; tamIA's cap is 24 h |
| `--trial-budget MIN` | `150` | how long one trial is assumed to need |
| `--nodes N` | computed | force the array size |
| `--account NAME` | `aip-yani` | |
| `--test` | | 1 node, 30 min — the smoke test |
| `--dry-run` | | print, submit nothing |

### 3.2 How the array is sized

```
slots per node = gpus x trials-per-gpu          # 8 on an H100 node by default
rounds         = floor(walltime / trial-budget) # 9 at 23 h and 150 min
capacity       = slots x rounds                 # ~72 trials per array task
array size     = ceil(outstanding / capacity)
```

With the defaults (`h100`, 2 trials/GPU, 23 h):

| Sweep family | Trials | Nodes (h100) | Nodes (h200) |
|---|---|---|---|
| `*_dense` | 3 | 1 | 1 |
| `*_static` | 15 | 1 | 1 |
| `*_gmp` | 45 | 1 | 1 |
| `*_set_rigl` | 180 | 3 | 2 |

**Over-provisioning is harmless.** A surplus array task finds every trial already
claimed and exits in seconds — the same property the W&B agents had. Under-provisioning
just leaves trials for a later submission. So round up when unsure.

### 3.3 Two trials per GPU

The default packs 2 trials onto each GPU, matching what you run on nibi, because one
trial leaves an H100 underused. They share the GPU's memory and SMs, so each is slower
than it would be alone — a throughput trade, not free doubling. Drop to
`--trials-per-gpu 1` if you see GPU OOM or the per-trial wall clock creeping past the
budget. On H200s (141 GB) you could try 3–4.

### 3.4 The raw path, without `submit.sh`

```sh
python tamia/expand_sweep.py tin_vgg16_class_inc_set_rigl    # -> tamia/trials/<sweep>.tsv
mkdir -p slurm_logs
sbatch --array=0-2 --account=aip-yani \
       --export=ALL,FIRE_SWEEP=tin_vgg16_class_inc_set_rigl \
       tamia/run_sweep.slurm
```

`run_sweep.slurm`'s own `#SBATCH` headers are the H100 defaults; `submit.sh` overrides
them on the command line. Submit from the repo root either way — the relative
`slurm_logs/` paths and `SLURM_SUBMIT_DIR` both depend on it.

### 3.5 Monitoring

```sh
squeue -u $USER
tail -f slurm_logs/tamia_<sweep>_<jobid>_0.out    # all slots, each line tagged [slot N]
tamia/requeue.sh <sweep>                          # done / failed / remaining
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
```

The tamIA portal at <https://portail.tamia.ecpia.ca/> shows live CPU/GPU utilisation
per job — worth a look on the first real sweep to confirm the node is actually full
and to sanity-check `--trials-per-gpu`.

### 3.6 Worked example: exactly one trial per GPU

The default packs 2 trials onto a GPU and lets each slot work through several in
turn. The opposite arrangement — one trial per GPU, one round, the whole sweep
running at once — is two submissions, sized so the slots add up to the trial count:

```sh
# 180 trials = 22 H200 nodes x 8 GPUs (176) + 1 H100 node x 4 GPUs (4)
tamia/submit.sh tin_vgg16_sample_inc_set_rigl \
    --gpu-type h200 --nodes 22 --trials-per-gpu 1 \
    --time 4:20:00 --trial-budget 240 --job-name tin_vgg16_sample_inc_set_rigl_h200

tamia/submit.sh tin_vgg16_sample_inc_set_rigl \
    --gpu-type h100 --nodes 1 --trials-per-gpu 1 --no-expand \
    --time 4:20:00 --trial-budget 240 --job-name tin_vgg16_sample_inc_set_rigl_h100
```

Two details make it behave:

- **`--trial-budget 240` against `--time 4:20:00` (260 min)** is what holds each
  slot to a single trial. The deadline guard will not claim a trial with less than
  the budget left, and once one has run there never is — so a slot takes one trial
  and stops. It still leaves 20 min of head-room at the start for modules, the venv
  and the preflight, which a budget any closer to 260 would eat into: too close and
  no worker claims anything and the job looks mysteriously empty.
- **`--no-expand` on the second submission.** `expand_sweep.py` rewrites
  `tamia/trials/<sweep>.tsv` in place, and by then the first job's workers are
  reading it line by line. The expansion is deterministic, so there is nothing to
  regenerate — just don't truncate the file under a running job.

Order does not matter and neither does over-provisioning: the claim is atomic, so
whichever node reaches a trial first owns it, and a slot that finds nothing left
exits in seconds. The second submission reports the first job's in-flight trials as
"claimed but unfinished" — expected while it is running, and not a reason to
`--clear-stale`.

---

## 4. Offline W&B

### How it works

`tamia/env.sh <sweep>` sets `WANDB_MODE=offline` and points `WANDB_DIR` at a
per-sweep directory. `train_st.py` needs no change — it already reads the mode from
the environment. Runs are written in full to a directory of their own **inside the
repo**:

```
<repo>/wandb_offline/<sweep>/wandb/offline-run-<timestamp>-<id>/
```

`wandb_offline/` is git-ignored, and `$FIRE_WANDB_ROOT` overrides where it goes
(`export FIRE_WANDB_ROOT=$SCRATCH/wandb` puts it back on scratch). Sweeps that ran
before this wrote to `$SCRATCH/wandb/<sweep>/`; `sync_wandb.sh` still searches there
too, so an older backlog uploads without anything being moved.

The queue's claim/done markers stay on `$SCRATCH` — those are thousands of empty
files, which is what `/project`'s inode quota minds. A sweep's W&B runs are a few
thousand real files; check headroom with `diskusage_report` before a large one.

Then, **from a login node** (the only place with internet):

```sh
tamia/sync_wandb.sh tin_vgg16_class_inc_set_rigl
tamia/sync_wandb.sh <sweep> --dry-run      # list what would upload
tamia/sync_wandb.sh --all                  # every sweep on disk
```

It is idempotent — W&B marks a synced run and the script skips it — so re-running
after an interruption is safe. Uploads go in chunks of 25 (`--batch N`) so stopping
between chunks loses nothing.

### Finding the runs afterwards

Offline runs are not attached to a W&B *sweep* object, so there is no sweep page and
no parallel-coordinates view keyed to a sweep ID. Instead, `env.sh` sets
`WANDB_RUN_GROUP` and `WANDB_TAGS`, so in the **DST Continual Learning** project you
filter by:

```
Group == tin_vgg16_class_inc_set_rigl        # one sweep
Tag   == tamia                               # everything from this cluster
```

Grouping by `Group` gives you the same per-sweep roll-up you are used to. Check the
run count there against the number `sync_wandb.sh` reported.

### Disk hygiene

The venv, the datasets and the queue state live on `$SCRATCH`, which is **not backed
up and is purged periodically**. The offline runs no longer do, so a purge does not
take them with it — but they now count against the project quota instead. Sync a
sweep soon after it finishes rather than letting months of runs accumulate; once
synced, W&B is the copy of record and `wandb_offline/<sweep>` can go. Check space
with `diskusage_report`.

---

## 5. Recovering a partial sweep

`train_st.py` has no checkpointing and tamIA caps jobs at 24 h, so a trial killed by
the wall clock has to be run again from scratch. The queue tracks this per trial:

```
$SCRATCH/fire-tamia/state/<sweep>/claimed/<tid>/   a worker took it (atomic mkdir)
$SCRATCH/fire-tamia/state/<sweep>/done/<tid>       it exited 0
$SCRATCH/fire-tamia/state/<sweep>/failed/<tid>     it exited non-zero
```

A trial that is **claimed but neither done nor failed** was abandoned — its node hit
the wall clock or crashed. Its claim blocks any future worker from retrying it, so
release it first:

```sh
tamia/requeue.sh <sweep>                  # report only; always safe
tamia/requeue.sh <sweep> --clear-stale    # release abandoned trials
tamia/requeue.sh <sweep> --retry-failed   # also re-queue the ones that errored
tamia/submit.sh  <sweep>                  # re-submit; only the outstanding trials run
tamia/requeue.sh --list                   # every sweep with state on disk
```

`--clear-stale` refuses to run while jobs for that sweep are still queued or running,
since those claims are live — releasing one would let a second worker start the same
trial and produce a duplicate W&B run. `--force` overrides it if you are certain.

`--reset` wipes all state for a sweep and re-runs everything; it asks for
confirmation, and note it does **not** delete the offline runs already on disk, so
syncing afterwards would produce duplicates unless you also clear
`wandb_offline/<sweep>`.

To find *why* a trial failed, its own stdout is in `logs/<log-subdir>/<run_name>.out`
(the subdir comes from the sweep YAML's `command:` block, exactly as on nibi); the
slot-tagged Slurm log has the `FAIL <tid> rc=N <label>` line telling you which config
it was.

---

## 6. What's in this directory

| File | |
|---|---|
| `env.sh` | **sourced**, not run. Modules, venv, offline W&B. Every other script uses it. |
| `expand_sweep.py` | `sweep_config/<sweep>.yaml` → `tamia/trials/<sweep>.tsv`. Validates every flag against `run_vision_st.sh` before writing. |
| `run_sweep.slurm` | The batch entry point. One array task = one whole node = `gpus x trials-per-gpu` concurrent trials pulled from the queue. |
| `submit.sh` | Expand + size the array + submit. The thing you normally run. |
| `build_cache.slurm` | One-off TinyImageNet decode, CPU node. |
| `sync_wandb.sh` | Login node: upload offline runs. |
| `requeue.sh` | Progress report and recovery. |
| `trials/` | Generated trial lists. Regenerated by `submit.sh` each time; safe to delete. |

---

## 7. Differences from the nibi setup

| | `slurm/dst_vision_sweep*.slurm` (nibi) | `tamia/` |
|---|---|---|
| Dispatch | `wandb agent` pulls from the sweep server | grid expanded locally; trials claimed via atomic `mkdir` on scratch |
| Allocation | 1 GPU per array task | whole node, 4 or 8 GPUs |
| Concurrency | 2 agents sharing 1 GPU | `gpus x trials-per-gpu` slots per node, dynamically balanced |
| W&B | online, live dashboard | offline, `sync_wandb.sh` afterwards; grouped by `Group ==` |
| Sweep setup | `wandb sweep <yaml>` → sweep ID → `sbatch ... "<ID>"` | `tamia/submit.sh <sweep-name>` |
| Resuming | grid sweeps may not reissue crashed trials; check for gaps manually | `requeue.sh` + re-submit runs exactly what is outstanding |
| Account | `rrg-yani` | `aip-yani` (verify in CCDB) |
| Job limits | nibi's 12 h scheduling bucket | 1 h ≤ job ≤ 24 h, ≤ 1000 jobs |

---

## 8. Gotchas

- **Submit from the repo root.** `slurm_logs/` paths and `SLURM_SUBMIT_DIR` both
  assume it. `submit.sh` `cd`s there for you.
- **`wandb sync` only works on a login node.** Running it inside a job will hang;
  the script warns if it detects `$SLURM_JOB_ID`.
- **A sweep YAML that is not `method: grid`** cannot be expanded offline — bayes and
  random need the server to sample. `expand_sweep.py` says so and stops.
- **`sweep_config/dst_vision_sweep.yaml` is stale** (it targets an older LoRA/DST
  program) and `expand_sweep.py` correctly rejects it, listing the flags
  `run_vision_st.sh` does not accept. That is the validator working, not a bug.
- **No `crontab` on tamIA**, so nothing here can be scheduled; there is an automation
  node (`robot.tamia.ecpia.ca`) if you ever need unattended submission.
- **Use what you request.** The portal will show if you asked for 48 cores and used
  4. Adjust `--trials-per-gpu` rather than leaving a node idle.
