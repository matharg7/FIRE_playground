# Sparse continual pre-training sweep

Does sparse training reduce plasticity loss? A model that is warm-started on one
corpus and then trained on another learns the new data worse than a model
trained from scratch. FIRE addresses this by reinitializing weights at the
switch; this sweep asks whether sparse training does too.

Each run trains GPT-2 (124M) over two chunks:

```
chunk 0: wikitext x110      13.15B tokens   saturate the model
              |
         intervention       none / FIRE / (sparse throughout)
              |
chunk 1: openwebtext x0.75   6.78B tokens   can it still learn?
```

Everything runs on one GPU. `train_sparse.py` covers every arm; `train.py` is
the original nanoGPT script, kept for reference and no longer used.

## Setup

```bash
git clone <repo> && cd FIRE_playground
cp bash_scripts/local.sh.example bash_scripts/local.sh   # set FIRE_ACCOUNT
./bash_scripts/build.sh
```

`build.sh` creates the environment, downloads and tokenizes the corpora, then
verifies the result. It is idempotent: re-running takes about ten seconds and
skips whatever is already in place, so it doubles as a health check. Use
`--force` to redo a step.

Tokenizing openwebtext is hours of CPU work. On a cluster whose policy frowns on
that at the login node, run that step inside a job instead:

```bash
./bash_scripts/build.sh env        # login node (needs internet for pip)
./bash_scripts/build.sh download   # login node (needs internet for the data)
salloc --cpus-per-task=8 --mem=32G --time=3:00:00
./bash_scripts/build.sh tokenize   # no internet needed
```

## Launching

```bash
./bash_scripts/sweep.sh            # list the runs and what they cost
./bash_scripts/sweep.sh --submit   # submit them as one job array
```

The dry run prints every run, its estimated hours, the cluster, the W&B mode and
the account. Nothing is submitted without `--submit`.

```
dense_vanilla                  11.1h  --method=vanilla --sparsifier=dense ...
rigl_s0.3_pr0.3_nmu100         13.3h  --method=vanilla --sparsifier=rigl ...
...
cluster: rorqual   W&B: offline   account: def-yani
30 runs · 385 GPU-hours · longest 13.3h (walltime 18:00:00)
```

Run a subset with `--filter`, which matches on the run name:

```bash
./bash_scripts/sweep.sh --filter dense --submit        # the three dense arms
./bash_scripts/sweep.sh --filter rigl_s0.5 --submit    # one sparsity level
```

## What the sweep contains

| arm | runs | what it answers |
|---|---|---|
| `dense_vanilla` | 1 | the warm-started baseline |
| `dense_full_reset` | 1 | trained on chunk 1 only: the control |
| `dense_fire` | 1 | does FIRE recover the loss? |
| `rigl_s{0.3,0.5,0.8}_pr{0.3,0.7,0.9}_nmu{100,1000,10000}` | 27 | sparse from step one |

The RigL grid varies sparsity, the fraction of weights rewired per update, and
how many updates happen across the run.

**Compare RigL configs within a sparsity level, not across.** A lower sparsity
has more capacity and will look better for reasons unrelated to plasticity.
Turning a config into a plasticity claim needs its own `full_reset` control at
the same sparsity, which this sweep does not yet include.

## Monitoring

```bash
squeue --me
tail -f logs/language/fire_lang_<jobid>_<task>.out
```

Each run logs to W&B, and prints machine-readable lines to its log:

```
EVAL chunk 1 global_iter 27000 local_iter 243 train 3.912 val 3.945
SPARSE itop_rate 0.41 mask_sparsity 0.80 weight_sparsity 0.80 ...
FIRE_SPECTRUM transformer.h.0.attn.c_proj before 1.41 after 1.29
```

A run that finishes writes a `DONE` marker in its output directory and then
**deletes its checkpoints** — about 19GB each, and W&B holds everything needed
for analysis. A run that fails keeps them, so it can be inspected.

### W&B

Set by the cluster profile, overridden in `local.sh`, or per command:

```bash
FIRE_WANDB_MODE=online ./bash_scripts/sweep.sh --submit
```

`offline` is the default where compute nodes have no internet. Push those runs
afterwards from somewhere that does:

```bash
./bash_scripts/sync_wandb.sh --dry-run   # what would be pushed
./bash_scripts/sync_wandb.sh             # push; safe to re-run
```

## Configuration

| file | holds | committed |
|---|---|---|
| `bash_scripts/sweep.conf` | the grid and the training settings | yes |
| `bash_scripts/clusters/<name>.sh` | modules, GPU flags, W&B default | yes |
| `bash_scripts/local.sh` | your account, your paths | no |

Paths come from `bash_scripts/env.sh` and can all be overridden:

```
FIRE_WORK        everything generated       $SCRATCH/fire, else <repo>/.work
  ├── venv       python environment
  ├── data       tokenized .bin files
  ├── output     run outputs and DONE markers
  ├── wandb      W&B run directories
  └── cache      HuggingFace / tiktoken
```

### Another cluster

Copy `bash_scripts/clusters/default.sh` to `<cluster>.sh` and set its modules
and how it wants a GPU requested. Detection uses `CC_CLUSTER` (set on all
Alliance machines), else the hostname; an unrecognised Alliance cluster falls
back to the shared `drac.sh` profile. Nothing else needs changing.

## Tests

```bash
cd language
pytest -m "not gpu"                          # ~2 min, no GPU needed
pytest -m gpu -k "not DDP"                   # inside a GPU job step
SLURM_JOB_ID=<job> pytest -m gpu -k DDP      # from outside a step
```

The DDP tests launch their own ranks with `srun`, so they must run outside a job
step: a step cannot create a step. Run the CPU tests on a compute node, not a
login node, where thread oversubscription makes them ten times slower.

## Measured numbers

One H100, GPT-2 124M, fp16, batch 60 x 8 accumulation (491,520 tokens/step):

| | tokens/sec | MFU | peak memory |
|---|---|---|---|
| dense, compiled | 500,259 | 43.2% | 41.6 GB |
| RigL, compiled | 417,335 | 36.1% | 60.9 GB |

Sparse training costs 3-6% per ordinary step; a mask update costs ~102ms and
happens every `delta_t` steps. Data loading is ~2% of step time once the page
cache is warm, and roughly 4x that when it is cold, which is why the launcher
pre-reads the `.bin` files before training starts.
