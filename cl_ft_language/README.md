# cl_ft_language — sparse training for continual fine-tuning on TRACE

Does sparse training help an LLM learn a sequence of tasks without forgetting
the earlier ones? This folder runs that experiment: dense fine-tuning against
[RigL](https://arxiv.org/abs/1911.11134) sparse training on the
[TRACE](https://arxiv.org/abs/2310.06762) benchmark of 8 sequential tasks,
with **full cumulative replay** — task *t* trains on the union of the training
data of tasks 0..*t*.

Everything here is self-contained: its own virtual environment, scripts and
tests, with no imports from `language/` or `bash_scripts/`. The only outside
dependency is the sparsimony library vendored at `vision/sparsimony`, installed
into the venv.

---

## 1. Quick start on a new cluster

```bash
# from the repo root
cp cl_ft_language/scripts/local.sh.example cl_ft_language/scripts/local.sh
$EDITOR cl_ft_language/scripts/local.sh          # set TRACE_ACCOUNT

./cl_ft_language/scripts/build_env.sh env        # venv + packages (~10 min)
./cl_ft_language/scripts/build_env.sh download   # TRACE data + models, login node
./cl_ft_language/scripts/build_env.sh check      # versions and imports

./cl_ft_language/scripts/sweep.sh                # list the runs (no submission)
./cl_ft_language/scripts/sweep.sh --submit       # queue them
```

Nothing hardcodes a path: every location comes from `scripts/env.sh`, which
derives them from `$SCRATCH` and the cluster name, and can be overridden by
exporting variables or editing the gitignored `scripts/local.sh`.

| Variable | Default | Holds |
|---|---|---|
| `TRACE_WORK` | `$SCRATCH/fire` | everything generated |
| `TRACE_VENV` | `$TRACE_WORK/venv-trace` | the virtual environment |
| `TRACE_DATA_DIR` | `$TRACE_WORK/data/trace` | the TRACE benchmark json |
| `TRACE_OUTPUT_DIR` | `$TRACE_WORK/output/trace` | run outputs |
| `TRACE_WANDB_DIR` | `$TRACE_WORK/wandb` | offline W&B runs |
| `HF_HOME` | `$TRACE_WORK/cache/huggingface` | model weights |
| `TRACE_ACCOUNT` | *(unset)* | the Slurm account |
| `TRACE_WANDB_MODE` | `offline` on Alliance clusters | `offline`, `online` or `disabled` |

On a cluster that needs particular modules, set `TRACE_MODULES` in
`scripts/local.sh`. The defaults are `StdEnv/2023 python/3.11.5
scipy-stack/2026a arrow/24.0.0 cuda` (with `cuda/13.2` pinned on rorqual).

---

## 2. What the experiment does

**Tasks**, in TRACE's order, with the score each one reports (all on a 0–1 scale):

| # | Task | What it is | Language | Score |
|---|---|---|---|---|
| 1 | C-STANCE | stance detection | Chinese | accuracy |
| 2 | FOMC | hawkish / dovish / neutral | English | accuracy |
| 3 | MeetingBank | meeting summarisation | English | ROUGE-L |
| 4 | Py150 | code completion | Python | edit similarity |
| 5 | ScienceQA | multiple-choice QA with reasoning | English | accuracy |
| 6 | NumGLUE-cm | commonsense arithmetic | English | accuracy |
| 7 | NumGLUE-ds | domain-specific arithmetic | English | accuracy |
| 8 | 20Minuten | text simplification | German | SARI |

**The loop.** For each task *t*:

1. clear the AdamW state (`--reset_optimizer`, on by default), keeping the
   optimizer object so sparsimony's hooks survive;
2. restart the drop-fraction schedule if `--drop_fraction_schedule per_task`;
3. train on the cumulative data of tasks 0..*t*, with warmup and a cosine
   learning rate that restarts each task;
4. evaluate tasks 0..*t* by generation (plus task *t*+1 when
   `--eval_lookahead`), and record the scores;
5. save the scores, predictions, a checkpoint and the W&B log.

**Metrics.** From the score matrix `R[t][i]` (task *i* after training through
task *t*):

- **OP** — mean score over the tasks seen so far;
- **BWT** — average change in each task's score since it was trained;
  negative means forgetting;
- **FWT** — how much better each task is *before* training on it than it was
  zero-shot, which needs `--eval_lookahead` and `--eval_zero_shot`.

**Sparsity.** The pretrained weights are magnitude-pruned to `--sparsity` at
the start; RigL then prunes and regrows every `delta_t` steps.

`--sparse_targets` chooses which Linear weights inside the decoder blocks get a
mask. The embeddings, the tied `lm_head` and the norms are never candidates
under any setting — `lm_head` is tied to the embedding in both models, so
masking it would mask the embedding too.

| `--sparse_targets` | masks | Qwen2.5-0.5B | SmolLM2-135M |
|---|---|---|---|
| `all_linear` | attention and MLP | 72.4% | 78.9% |
| **`mlp`** (default) | `gate/up/down_proj` | **63.5%** | **59.2%** |
| `up_down` | `up/down_proj` only | 42.3% | 39.5% |

The default prunes the MLP only; attention stays dense. `gate_proj` is included
because it is an up-projection by shape (hidden → intermediate), the twin of
`up_proj` in the SwiGLU pair.

**`--sparsity` is measured within the targeted set, not over the model**, so
the same value prunes less as the set narrows: at `mlp`, `--sparsity 0.2`
removes 12.7% of Qwen's weights; at `up_down` it removes 8.5%. The sparsifier
prints `coverage` and `effective_global_sparsity` at startup, and comparing a
sparsity value across two target sets is meaningless without them.

---

## 3. Layout

```
cl_ft_language/
├── README.md, PLAN.md          this file; the plan and running decisions
├── src/
│   ├── config.py               every setting, and its CLI flag
│   ├── data.py                 tasks, cumulative replay, length-grouped batching
│   ├── evaluate.py             generation, per-task scores, eval loss
│   ├── cl_metrics.py           the score matrix, OP / BWT / FWT
│   ├── sparse_utils.py         RigL and SET on HF decoders; masks and metrics
│   ├── train.py                the continual training loop (entry point)
│   └── profile_trace.py        cost measurements (data, throughput, eval)
├── scripts/
│   ├── env.sh                  paths, modules and the cluster profile
│   ├── local.sh.example        per-user settings (copy to local.sh)
│   ├── build_env.sh            env | download | check
│   ├── sweep.conf              resources, and what the sbatch sweep runs
│   ├── sweep.sh                builds and submits the job array
│   ├── sweep_rigl.yaml         W&B sweep: the RigL grid
│   ├── sweep_dense.yaml        W&B sweep: the dense baseline
│   ├── launch_wandb_sweep.sh   registers a sweep, queues one agent per run
│   ├── wandb_agent.sh          one agent = one array task = one run
│   ├── run_trace.sh            one run (an array task, or standalone)
│   ├── profile.sbatch          the profiling job
│   └── sync_wandb.sh           push offline runs, from a login node
├── tests/                      pytest; GPU tests are marked `gpu`
└── trace/                      the vendored TRACE benchmark (see §7)
```

---

## 4. Running

### The W&B sweep (preferred)

The grid lives in a W&B sweep config, and each Slurm array task runs one agent
that takes exactly one configuration (`wandb agent --count 1`). One run per
job, so a job's walltime is one run's walltime and a crash costs one run rather
than the rest of the queue. The W&B server hands out each grid point once, so
agents never collide and the agent count is independent of the grid size.

```bash
./cl_ft_language/scripts/launch_wandb_sweep.sh scripts/sweep_rigl.yaml           # dry run
./cl_ft_language/scripts/launch_wandb_sweep.sh scripts/sweep_rigl.yaml --submit
./cl_ft_language/scripts/launch_wandb_sweep.sh scripts/sweep_dense.yaml --submit
./cl_ft_language/scripts/launch_wandb_sweep.sh --resume <sweep_id> --agents 2 --submit
```

The dry run prints the grid, the run count and the resources, and contacts
nothing. Everything lands in the `dst_trace_benchmark` project.

`method: grid`, so every combination becomes exactly one run — this is a
designed experiment, not a hyperparameter search. To add a dimension, promote a
parameter from `{value: x}` to `{values: [a, b]}`. Resources come from
`sweep.conf` (`TIME`, `CPUS_PER_TASK`, `MEM_PER_CPU`, `GPUS_PER_TASK`).

Dense is a **separate** yaml on purpose: it ignores every sparse knob, so in
one grid it would be multiplied out into identical duplicate runs.

This needs `TRACE_WANDB_MODE=online`, since agents pull their configs from the
W&B server; the launcher refuses to run otherwise.

### The sbatch sweep (no W&B server involved)

`scripts/sweep.conf` holds the settings; edit it rather than `sweep.sh`. By
default it runs 4 jobs: dense, and RigL at 5%, 10% and 20% sparsity, all with
`grow_init=previous` and `drop_fraction_schedule=per_task`.

```bash
./cl_ft_language/scripts/sweep.sh                       # list the runs
./cl_ft_language/scripts/sweep.sh --submit              # submit the array
./cl_ft_language/scripts/sweep.sh --filter rigl --submit    # only RigL runs
./cl_ft_language/scripts/sweep.sh --dense-only --submit
```

Add seeds or arms by editing `SEEDS`, `SPARSITIES`, `GROW_INITS` and
`DROP_FRACTION_SCHEDULES` in `sweep.conf`.

### A single run

```bash
sbatch --account=$TRACE_ACCOUNT --gpus-per-task=1 --cpus-per-task=6 \
       --mem-per-cpu=8G --time=08:00:00 \
       cl_ft_language/scripts/run_trace.sh --sparsifier rigl --sparsity 0.1
```

Or interactively, which is also how to debug:

```bash
salloc --account=$TRACE_ACCOUNT --gpus-per-task=1 --cpus-per-task=6 \
       --mem-per-cpu=8G --time=1:00:00
source cl_ft_language/scripts/env.sh && trace_load_modules && trace_activate
python cl_ft_language/src/train.py --model HuggingFaceTB/SmolLM2-135M \
    --tasks C-STANCE,FOMC --subset 500 --max_train_per_task 32 \
    --max_eval_per_task 8 --wandb_mode disabled
```

### Afterwards

```bash
./cl_ft_language/scripts/sync_wandb.sh --dry-run   # on a login node
./cl_ft_language/scripts/sync_wandb.sh
```

Only runs with a `DONE` marker are synced, and each is pushed once.

---

## 5. Settings worth knowing

Everything in `src/config.py` is a flag; `--max-prompt-len` and
`--max_prompt_len` both work. The ones that matter most:

| Flag | Default | Meaning |
|---|---|---|
| `--model` | SmolLM2-135M | the sweep uses Qwen2.5-0.5B |
| `--tasks` | all 8 | comma-separated, in training order |
| `--subset` | 5000 | training examples per task (5000, 1000 or 500) |
| `--epochs_per_task` | `1` | one number, or one per task (`5,3,7,5,3,5,5,7`) |
| `--max_prompt_len` / `--max_ans_len` | 1024 / 512 | prompts truncate from the left |
| `--batch_size` | 8 | 47 GB peak for Qwen at a 1024 cap |
| `--learning_rate` | 1e-5 | TRACE's full fine-tuning rate |
| `--length_grouped` | True | batches of similar length; 68% → 25% padding, ~1.6× faster |
| `--reset_optimizer` | True | clear AdamW state at each task boundary |
| `--sparsifier` | dense | `dense`, `rigl` or `set` |
| `--sparsity` | 0.1 | fraction of masked weights **within the targeted layers** |
| `--sparse_targets` | mlp | which weights are maskable (`all_linear`, `mlp`, `up_down`) |
| `--sparse_distribution` | erk | how sparsity is split across layers (`erk` or `uniform`) |
| `--grow_init` | zero | `previous` regrows a weight at its former (or pretrained) value |
| `--drop_fraction_schedule` | global | `per_task` restarts the cosine each task |
| `--num_mask_updates` / `--t_end_ratio` / `--pruning_ratio` | 100 / 0.8 / 0.3 | the mask schedule; the sweep uses 1000 updates (`delta_t` 94) |
| `--max_eval_per_task` | 0 (all) | the sweep uses 500 |
| `--eval_zero_shot` / `--eval_loss` / `--eval_lookahead` | True | see §2 |
| `--max_steps_per_task` | 0 | cap steps per task, for smoke runs |

### grow_init and the drop-fraction schedule

These were migrated from the `dst-fire-full-reset` branch of the vision
experiments.

- **`--grow_init previous`.** sparsimony normally sets a regrown weight to 0.
  With `previous` it keeps whatever value is already stored: the value it had
  when it was pruned, or its **pretrained** value if it was never active. That
  matters more here than in the vision experiments, which train from scratch.
  It requires `--weight_decay 0` (the default): AdamW's decoupled decay shrinks
  masked weights too, since they get no gradient, so with decay on the regrown
  value is a shrunken one. The code warns if you do that.
- **`--drop_fraction_schedule per_task`.** With the default `global` schedule,
  the topology freezes at 80% of the *whole run*, which falls inside the last
  task, so the final task mostly trains a frozen mask. `per_task` restarts the
  cosine at each boundary over 80% of *that task's* steps, so every task
  updates its topology and each one still finishes with a settled mask.
  (`--t_end_ratio 1.0` reproduces the vision branch, which used the full task
  length.) The update cadence `delta_t` stays global, so longer tasks get more
  updates — and under cumulative replay the tasks grow.

---

## 6. Outputs

Each run writes to `$TRACE_OUTPUT_DIR/<run_name>/`:

| File | Contents |
|---|---|
| `config.json` | every setting used |
| `scores.json` | the matrix `R`, the zero-shot `baselines`, and eval losses |
| `eval/step<t>_<i>_<task>.json` | score, all metrics, and **every prediction with its label** |
| `eval/zero_shot_<i>_<task>.json` | the same, before any training |
| `summary.json` | OP, BWT, FWT, per-task timings and sparsity at each boundary |
| `checkpoint_task<t>/` | a plain HF checkpoint, plus `masks.pt` for sparse runs |
| `DONE` | written last; `sync_wandb.sh` only syncs runs that have one |

Sparse checkpoints are saved with the masks folded in, so they load with
`AutoModelForCausalLM.from_pretrained` like any other model.

W&B logs `train/*` and `train_loss_by_task/*` every 10 steps, `perf/*`
(tokens/s, peak memory), `dst/*` for sparse runs (mask and weight sparsity,
ITOP rate, pruning ratio), and after each task `eval/<task>`,
`eval_loss/<task>`, `cl/op`, `cl/bwt`, `cl/fwt` and `time/*`.

---

## 7. The vendored TRACE code

`trace/` is the upstream TRACE repository. We reuse its data loading, its
collator and its metrics, and we do not use its DeepSpeed training code or its
CL-method classes (EWC, GEM, …). The changes we made to it:

- **`sari.py`** (new): a local copy of HuggingFace's SARI metric, which
  `datasets.load_metric` no longer provides. It reproduces the published
  reference value.
- **`metrics.py`**: uses it, and `caculate_sari` returns a dict rather than the
  1-tuple a stray comma produced.
- **`inference/prompts.py`** (new): holds the task prompt constants, so the
  collator no longer imports the whole DeepSpeed training stack.
- **`utils/data/data_collator.py`**: skips BOS for tokenizers without one
  (Qwen2.5), and **fixes an upstream bug** — a training example that hit the
  length limit lost its EOS and shifted the label mask one token into the
  prompt. Examples are now built from separately tokenized parts:
  `[BOS] + prompt (truncated from the left) + answer (capped) + EOS`.
- **`evaluations/eval_ScienceQA.py`**: no longer crashes on an empty
  generation.
- **`utils/data/data_utils.py`**, **`utils/model/model_utils.py`**: fixes for
  current torch and transformers.

TRACE's own `scripts/` still contain the authors' hardcoded paths; we don't use
them.

---

## 8. Tests

```bash
source cl_ft_language/scripts/env.sh && trace_load_modules && trace_activate
cd cl_ft_language
python -m pytest -q tests/                 # CPU tests, a few minutes
python -m pytest -q tests/ -m gpu          # inside a GPU job
```

The GPU tests include a 2-task end-to-end run, and they check that sparsity is
exact at every task boundary, that the optimizer's moments are zeroed for
masked weights, and that a sparse checkpoint reloads as a plain model giving
identical outputs.

---

## 9. Cost

Measured on an H100 with `src/profile_trace.py`; see PLAN.md for the details.

| | SmolLM2-135M | Qwen2.5-0.5B |
|---|---|---|
| training | 23.5k tokens/s | 14.2k tokens/s |
| 1 epoch per task | 0.74 h | 1.05 h |
| TRACE's epochs (5,3,7,…) | 3.9 h | 5.6 h |
| 36 evaluations, 500 examples per task | 1.5 h | 1.2 h |
| peak memory (batch 8, 1024 cap) | 21 GB | 47 GB |

RigL costs the same per step as dense; a mask update takes about 0.2 s. Those
training figures predate length grouping, which made training about 1.6×
faster. A sweep run is roughly 5 hours.

To re-measure on different hardware:

```bash
MODELS="Qwen/Qwen2.5-0.5B" sbatch --account=$TRACE_ACCOUNT --time=01:00:00 \
    cl_ft_language/scripts/profile.sbatch
```
