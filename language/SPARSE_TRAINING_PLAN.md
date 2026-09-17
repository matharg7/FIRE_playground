# Sparse training for continual LLM pre-training — plan

**Goal:** add sparsimony sparse training to `language/train.py` and compare
**Dense, Static, GMP, SET, RigL and FIRE** on the two-chunk continual
pre-training setup (Fig 3), mirroring `vision/train_st.py`.

**Hypothesis:** sparse training reduces plasticity loss when training
continues on new data.

**Ground rules**
- Every task is the smallest unit that can be validated on its own, and ships
  with its own tests. Implement one task at a time; tick it off here when its
  tests pass.
- Tests: `🖥` runs on CPU (login node or CPU allocation); `🎮` needs a GPU
  allocation.
- Run CPU tests from `language/`: `pytest -m "not gpu"`.
- Compute nodes have no internet. W&B logs offline; finished runs are synced
  from the login node.

---

## Decisions

| # | Decision | Status |
|---|---|---|
| D1 | All 6 methods run **both chunks in one process** (as in vision). FIRE is applied in-process at the chunk boundary. The existing checkpoint-handoff path is left untouched. | Accepted 2026-09-15 |
| D2 | **One mask-update schedule spanning both chunks:** `t_end = 0.8 × (steps in chunk 0 + steps in chunk 1)`, where steps = optimizer steps actually taken. | Confirmed 2026-09-15 |
| D3 | Sparsify **only the Linear weights inside the 12 blocks** (48 tensors, 84,934,656 weights). `lm_head`/`wte` (tied), `wpe` and LayerNorms stay dense. | Accepted 2026-09-15 |
| D4 | Build and profile at **sparsity 0.9**; choose the grid after profiling. | Accepted 2026-09-15 |
| D6 | **New script `train_sparse.py` runs all six arms** (dense, FIRE, Static, GMP, SET, RigL). `train.py` is the specification: `train_sparse.py` must be functionally identical on the dense and FIRE paths, proven by a parity test. Once parity passes, `train.py` is never run again (kept for reference). | Accepted 2026-09-16 |
| D5 | **1 seed.** Whether runs are independent single-GPU jobs or multi-GPU (DDP) jobs is decided at T4.3 from profiling. | Revised 2026-09-15 |

### D2 at paper scale (tokens per step 491,520)

| Quantity | Value |
|---|---|
| Chunk 0 steps (wikitext, ratio 1.0, replay 400) | 97,283 (`chunk_num_iters` 97,282 + 1: the loop runs one extra step) |
| Chunk 1 steps (openwebtext, ratio 1.0, replay 2) | 36,766 |
| Total steps | 134,049 |
| `t_end` (0.8 × total) | 107,239, i.e. **9,956 steps into chunk 1 (27% of chunk 1)** |
| RigL/SET `delta_t` (`t_end // 500`, as in vision) | 214 |
| GMP `t_accel` (0.2 × total) | 26,809 (inside chunk 0) |
| GMP `delta_t` (`(t_end − t_accel) // 500`) | 160 |

Consequence: RigL and SET change topology only during the first 27% of the
new data; masks are frozen for the remaining 73%.

### D5 rough budget (1 seed, 6 methods, before sparse overhead)

Measured dense throughput: ~310k tokens/s on 1×H100 (batch 8, uncompiled).

| Layout | Per run | Fits 24h? | GPU-hours (6 runs) |
|---|---|---|---|
| 1×H100 per run | ~59 h | No: needs the 3-day partition or T4.4 resume | ~355 |
| 4×H100 DDP per run | ~15 h (assumes linear scaling) | Yes | ~360 |

Replace these with measured numbers in T4.3.

---

## Equivalence with `train.py` (D6)

`train_sparse.py` must implement the **same logic** as `train.py`: same chunk
sizing, same learning-rate schedule, same optimizer reset between chunks, same
eval cadence, same interventions, same checkpoint semantics. It is not required
to reproduce losses bit for bit.

Checked by unit tests on the shared logic (step counts, LR schedule, run naming)
plus an end-to-end smoke run. `train.py` stays in the repo for reference and is
not run again.

---

## Findings that shaped the plan (2026-09-15)

- sparsimony fails to import in `fire-env`: `parametrization/fake_sparsity.py`,
  `dst/static.py` and `parametrization/dfsb.py` import `deepspeed`, which is not
  installed (0.18.1 is in the DRAC wheelhouse). sparsimony also pins
  `transformers<5` (only `transformers_callback.py` uses it).
- RigL, SET and GMP broadcast masks from rank 0 after each topology update;
  Static broadcasts once in `prepare()`. RigL therefore regrows from rank 0's
  local (not all-reduced) gradient.
- `lm_head.weight` is tied to `transformer.wte.weight` (`model.py:138`).
- `train.py` run names / `out_dir` contain no sparse settings, so sparse runs
  would overwrite each other's checkpoints.
- `train.py` never calls `wandb.finish()`.
- pytest is not installed in `fire-env` (9.1.1 is in the wheelhouse).
- W&B was not logged in (no `~/.netrc`, `~/.wandb_token` or `WANDB_API_KEY`).
- GPU walltime limits: interactive 8 h; by-node/by-GPU 3 h, 12 h, 24 h, 3 d, 7 d.

## Bug found and fixed: DDP + sparse deadlock (2026-09-16)

A 2-rank sparse run hung and died with a NCCL ALLREDUCE timeout.

**Cause.** sparsimony registers every mask as a module *buffer*, and DDP defaults
to `broadcast_buffers=True`, so each forward pass broadcasts the masks. Evaluation
runs on rank 0 only, so rank 0 issued collectives the other ranks never matched
and the run desynchronised.

**Fix** (`train_sparse.py`): evaluate on the unwrapped model, and construct DDP
with `broadcast_buffers=False` — sparsimony already keeps masks in sync itself.

This only shows up with sparse + DDP + rank-0-only eval, which is exactly the
sweep configuration, and never in single-GPU tests.

---

## Open questions

- ~~GMP never quite reaches its final sparsity~~ **Resolved 2026-09-16.**
  sparsimony's `AcceleratedCubicScheduler` divides by `t_end` instead of
  `(t_end - t_accel)`, ending at 0.8969 instead of 0.9.
  `sparse_utils.CorrectedAcceleratedCubicScheduler` spans the right interval and
  is the default (`--gmp_correct_cubic`). The vendored library is unchanged, so
  vision's results are unaffected.

---

## Phase 0 — Foundation

- [x] **T0.1 Commit current tooling** 🖥
  - Commit `bash_scripts/setup_language.sh`, `bash_scripts/run_language.sh`,
    `language/data/inspect_data.py`, the dataset-name fixes in both
    `prepare.py`, `.vscode/launch.json`, and this plan.
  - Add `language/output` to `.gitignore`.
  - Validate: `git status` is clean.
  - Done 2026-09-15 (`e115cb2`). Note: `language/output` was committed as a symlink to
    `/scratch/matharg7/fire_output/language` instead of being gitignored.
- [x] **T0.2 Make sparsimony importable** 🖥
  - Install `deepspeed` and `pytest` from the wheelhouse into `fire-env`; add
    them to `setup_language.sh venv`.
  - Test: `from sparsimony import rigl, set, gmp, static` succeeds.
  - Done 2026-09-16: installed deepspeed 0.18.1 + pytest 9.1.1 from the wheelhouse;
    added to `setup_language.sh`. transformers 5.14.1 works despite sparsimony's `<5` pin.
    Tests in `tests/test_env.py`.
- [x] **T0.3 Test scaffold** 🖥
  - `language/tests/conftest.py`: import paths, tiny GPT fixture, tiny
    `train.bin`/`val.bin` fixture. `pytest.ini` with `gpu` and `slow` markers.
  - Test: a trivial test passes; `pytest -m "not gpu"` runs.
  - Done 2026-09-16: `pytest.ini`, `tests/conftest.py` (path setup, `tiny_gpt`,
    `tiny_data_dir`, auto-skip of gpu tests), `tests/helpers.py`. 9 tests pass in 6s.
- [x] **T0.4 CPU end-to-end harness** 🖥 — depends on T0.3
  - `tests/helpers.py` runs a training script in a temp dir with a 2-layer GPT,
    fake data, `--device=cpu --dtype=float32 --compile=False --wandb_log=False`.
  - `train.py` prints machine-readable `EVAL chunk … train … val …` lines (its
    eval print was commented out and used the dead `iter_num`); this is the only
    change made to `train.py`.
  - Test: a `vanilla` run finishes both chunks, losses are finite and decrease.
  - Done 2026-09-16: `tests/helpers.py` (`make_tiny_data`, `stage_run`, `run_train`).
    Threads are pinned to 1 in the subprocess: torch otherwise starts one per visible
    core and the contention dominates. **Run tests on a compute node, not the login
    node**: the suite takes 43 s under `salloc`, and over 10 min on the login node.
- [x] **T0.5 W&B login** 🖥
  - User runs `wandb login` on the login node (key saved to `~/.netrc`).
  - Validate: `wandb.Api().viewer` returns the account; confirm write access to
    the entity (vision uses `ucalgary`).
  - Done 2026-09-15: user `matharg`, credentials in `~/.netrc`, default entity `ucalgary`
    (teams: ucalgary, matharg, enel645); `ucalgary` readable. Write access is confirmed by
    the first real sync (T2.6). `train.py` passes no entity, so runs go to `ucalgary`.

## Phase 1 — Sparse building blocks (`language/sparse_train.py`)

Tasks in this phase don't depend on each other's code.

- [x] **T1.1 Step counter** 🖥 — depends on T0.4
  - `count_chunk_steps(chunk_config, dataset_lens, tokens_per_iter)` returning
    per-chunk optimizer steps, matching `train.py` (including the +1 step and
    zero-ratio chunks). `train.py` calls it.
  - Tests: paper settings give 97,283 / 36,766 / total 134,049; ratio 0 gives 0 steps.
  - Done 2026-09-16 as `count_chunk_steps` in `train_sparse.py`; `tests/test_schedule.py`.
- [x] **T1.2 Layer selection** 🖥
  - `get_sparse_targets(model)` implementing D3.
  - Tests: GPT-2 config gives 48 tensors and 84,934,656 weights; no `lm_head`,
    embeddings or LayerNorm.
  - Done 2026-09-16: `sparse_utils.get_sparse_targets`; verified on the real config.
- [x] **T1.3 Tied-weight safety** 🖥 (tests only)
  - Done 2026-09-16: confirmed `lm_head.weight is wte.weight` after prepare, the
    embedding stays fully dense, and the model still runs.
- [x] **T1.4 Sparsifier factory** 🖥
  - `build_sparsifier(name, model, optimizer, total_steps, hp)` for dense,
    static, gmp, set, rigl; derives `t_end`, `delta_t`, `t_accel` per D2.
  - Done 2026-09-16: `sparse_utils.build_sparsifier` + `sparsifier_schedule`.
    GMP question resolved: `CorrectedAcceleratedCubicScheduler` spans
    t_accel..t_end so GMP reaches 0.9 instead of 0.8969. Controlled by
    `--gmp_correct_cubic` (default True); False reproduces vision's behaviour.
- [x] **T1.5 Optimizer reset compatibility** 🖥 (tests only)
  - Done 2026-09-16: `train_sparse.py` keeps one optimizer object and clears its
    state between chunks, so sparsimony's hook stays registered (vision re-registers
    one per chunk). Tested via `test_optimizer_reset_keeps_sparsifier_working`.
- [x] **T1.6 Mixed precision + micro-steps** 🖥 (tests only)
  - Done 2026-09-16 (GPU): `tests/test_gpu.py::TestMixedPrecision` runs every method
    in fp16 with 4 micro-steps and in bf16; sparsity holds at 0.9 and losses stay finite.
- [x] **T1.7 Sparse metrics** 🖥
  - `sparse_metrics(...)`: mask sparsity, weight sparsity, ITOP, pruning ratio
    (adapted from `vision/dst_log_utils.py`).
  - Done 2026-09-16: `sparse_metrics`, `ITOPTracker`, `get_sparsity_stats` in
    `sparse_utils.py`; logged to wandb and printed as `SPARSE` lines.

## Phase 2 — Build `train_sparse.py` (D6)

New files: `language/config_sparse.py`, `language/train_sparse.py`,
`language/sparse_utils.py`. `train.py` is not modified further.

- [x] **T2.1 Config module** 🖥
  - `config_sparse.py`: `Config` class + real `argparse` (style of
    `vision/config_st.py`), accepting both `--c0_dataset` and `--c0-dataset`.
    Every `train.py` setting plus the sparse ones and `fire_at_boundary`.
  - Fixes the `exec`-based configurator's type trap (`--c0_data_replay_ratio=1.0`
    currently crashes because the default is an `int`).
  - Tests: defaults match `train.py`'s; both flag spellings parse; bools accept
    `False`/`false`/`0`; unknown flag errors.
  - Done 2026-09-16: `config_sparse.py`, `tests/test_config_sparse.py`.
- [x] **T2.2 Dense/FIRE/SNP training script** 🖥 — depends on T2.1
  - `train_sparse.py` with the same logic as `train.py`, supporting both the
    checkpoint-handoff mode and D1 (in-process boundary intervention). Written
    correctly from the start: rank-safe checkpoint
    writes, `wandb.finish()`, seeded `random`, unwrapped `state_dict` (no
    `_orig_mod.` prefix), run names that include every setting.
  - Tests: unit tests for `count_chunk_steps`, `get_lr`, run naming.
  - Done 2026-09-16: `train_sparse.py`. `vocab_size` is now a flag (train.py hardcodes
    50304), which is what makes the CPU tests fast. SNP works but has no e2e test yet
    (it needs an init checkpoint, and SNP is not one of the six arms).
- [x] **T2.3 Equivalence tests** 🖥 — depends on T2.2
  - Unit tests that the shared logic matches `train.py`: chunk step counts, LR
    schedule values, eval cadence, optimizer reset. Plus an e2e smoke run.
  - Done 2026-09-16: `tests/test_schedule.py` (chunk 0 = 97,282 iters and chunk 1 =
    36,765 iters as in the D2 table), `tests/test_train_e2e.py` (dense, FIRE,
    full_reset, checkpoint files). 79 tests pass in 43 s.
- [x] **T2.4 Sparsifier integration** 🖥 — depends on T2.2, T1.4
  - `sparse_utils.py` factory + metrics wired into `train_sparse.py`: built after
    the optimizer, before `torch.compile`/DDP; `step()` after `scaler.step()`.
  - Done 2026-09-16: `tests/test_sparse_e2e.py`. All four methods train both chunks
    and reach 0.9; GMP ramps up while the others start sparse; ITOP rises for
    SET/RigL and stays flat for Static. 145 tests pass in 70 s.
- [x] **T2.5 Offline W&B + DONE marker** 🖥
  - Offline `wandb.init` to `$SCRATCH`, `wandb.finish()`, `DONE` marker and run
    manifest on clean exit; sparse metrics logged at eval cadence, rank 0 only.
  - Test: offline dir created, `DONE` only on success, metrics present.
  - Done 2026-09-16: `--wandb_mode` (default offline) and `--wandb_dir` (default
    `$SCRATCH/fire_output/wandb`). A clean finish writes `DONE` in the run's output
    directory recording the offline run path; a crashed run writes none.
- [x] **T2.6 Login-node sync script** 🖥 — depends on T2.5
  - `bash_scripts/sync_wandb.sh`: sync each `DONE` run exactly once; `--dry-run`.
  - Test: with a stub `wandb`, the right runs are synced once each.
  - Done 2026-09-16: `bash_scripts/sync_wandb.sh`. Syncs runs with `DONE` and no
    `SYNCED`, so re-running is safe; `--dry-run` lists; failures are reported and
    leave the run unmarked for a retry.

## Phase 3 — Correctness on real hardware

- [x] **T3.1 `torch.compile` compatibility** 🎮
  - Done 2026-09-16 (GPU): all five configurations run compiled in fp16, and compiled
    matches eager. **triton was missing from the venv** (`torch.compile` fails with
    `TritonMissing` without it) — installed, and added to `setup_language.sh`.
    sparsimony's reparametrization needed no changes.
- [x] **T3.2 Masks identical across GPUs** 🎮 (tests only)
  - Done 2026-09-16 (2 GPUs): `tests/ddp_mask_check.py` gives each rank different
    weights and data, then compares masks. All four methods report MASKS_IDENTICAL,
    confirming sparsimony's broadcast keeps ranks on one subnetwork.
- [x] **T3.3 Sparse checkpoints** 🖥
  - Done 2026-09-16: `sparse_utils.flatten_sparse_state_dict` folds masks into the
    weights and stores them separately, so a sparse checkpoint loads into a plain
    GPT. `tests/test_checkpoints.py`.
- [x] **T3.4 One writer per checkpoint** 🎮
  - Done 2026-09-16 (2 ranks, 2 nodes): one run directory with init/best/chunk
    checkpoints and a single DONE marker, all written by rank 0 and loadable.
- [x] **T3.5 Seed the `wiki_owt` mixing** 🖥
  - Done 2026-09-16 in `train_sparse.py` (`random.seed(seed + rank)`).

## Running the tests

```bash
cd language
# CPU tests (~80 s). Use a compute node: on the login node torch starts one
# thread per core and the suite takes 10x longer.
srun --jobid=<cpu job> --overlap --cpus-per-task=4 python -m pytest -m "not gpu" -q

# GPU tests other than DDP: inside a GPU job step
srun --jobid=<gpu job> --overlap --ntasks=1 --gpus-per-task=1 --gpu-bind=none \
     python -m pytest -m gpu -q -k "not DDP"

# DDP tests: from OUTSIDE a job step (a step cannot create a step); they launch
# their own ranks with srun. SLURM_JOB_ID must point at an allocation.
SLURM_JOB_ID=<gpu job> python -m pytest -m gpu -q -k DDP
```

Multi-rank tests stage on `$SCRATCH`, not pytest's `/tmp`, which is node-local.

---

## Running multi-GPU on rorqual (learned 2026-09-16)

`torchrun` does not work here: SLURM binds one GPU per task, so one process
cannot see them all. Use srun, which `train_sparse.py` now supports
(`adopt_slurm_env` fills in RANK/WORLD_SIZE/LOCAL_RANK from SLURM).

```bash
# allocation: --mem is rejected for multi-task jobs, use --mem-per-cpu
salloc --account=def-yani --ntasks=2 --gpus-per-task=1 --cpus-per-task=8 \
       --mem-per-cpu=4G --time=2:00:00 --no-shell

# the step needs BOTH flags: without --gpus-per-task the ranks share one GPU,
# and without --gpu-bind=none NCCL fails with "invalid device ordinal"
srun --jobid=<id> --overlap --ntasks=2 --gpus-per-task=1 --gpu-bind=none ...
```

---

## Phase 4 — Profiling (go/no-go gate)

- [x] **T4.1 Single-GPU profile** 🎮
  - `profile_sparse.py`; raw JSON in `language/profiling/`.
  - Measured 2026-09-16 on one H100, GPT-2 124M, fp16, batch 12 x 1024, real data:

  | method | eager tok/s | MFU | compiled tok/s | MFU | peak mem (eager/compiled) |
  |---|---|---|---|---|---|
  | dense  | 337,963 | 29.2% | 433,859 | 37.5% | 13.4 / 9.7 GB |
  | static | 320,198 | 27.7% | 421,342 | 36.4% | 13.8 / 10.1 GB |
  | gmp    | 328,606 | 28.4% | 427,324 | 36.9% | 13.8 / 10.1 GB |
  | set    | 319,772 | 27.6% | 418,565 | 36.2% | 13.8 / 10.1 GB |
  | rigl   | 318,322 | 27.5% | **282,146** | 24.4% | 14.5 / **14.5 GB** |

  - Sparse costs only 3-6% per ordinary step. A mask update costs ~102 ms
    (about 3 steps) and happens every `delta_t`=214 steps at full scale, so
    under 0.1% overhead.
  - **`torch.compile` makes RigL 11% slower**, not faster, and costs 4.4 GB more
    memory than the others: its dense-gradient backward hook breaks the graph.
    Run RigL eager, everything else compiled.
  - Eval (20 batches): 3.7 s. Checkpoint: 1.49 GB, 5.3 s to write.
  - Data I/O is ~5% of step time once the file is in page cache, but the first
    pass is much slower: a cold-cache run measured 90k tok/s instead of 338k.
- [ ] **T4.2 Multi-GPU scaling** 🎮
  - Same measurements at 1, 2 and 4 GPUs.
  - **Request GPUs on one node** (`--nodes=1 --ntasks-per-node=N --gpus-per-task=1`).
    A first attempt landed 2 GPUs on two different nodes and dense throughput
    collapsed to 88k tok/s (vs 338k on one GPU), because every step's gradient
    all-reduce crossed the network.
  - Output: scaling efficiency per method.
- [ ] **T4.3 Cost model and job-layout decision** 🖥
  - Script: grid (6 methods × sparsities × 1 seed × tokens) + T4.1/T4.2 →
    GPU-hours, wall time per run, fit under 24 h, disk (checkpoints + W&B
    offline dirs), for **both** layouts: independent 1-GPU jobs and multi-GPU
    DDP jobs.
  - Tests: unit tests on the arithmetic.
  - Output: the approved budget and the D5 layout decision.
- [ ] **T4.4 Mid-run resume** 🎮 — only if T4.3 shows runs exceed 24 h
  - Restore model, masks, sparsifier, optimizer, GradScaler, counters, RNG.
  - Test: kill-and-resume loss matches an uninterrupted run.

## Phase 5 — Sweep

- [ ] **T5.1 Single-run launcher** 🎮
  - `bash_scripts/run_language_sparse.sh --sparsifier … --sparsity … --seed …`
    in the layout chosen at T4.3.
  - Validate: one short job finishes and syncs.
- [ ] **T5.2 Sweep generator** 🖥
  - Grid → job array; `--dry-run` prints the T4.3 budget.
  - Test: expected job count and unique run names.
- [ ] **T5.3 Pilot sweep** 🎮
  - All 6 methods, 1 seed, reduced scale that keeps the chunk-length ratio.
  - Validate: sanity-check curves in W&B before the full sweep.
- [ ] **T5.4 Full sweep** 🎮
  - 6 methods, 1 seed. Launch, sync, collect results into CSV and plots.

---

## Progress log

| Date | Task | Result / notes |
|---|---|---|
| 2026-09-15 | Plan | Created; D1, D3, D4 accepted; D2 confirmed; D5 revised to 1 seed |
| 2026-09-15 | T0.1 | Committed by user (`e115cb2`); `language/output` symlink committed, not ignored |
| 2026-09-15 | T0.5 | Logged in as `matharg`; default entity `ucalgary` |
| 2026-09-16 | T0.2, T0.3 | deepspeed+pytest installed; pytest scaffold; 9 tests pass |
| 2026-09-16 | D6 | New `train_sparse.py` runs all arms; `train.py` becomes the reference |
| 2026-09-16 | T0.4, T2.1, T2.2, T2.3 | `config_sparse.py` + `train_sparse.py` + harness; 79 tests pass in 43 s on a compute node |
| 2026-09-16 | T1.1–T1.5, T1.7, T2.4 | `sparse_utils.py`; all six arms train end to end; 145 tests pass in 70 s |
| 2026-09-16 | T2.5, T2.6 | Offline W&B + DONE markers + `sync_wandb.sh`; 159 tests pass in 77 s |
| 2026-09-16 | T1.6, T3.1, T3.2, T3.3, T3.5 | GPU pass: compile (needed triton), fp16+micro-steps, masks identical across 2 ranks, sparse checkpoints |
| 2026-09-16 | T3.4 | Fixed a DDP+sparse deadlock (mask buffers broadcast on every forward); 2-rank run completes and writes one set of checkpoints |
| 2026-09-16 | Phase 3 | Complete: 6/6 DDP tests and 12/12 other GPU tests pass |
| 2026-09-16 | T4.1 | Profiled all 5 arms; sparse costs 3-6%/step; compile helps all but RigL |
