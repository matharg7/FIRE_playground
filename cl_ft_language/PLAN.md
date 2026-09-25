# Sparse training × continual learning on TRACE

## Status / resume here

Last updated 2026-09-21.

- **Done:**
  - Planning.
  - Step 0: `CLAUDE.md` no longer forbids Claude from running commands; it still asks before destructive actions, commits, pushes and new branches.
  - Model choice for profiling.
- **T0.1 done (2026-09-21).**
  - Venv at `/scratch/matharg7/fire/venv-trace`.
  - Versions: torch 2.14.0, transformers 4.57.6 (the newest 4.x, which satisfies sparsimony's `<5` pin), datasets 5.0.0, deepspeed 0.18.1.
  - sparsimony is installed editable from `vision/sparsimony`, which needs hatchling from the wheelhouse.
  - `scripts/build_env.sh check` prints READY on the login node.
- **T0.2 and T0.3 done (2026-09-21):**
  - `scripts/build_env.sh download` fetched the TRACE archive with gdown, plus both models.
  - `srun ... build_env.sh check` on an H100 prints READY; both models load in bf16 with sdpa and generate.
- **Phase 0 is complete.**
- **T1.1 done (2026-09-21):** `tests/test_trace_compat.py`, 36 passing on the login node in about 30 s. The changes to `trace/` are listed under T1.1 below.
- **T1.2 done (2026-09-21):** `src/data.py`, with `tests/test_data.py` (13 passing).
- **T1.3 done (2026-09-21):** `src/evaluate.py`, with `tests/test_evaluate.py` (15 passing on an H100; 3 of them are GPU tests).
- **T1.4 done (2026-09-21):** `src/cl_metrics.py`, with `tests/test_cl_metrics.py` (8 passing). **Phase 1 is complete.**
- **T2.1 and T2.2 done (2026-09-21):** `src/train.py` and `src/config.py`, with `tests/test_train.py` (12 passing; 2 of them GPU tests).
  - The 2-task smoke run on SmolLM2 learns the answer format (FOMC reaches 0.75 on 8 test examples after 32 steps) and writes all outputs.
- **Phase 2 is complete** (commit `85fc428`).
- **T3.1 done:** `src/sparse_utils.py`, with `tests/test_sparse_utils.py` (8 passing).
- **T3.2–T3.4 done:** RigL is wired into `src/train.py` and `src/config.py`, with `tests/test_rigl.py` (19 GPU tests passing, including the trainer and eval GPU tests). **Phase 3 is complete.**
- Phase 3 committed as `b669ce5`.
- **T4.1 and T4.2 done (2026-09-22).** Decisions, from the cost model below:

  | Decision | Choice |
  |---|---|
  | Model | **Qwen2.5-0.5B** (multilingual, so C-STANCE and 20Minuten are meaningful) |
  | Epochs | **TRACE's own** `5,3,7,5,3,5,5,7`, over the cumulative data |
  | Prompt cap | **1024** (Qwen OOMs at 2048 with batch 8; TRACE's inference scripts also used 1024) |
  | Eval | **500 test examples per task**, with zero-shot scores, eval loss and look-ahead (FWT) |
  | Batching | **Length-grouped** (new; see below) |
  | Batch size | 8, no gradient accumulation needed at this size |

  **Cost per run (H100):** about 3.5 h training and 1.2 h evaluation, so roughly 4.7 h. That exceeds the 3 h partition limit, so runs need the 12 h partition or T4.3's resume.

- **Logging added (2026-09-22), at the user's request:**
  - `--eval_zero_shot` (default on): every task scored with the untrained model before task 0. Saved as `baselines` in `scores.json` and `eval/zero_shot_<i>_<task>.json`, logged as `eval_zero_shot/<task>`. It costs about 15 min for Qwen with 500 examples per task, and cannot be reconstructed afterwards.
  - `--eval_loss` (default on): teacher-forced loss on the answer tokens per task, logged as `eval_loss/<task>` and stored per step in `scores.json`. One forward pass per batch, a few seconds per task, and it moves smoothly where exact match is jumpy.
  - `--eval_lookahead` (default **on**, the user's choice for the sweep): also scores task t+1 at each boundary, filling R[t][t+1]. Forward transfer (FWT) needs it, since each task must be scored just before it is trained. It costs about 20 min per Qwen run. Turning it off still leaves OP and BWT; only FWT is omitted.
    - Worth having: in the smoke run the model scored 0.75 on FOMC after training only on C-STANCE, never having seen FOMC. Both are single-letter classification tasks, so the answer format transfers.
  - `ScoreMatrix.summary()` now returns FWT when it is computable, and `can_fwt()` reports whether it is.
  - Declined: git commit / Slurm ids in the config, padding fraction per window, per-layer sparsity and mask churn.

- **Length-grouped batching** (`--length_grouped`, on by default), added after profiling showed 68% of every batch was padding:
  - `data.LengthGroupedSampler`: per epoch, shuffle, cut into megabatches of 50 batches, sort each megabatch by length, then shuffle the batch order. Every example appears once per epoch; batches still mix tasks, just tasks of similar length.
  - Measured on Qwen: padding 68% → 25%, throughput 37.7 → 58.7 examples/s dense and 39.2 → 63.4 with RigL (about 1.6×).
  - Note: with grouping, the first few batches no longer contain every task; that holds over an epoch instead.

- **T4.1 details:**
  - `src/profile_trace.py` has three parts (`data`, `train`, `eval`); `scripts/profile.sbatch` runs all of them for both models.
  - Job 21551889 (3 h limit) sat pending for about 16 h, with an estimated start a day later, so it was cancelled on 2026-09-22.
  - Resubmitted as two 1 h jobs, one per model: 21600288 (SmolLM2) and 21600289 (Qwen), with `MODELS=<model> sbatch --time=01:00:00 ...`. Short jobs fill gaps in the queue much sooner.
  - Results go to `$TRACE_OUTPUT_DIR/profile/*.json`; the logs are `logs/trace-profile-<model>-<jobid>.out`.
  - **SmolLM2-135M results** (job 21600288, 5.6 min, H100). Qwen is still pending.
    - **Tokens:**
      - One epoch per task of full cumulative training is 62.5M real tokens at a 1024 prompt cap (4.25× the sequential total) and 80.5M at a 2048 cap.
      - MeetingBank has a p95 prompt of 14.9k tokens; 55% of its prompts are truncated at 1024 and 38% at 2048. Py150 truncates 20% at 1024, 20Minuten 32% at 1024 (1.5% at 2048).
      - 3.3% of ScienceQA answers exceed 512 tokens.
    - **Training at batch 8:**
      - dense: about 23–24k real tokens/s;
      - RigL at 10%: 21–24k, the same within noise;
      - a mask update costs 0.18 s and prepare about 1–2 s, so the earlier 124 s was a one-off, not RigL.
      - **65–71% of each batch is padding**, so length-grouped batching could give 2–3×.
      - Peak memory: 21 GB at cap 1024, 35–38 GB at cap 2048 (the logits dominate).
    - **One run** (1 epoch per task, cap 1024): about 45 min of training. TRACE's epochs (5,3,7,5,3,5,5,7) would take about 4 h.
    - **Eval, worst case** (the untrained model runs to the token cap; batch 16), per full test set:
      - ScienceQA 33 min, MeetingBank 8.6, Py150 5.9, C-STANCE 2.5, 20Minuten 1.8, the rest < 0.3.
      - The full triangle of evaluations is about 4 h, dominated by ScienceQA (4 × 33 min).
    - Log noise: "n_grow_per_tile == 0" warnings from sparsimony. At 10% sparsity, ERK leaves many layers dense, so those layers have nothing to regrow.
  - **Qwen2.5-0.5B results** (job 21601572, 5.75 min, H100):
    - 1 epoch per task of cumulative training is 53.9M tokens at a 1024 cap; truncation is 54% for MeetingBank, 19% for Py150, 12% for 20Minuten.
    - Training: dense 14.0k real tokens/s, RigL 14.5k, both at 68% padding before length grouping. Peak memory 47.5 GB dense and 51.8 GB with RigL. **Both arms OOM at a 2048 cap with batch 8.**
    - Eval, untrained: 39 min for all eight full test sets, 3.0 h for the full triangle of 36. Qwen stops generating sensibly (ScienceQA averages 124 tokens against the 512 cap, where SmolLM2 always hit the cap).
  - **Cost model**, prompt cap 1024, before length grouping:

    | | SmolLM2-135M | Qwen2.5-0.5B |
    |---|---|---|
    | tokens/s | 23.5k | 14.2k |
    | 1 epoch/task | 0.74 h | 1.05 h |
    | 2 epochs/task | 1.5 h | 2.1 h |
    | TRACE epochs | 3.9 h | 5.6 h |
    | 36 evals, full test sets | 4.0 h | 3.0 h |
    | 36 evals, 500/task | 1.5 h | 1.2 h |
  - Queue note: 3 h jobs waited more than a day; 30–60 min jobs started within about 2 h. Keep profiling and test jobs short.
  - To check: in the RigL smoke run, the first task's 16 steps took 124 s against 5 s for the second task's 32 steps. That looks like a one-time start-up cost; the profile times mask updates separately to confirm it.
  - Fixed: `src/data.py` now keeps `src/` ahead of `trace/` on `sys.path`, because `trace/train.py` was shadowing `src/train.py`. Tested.
  - Smoke-profile eval number for SmolLM2 (untrained, so answers run to the token cap): ScienceQA's full test set would take about 130 min per evaluation, and MeetingBank about 35 min. Eval subsampling and/or faster generation are needed.

## First launch failed: NaN at step 80 (2026-09-23)

The first sweep (4 runs, nibi) went to NaN in task 0 and never recovered --
1994 of 2001 logged steps were NaN. Cancelled after ~3 GPU-hours. Two
independent causes, neither of them the sparse code: the dense arm, which never
builds a sparsifier, failed identically at the same step.

**Cause 1: the effective batch was 16x too small.** TRACE trains at 128
(`per_device_train_batch_size 2` x `gradient_accumulation_steps 8` x 8 GPUs,
from `trace/scripts/train_seq_cl.sh`). Our `sweep.conf` had batch 8 with
`gradient_accumulation_steps=1`, i.e. 8. T4.2's "batch 8, no gradient
accumulation needed at this size" was reasoning about *memory* and silently
changed the optimisation; `learning_rate=1e-5` is TRACE's value for batch 128.
Measured gradient norms were 28-257 against `grad_clip=1.0` on every step.

**Cause 2: bf16 overflows on Qwen2.5-0.5B, and nothing guarded the step.**
A GPU bisect (`src/debug_nan.py`, 120 steps, identical batches) showed:

| dtype | optimizer | result |
|---|---|---|
| bfloat16 | fused AdamW | NaN at step 73 |
| bfloat16 | unfused AdamW | NaN at step 73, identical |
| float32 | fused AdamW | clean through 120 steps |

The forward is healthy at the failing step (loss 0.527, logits absmax 31.6);
the inf is born in the **backward**, first in `model.embed_tokens.weight`.
`clip_grad_norm_` then scales every parameter by `max_norm/nan`, turning all
290 tensors into NaN in one step -- permanent, unrecoverable, from one bad
micro-batch.

Qwen2.5-0.5B has vocab 151,936 over hidden 896, so the **tied** embedding is
27.5% of the model and takes gradient from both the lookup and `lm_head`.
TRACE ran Llama-2-7b-chat (vocab 32k / hidden 4096), where it is ~2%. Their
bf16 stability never transferred to this shape.

Skip rates at the corrected effective batch of 128, with the guard counting
rather than aborting:

| task | bf16 skipped | fp32 skipped |
|---|---|---|
| C-STANCE | 5/60 (8.3%) | 0 |
| MeetingBank | 2/12 (17%) | 0 |
| Py150 | 5/12 (42%) | 0 |

bf16 is unusable here: the skips correlate with long sequences, so it would
systematically drop the hardest examples. **fp32 is a deliberate deviation from
TRACE**, forced by the model choice, and costs only 5-15% throughput.

**Why not DeepSpeed, as TRACE uses.** sparsimony's `DSTMixin` rejects any
optimizer by exact type (`type(optimizer) not in {SGD, AdamW, Adam}`), and RigL
depends on `optimizer.state[original_param]["exp_avg"] *= mask` via
`register_step_post_hook` -- ZeRO flattens parameters into buckets, so that
lookup would not resolve and `test_adamw_moments_are_zero_for_masked_weights`
would fail. On one GPU ZeRO-2 also buys nothing. Separately it would not have
fixed bf16: we already keep fp32 master weights and fp32 grads, and DeepSpeed
casts the model *to* bf16 where autocast keeps softmax/layernorm/loss in fp32.
The one thing it gave TRACE -- skipping non-finite steps -- is now in
`train_task` directly.

**Fixes.**
- `train.py`: skip the optimizer step when the grad norm is not finite, count
  it, log `train/skipped_steps`, and abort above `--max_skipped_ratio` (5%)
  rather than degrading quietly for five hours.
- `lr_schedule` config: `constant` (TRACE's
  `get_constant_schedule_with_warmup` with 0 warmup steps) is now the default;
  `cosine` is our own variant.
- Effective batch 128 as **4 x 32**, not 8 x 16: fp32 at micro-batch 8 peaked
  at 61.4GB and hit an allocator OOM on Py150. At 4 the heaviest arm
  (fp32 + RigL 0.3) peaks at 38.2GB. The optimizer update is identical.
- `dtype` default `float32`.
- `num_mask_updates` 1000 -> 60. This follows from the batch change: the run is
  **7,340** optimizer steps at effective batch 128, not 117,500, so 60 updates
  give `delta_t=97` -- RigL's published ~100. At 1000 it would be every 5 steps.

**Measured after the fix** (H100, fp32, 4 x 32, `sparse_targets=mlp`):

| task | arm | tok/s | peak mem | skipped |
|---|---|---|---|---|
| MeetingBank | dense | 29,088 | 34.4 GB | 0 |
| MeetingBank | rigl 0.3 | 27,790 | 38.2 GB | 0 |
| Py150 | dense | 22,359 | 34.4 GB | 0 |
| Py150 | rigl 0.3 | 20,906 | 38.2 GB | 0 |

Estimated run: 2.8-3.8 h training + ~1.8 h eval = 4.6-5.6 h against a 06:50
walltime. The eval figure is from bf16 profiling and generation is now fp32
too, so it may be ~30% higher; `scores.json` is written after every task, so an
overrun costs the tail rather than the run.

**Lesson.** The GPU tests all run SmolLM2-135M; Qwen2.5-0.5B had only ever been
profiled for throughput, never trained. The reference implementation should be
diffed against before the first launch, not after it fails.

## Second launch stopped: delta_t starved the early tasks (2026-09-23)

Noticed as a rising `dst/pruning_ratio` across task boundaries
(0.094 -> 0.141 -> 0.284 -> ... -> 0.298). The scheduler was fine; the cadence
was not.

`delta_t` is derived from the **whole run**
(`t_end_ratio * total_steps // num_mask_updates`), but `per_task` restarts the
cosine with `t_end = t_end_ratio * that task's steps`. After the effective-batch
fix the run is 7,340 steps and C-STANCE is only **195**, so `delta_t=97` was 62%
of its entire update window:

| task | steps | t_end | updates at delta_t=97 | logged pr |
|---|---|---|---|---|
| C-STANCE | 195 | 156 | **1** | 0.094 |
| FOMC | 234 | 187 | **1** | 0.141 |
| MeetingBank | 820 | 656 | 6 | 0.284 |
| 20Minuten | 2187 | 1749 | 18 | 0.298 |

With one sample point per short task, a short `t_end` lands late on the cosine
(low value) and a long `t_end` lands early (high value) -- hence the rise. The
real damage is that RigL barely ran on the first two tasks, which is where BWT
measures forgetting.

**Cause.** When the effective batch went to 128, `num_mask_updates` was cut
1000 -> 60 to keep `delta_t ~= 97`, "matching RigL's published delta_t=100".
That was reasoning in steps while the step had changed size: at batch 128 one
step is 128 examples, so `delta_t=97` is 12,416 examples between updates, 16x
less frequent than the same setting at batch 8. **delta_t has to be read in
data, and it has to fit the shortest task, not the whole run.**

**Fix.** `num_mask_updates=600` -> `delta_t=9` (1,152 examples between updates,
close to the original 752): C-STANCE 17 updates, FOMC 20, ... 20Minuten 194,
649 over the run, ~2 min of mask cost. Every task's cosine now starts at
~0.2975, i.e. the full `pruning_ratio`, as intended.

**Guardrail.** `sparse_utils.check_update_cadence()` runs at startup, prints
the per-task update counts, and raises if any task gets fewer than
`--min_updates_per_task` (default 10), naming the `num_mask_updates` that would
fix it. Regression tests in `tests/test_rigl.py` pin both the starved case and
the shipped default. This class of mismatch is now a startup error rather than
a curve to notice three tasks in.

## Over-training on a 3-task subset (branch `dst_cl_subset_overtrain`, 2026-09-25)

**Pre-registered before any run.**

**Hypothesis.** Under heavy over-training with cumulative replay, RigL s0.3
generalises better than dense fine-tuning **at matched training loss**, because
reduced capacity limits memorisation.

**Why this and not the full benchmark.** The 8-task sweep found dense and
rigl s0.3 indistinguishable at matched settings, but dOP (sparse - dense) rose
monotonically as the model was pushed harder:

| lr | dOP | dense OP | s0.3 OP | dense train loss | s0.3 train loss |
|---|---|---|---|---|---|
| 1e-5 | -0.0013 | 0.5572 | 0.5559 | 0.0256 | 0.0612 |
| 3e-5 | -0.0002 | 0.5264 | 0.5262 | 0.0094 | 0.0176 |
| 1e-4 | **+0.0050** | 0.4281 | 0.4331 | 0.0250 | 0.0261 |

At 1e-4 both arms reach the **same** training loss and sparse still generalises
slightly better, which rules out the crude confound ("sparse merely fits
less"). +0.0050 is an eighth of the +-0.04 noise band on one seed, so this is a
hypothesis, not a result.

**Design.** `FOMC -> ScienceQA -> NumGLUE-cm`, cumulative replay kept (it is
the setting being targeted, not a variable), epochs `10,10,20` against TRACE's
own `3,3,5` for these tasks. 3,514 optimizer steps. Arms: dense and rigl s0.3.
Seed 0 only; seeds are a separate later experiment.

**Task choice is on measurement quality and cost, decided before looking at
any outcome:** all three have a zero-shot baseline of exactly 0.000, so FWT is
clean (C-STANCE ranges 0.000-0.170 across arms, MeetingBank 16x, 20Minuten
starts at 0.363); all are short-prompt; they span classification, reasoning and
arithmetic. Deliberately **not** the three tasks where sparse happened to win
at 1e-4 -- selecting those would not replicate. An exhaustive search over
arm x subset earlier found a best cherry-pick of +0.0248 on 3 tasks where the
honest 8-task number was -0.0039; that is what ~1,500 combinations of noise
produce.

**Read-out.** dOP at matched `train_loss_final` (now recorded per task in
`summary.json`), not absolute OP. **OP is expected to FALL for both arms** --
over-training took dense from 0.557 to 0.428 at lr=1e-4. A lower OP is the
manipulation working, not a regression.

**Decision rule.** If dOP at matched training loss is within +-0.01, record a
null and stop pursuing over-training. If sparse leads by more, the next step is
seeds (3) and an over-training dose-response (`5,5,10` / `10,10,20` /
`20,20,40`), since only monotone trends have survived in this project.

**Deliberately NOT controlled: weight decay.** `weight_decay=0` is
load-bearing, because AdamW's decoupled decay shrinks masked weights (which get
no gradient), so `grow_init=previous` would regrow a shrunken value -- there is
a test for this. A "dense + weight decay" arm would therefore give dense a
regularizer that sparse structurally cannot take. Matching on training loss
answers the same question without that asymmetry.

**Cost.** 1.76 h per run on one H100, from the calibration
`s/step = 1.165 + 0.00284 * mean_padded_width` (reproduces 1.485 s/step at
width 112.6 and 2.555 at 489.5). Scale ~1.7x on an A100 (3.0 h), ~3.5x on a
V100 (6.2 h). Peak memory ~20-25 GB, well below the 8-task sweep's 38 GB,
because MeetingBank (mean padded width 1142) is not in the subset.
W&B project: `dst_trace_overtrain`, kept separate from `dst_trace_benchmark`.

## Migration of the `dst-fire-full-reset` branch (2026-09-22)

Verified against that branch's code, then migrated. See README §5 for how to use it.

**What the branch does, and whether it is right:**
- **`grow_init="previous"` — correct.** Pruning only clears the mask; sparsimony never clears the stored weight, so skipping the zero-fill makes a regrown weight resume the value it had when pruned. A weight that was never active comes back at its **pretrained** value, which matters here in a way it does not in the vision experiments (those train from scratch).
  - **Caveat found while testing:** AdamW's decoupled weight decay shrinks *every* parameter each step, including masked ones with no gradient. With `weight_decay > 0`, "previous" therefore regrows a decayed value. Our default is 0, and `build_sparsifier` now warns otherwise. There is a test for the decay effect.
- **`per_task` drop-fraction schedule — correct**, but the vision code sets `t_end` to the **full** task length, not 80% of it. A mask update can then land in a task's final steps, leaving regrown weights untrained — and TRACE evaluates immediately after each task. Our version uses `t_end_ratio` of the task's steps (0.8 by default); `--t_end_ratio 1.0` reproduces vision exactly.
- **"Trains on the final stage" — correct, and the point of it.** With the global schedule, `t_end` is 80% of the whole run, which falls inside the last task, so the final task mostly trains a frozen mask.
- **`delta_t` stays global** in vision, so tasks get different numbers of updates. Under cumulative replay the tasks grow, so this skews towards later tasks. Kept as in vision, and noted in the README.
- Not migrated: FIRE and full_reset at task boundaries. The first comparison is dense vs RigL only (the user's decision).

**Changes:** the same `grow_mask` patch in the vendored `vision/sparsimony` (backwards compatible, default `zero`); `--grow_init`, `--drop_fraction_schedule` and SET support in `src/config.py` and `src/sparse_utils.py`; `restart_drop_fraction_schedule` called at each boundary in `src/train.py`; `tests/test_grow_init_and_schedule.py` (13 tests).

## Moving to another cluster (decided 2026-09-22)

**Why.** One sweep run costs about 5.3 h on an H100: 3.5 h training (TRACE's epochs over cumulative data, with length grouping), 1.2 h evaluation, 0.25 h zero-shot and 0.35 h look-ahead. On rorqual that does not fit:
- the partition we have been using caps jobs at 3 h, and the 12 h partition means far longer waits;
- observed queue behaviour: a 3 h job waited more than a day and never ran, while 30–60 min jobs started within about 2 h;
- the first sweep is 4 runs (dense, plus RigL at 5/10/20%), so about 21 GPU-hours, and more with extra seeds.

A cluster with longer job limits lets a run finish in one job, and avoids building resume support (T4.3) purely to work around the queue.

**Migration checklist:**
1. Push the `trace` branch and clone it on the new cluster (needs the user's OK to push).
2. `cl_ft_language/scripts/build_env.sh env download check` there. Scratch is not shared between clusters, so the venv, TRACE data and model weights are all rebuilt. Allow for the Google Drive download and the two HF models.
3. Check the cluster profile in `scripts/env.sh`: it keys off `CC_CLUSTER` and pins `cuda/13.2` only on rorqual; elsewhere it uses the site default. Add a case if the new cluster needs specific modules.
4. Check the account name and whether GPU jobs get the same automatic `_gpu` suffix; set `TRACE_ACCOUNT` in `scripts/local.sh` (gitignored).
5. Re-run `pytest tests/` (CPU) and `pytest tests/ -m gpu` in a short GPU job to confirm the environment.
6. Re-check throughput with `profile_trace.py --part train` if the GPU differs from an H100; the cost model above assumes H100 numbers.
7. W&B stays offline if the compute nodes have no internet; sync from a login node.

**Open:** T4.3 (resume from a task boundary) becomes optional if the new cluster allows jobs of 6 h or more. It is still worth having as insurance against a run being cut short.

## Goal
See whether sparse training (RigL first) helps continual fine-tuning of a small pretrained LLM on the TRACE benchmark (8 sequential tasks), compared with dense training.

## Decisions

| Decision | Choice |
|---|---|
| Models (profiling) | `HuggingFaceTB/SmolLM2-135M` (fast, English-only) and `Qwen/Qwen2.5-0.5B` (multilingual). Profiling decides which one the sweep uses |
| Hardware | 1 GPU per run |
| Replay | Full cumulative: task t trains on the union of training data from tasks 0..t |
| First comparison | **Dense vs RigL only.** No FIRE, no Static/GMP/SET, no no-replay baseline for now |
| Sparsity | 5%, 10%, 20% (higher later) |
| Mask schedule | One global schedule across all tasks (per-task restart is future work) |
| Trainer | New plain-PyTorch trainer (torch AdamW, bf16). No DeepSpeed: sparsimony rejects FusedAdam (`vision/sparsimony/sparsimony/dst/base.py:30-35`) |
| Testing | GPU tests directly; no CPU end-to-end or CPU profiling scripts |
| Tasks | Follow TRACE closely; may drop tasks after profiling |

## Folder rule
- `cl_ft_language/` is self-contained and never imports from `language/`. Copying files in and adapting them is fine.
- The training script, config and cluster scripts are independent of `language/` and `bash_scripts/`.
- Layout:
  - `trace/`: upstream TRACE, with light compatibility fixes only.
  - `src/`: `config.py`, `data.py`, `evaluate.py`, `cl_metrics.py`, `sparse_utils.py`, `train.py`, `metrics_compat.py`.
  - `tests/`
  - `scripts/`: `env.sh`, `build_env.sh`, `run_trace.sh`, `sweep.conf`, `sweep.sh`, `sync_wandb.sh`.
- **Venv:** a new one on scratch, at `$SCRATCH/fire/venv-trace`, following the `language/` pattern.
  - It shares `$SCRATCH/fire`'s data, output, W&B and HF cache dirs.
  - TRACE data goes under `$FIRE_DATA_DIR/trace/`.
  - sparsimony is installed with `pip install -e vision/sparsimony`.
- When copying `language/sparse_utils.py`, keep these fixes:
  - the corrected GMP cubic scheduler;
  - `sparsifier.step()` once per optimizer step, after `optimizer.step()`;
  - the sparsifier is built after the optimizer and before compile, and RigL runs eager (compile makes it slower);
  - the optimizer state is cleared, not the optimizer rebuilt, at task boundaries (this keeps sparsimony's momentum hook);
  - masks are folded when checkpointing.

## What we reuse
- **From TRACE (`trace/`):**
  - `utils/data/raw_datasets.py` `LocalJsonFileDataset`
  - `utils/data/data_utils.py` `create_prompt_dataset`
  - `utils/data/data_collator.py`: pads on the left, masks the prompt to -100, truncates from the left
  - `evaluations/eval_*.py` and `metrics.py`
  - generation logic in `inference/infer_single.py`
- **From `language/` (copied):**
  - `sparse_utils.py`: `build_sparsifier`, `sparsifier_schedule`, DST metrics, ITOP tracker, `flatten_sparse_state_dict`.
  - `train_sparse.py` as the reference for step and build order, the `DONE` marker and W&B offline.
  - `bash_scripts/{env.sh, clusters/drac.sh, build.sh, sync_wandb.sh}` and `language/tests/conftest.py` as templates.

## Facts about the TRACE code (from exploration)
- **Data:** not in the repo. Download it from the Google Drive link in `trace/README.md`. Each task is a folder of `train.json`, `eval.json` and `test.json`, each holding a list of `{"prompt","answer"}` records.
- **Upstream setup:**
  - LLaMA-2-7B, DeepSpeed ZeRO-2/3, bf16, LR 1e-5.
  - Epochs per task `5,3,7,5,3,5,5,7`.
  - `max_prompt_len` 1024 (2048 in the naive script), `max_ans_len` 512.
- **Upstream replay (`training/replay.py`):** a separate phase after each task. It takes the first 10% of each past task, in file order, plus Lima. We do NOT use it; we use full cumulative data instead.
- **Metrics:** OP and BWT are not computed anywhere upstream, so we compute them ourselves.
- **Library versions:** upstream pins `transformers==4.31.0`, and the code relies on these old APIs:
  - `transformers.deepspeed.HfDeepSpeedConfig`
  - `datasets.load_metric` (used for SARI)
  - a forced flash-attn monkey-patch (`training/main.py:44-48`)

## TRACE tasks (default order)

| # | Task | What it is | Language | Metric (paper) | Notes |
|---|---|---|---|---|---|
| 1 | C-STANCE | Stance detection | Chinese | accuracy | Short |
| 2 | FOMC | Fed statement hawkish/dovish/neutral classification | English | accuracy | Short |
| 3 | MeetingBank | Meeting transcript summarisation | English | ROUGE-L | **Long prompts**, costly to generate |
| 4 | Py150 | Python code completion | Code | edit similarity | Medium |
| 5 | ScienceQA | Multiple-choice science QA with reasoning | English | accuracy | |
| 6 | NumGLUE-cm | Commonsense arithmetic | English | accuracy | Short |
| 7 | NumGLUE-ds | Domain-specific arithmetic | English | accuracy | Short |
| 8 | 20Minuten | News text simplification | German | SARI | Long-ish |

The paper uses 5,000 training samples per task; test sets range from a few hundred to about 2k examples.

## Task list

### Phase 0 — Environment
- [x] **T0.1** `scripts/env.sh` and `scripts/build_env.sh env` build the new scratch venv. Per-user settings go in `scripts/local.sh` (gitignored), which sets `TRACE_ACCOUNT=def-yani`.
  - Modules: StdEnv/2023, python/3.11, cuda, arrow.
  - Wheelhouse packages: `torch`, `transformers<5` (sparsimony pin), `datasets`, `deepspeed` (imported by sparsimony), `sentencepiece`, `nltk`, `rouge`, `fuzzywuzzy`, `sacrebleu`, `sacremoses`, `wandb`, `pytest`, `pandas`.
  - Not installed: `flash_attn` (we use sdpa attention).
- [x] **T0.2** `build_env.sh check`: versions, CUDA, the sparsimony import, and both models loading on GPU. Run it on a GPU with:

  ```
  srun --account=def-yani --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 --mem-per-cpu=8G --time=00:15:00 ./cl_ft_language/scripts/build_env.sh check
  ```
- [x] **T0.3** `build_env.sh download`, on a login node: the TRACE data into `$TRACE_DATA_DIR` and both models into `$HF_HOME`.
  - No NLTK data is needed: TRACE uses only `sentence_bleu`.
  - Moved to T1.1: a local SARI implementation (`load_metric` is gone).

### Data layout and sizes (from T0.3)
- Root: `/scratch/matharg7/fire/data/trace/TRACE-Benchmark/`.
- Subsets:
  - `LLM-CL-Benchmark_5000`: the paper setup, 5000 training examples per task;
  - `LLM-CL-Benchmark_1000` and `LLM-CL-Benchmark_500`: 1000 and 500 per task;
  - `LLM-CL-Benchmark_Reasoning`: a reasoning variant.

  Each task folder also has `Lima`. The `__MACOSX/` folder is junk from the zip.

Sizes for the 5000 subset (character counts; tokens come later, in T4.1):

| Task | eval / test | Train prompt chars (median / max) | Train answer chars (median / max) |
|---|---|---|---|
| C-STANCE | 2000 / 2000 | 153 / 269 | 1 / 1 |
| FOMC | 496 / 496 | 312 / 1385 | 1 / 1 |
| MeetingBank | 687 / 692 | **5647 / 373826** | 338 / 1100 |
| Py150 | 2000 / 2000 | 663 / 162477 | 32 / 256 |
| ScienceQA | 2000 / 2000 | 275 / 1195 | 805 / 3690 |
| NumGLUE-cm | 41 / 81 | 193 / 474 | 2 / 18 |
| NumGLUE-ds | 164 / 325 | 138 / 282 | 2 / 5 |
| 20Minuten | 200 / 200 | **2220 / 10418** | 261 / 910 |

MeetingBank and 20Minuten dominate the cost. Truncation from the left caps the extreme prompts.

### Phase 1 — Data + eval
- [x] **T1.1** Compatibility fixes in `trace/`, tested by `tests/test_trace_compat.py`:
  - `sari.py` (new): a vendored copy of the HF `datasets` SARI metric. It matches the metric card's reference value, 26.9536…
  - `metrics.py`: uses it in place of `load_metric`. `caculate_sari` now returns the dict, not the 1-tuple upstream produced through a stray comma.
  - `inference/prompts.py` (new): holds `TASK_PROMT` and `Constrained_PROMPT`. Both `ICL.py` and `data_collator.py` import them from here, so the collator no longer pulls in the whole DeepSpeed/CL-method stack.
  - `utils/data/data_collator.py`:
    - Skips BOS when the tokenizer has none. Qwen2.5 has none, and upstream crashed on it.
    - **Upstream bug fix:** training examples are now built as `[BOS] + prompt (left-truncated) + answer (capped at max_ans_len, start kept) + EOS` from separately tokenized parts. Upstream tokenized the pair jointly. For truncated examples (the long MeetingBank and Py150 prompts), that dropped EOS and shifted the label mask one token into the prompt.
  - `utils/data/data_utils.py`: `torch.load(..., weights_only=False)` for the pickled dataset caches, which torch ≥ 2.6 refuses by default.
  - `utils/model/model_utils.py`: imports `HfDeepSpeedConfig` from `transformers.integrations`.
  - Left alone: the flash-attn patch in `training/main.py`. We don't use that file, and we use sdpa attention.
  - Tokenizer notes:
    - SmolLM2 has no pad token; Qwen2.5 has no BOS token.
    - Both default to right padding, so our loader must set left padding and truncation, pad = eos, and should not invent a BOS for Qwen.
    - TRACE's `load_hf_tokenizer` works but sets bos = eos for Qwen.
  - Environment note: always load the full `TRACE_MODULES` (`trace_load_modules`), because `packaging` comes from `scipy-stack`.
  - Run the tests with: `source scripts/env.sh && trace_load_modules && trace_activate && python -m pytest -q tests/`.
- [x] **T1.2** `src/data.py`:
  - `load_tokenizer`: left padding and truncation; pad = eos when missing (SmolLM2); Qwen's missing BOS is not invented.
  - `load_task` / `load_tasks`: TRACE's `create_dataset`, with no `.pt` caching. The `subset` argument picks 5000, 1000 or 500 examples per task. `max_train`, `max_eval` and `max_test` draw seeded random subsets, whereas TRACE took the first N.
  - `cumulative_train(task_data, t)`: a `ConcatDataset` of the train splits of tasks 0..t.
  - `TaskCollator`: TRACE's collator plus a `task_ids` tensor, for per-task loss.
  - `train_loader`: seeded shuffle, so batches mix all tasks seen so far.
  - `eval_loader`: keeps dataset order, as TRACE's metrics assume.
  - `src/` puts `trace/` on `sys.path` itself.
- [x] **T1.3** `src/evaluate.py`:
  - `score_task` returns a [0, 1] scalar per task, following the paper:
    - accuracy for C-STANCE, FOMC, NumGLUE and ScienceQA (answer letter);
    - ROUGE-L for MeetingBank;
    - similarity/100 for Py150;
    - SARI/100 for 20Minuten.
  - `generate` matches TRACE's decoding (new tokens only, special tokens skipped, no strip, so exact match is strict). It uses greedy decoding instead of sampling at temperature 0.1.
  - `MAX_NEW_TOKENS` caps generation per task: 8 for the classification tasks, up to 512 for ScienceQA.
  - `evaluate_task` / `evaluate_tasks(upto=t)` also return the predictions and the seconds taken. Eval subsampling is `data.load_task(max_test=N)`.
  - Upstream fix in `trace/evaluations/eval_ScienceQA.py`: an empty generation crashed it (`pred[0]`).
  - Notes:
    - SARI scores an empty output highly on short sources, because it rewards deletions. That is a property of the metric.
    - In bf16, greedy output can change with batch size (rounding with different padding). In fp32, batched and unbatched are identical, as tested.
  - GPU tests: `srun --account=def-yani --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 --mem-per-cpu=8G --time=00:20:00 bash -c 'source cl_ft_language/scripts/env.sh && trace_load_modules && trace_activate && cd cl_ft_language && python -m pytest -q tests/'`
- [x] **T1.4** `src/cl_metrics.py`: `ScoreMatrix`, a T×T matrix where NaN means not evaluated.
  - `record(t, {task: score})`, then `op(t)` and `bwt(t)` (TRACE paper definitions), plus `fwt(baselines)`. FWT needs a zero-shot baseline and an evaluation of task i just before it is trained.
  - A missing score raises an error rather than being silently averaged.
  - Saves to and loads from JSON.

### Phase 2 — Dense CL trainer (GPU)
- [x] **T2.1** `src/train.py` and `src/config.py`.
  - Setup: fp32 master weights with bf16 autocast, fused torch AdamW (no weight decay on 1-D parameters), gradient clipping at 1.0, sdpa attention.
  - The loss is computed by hand (it equals HF's, as tested), so we also get per-task token losses (`train_loss_by_task/i`).
  - Position ids start at 0 on the first real token of left-padded rows, matching generation.
  - Per task t:
    1. clear the optimizer state (`--reset_optimizer`, default true), keeping the optimizer object;
    2. train on the cumulative data 0..t with warmup plus cosine, restarting per task;
    3. evaluate tasks 0..t on `--eval_split` (default test) with `--max_eval_per_task`;
    4. write `scores.json`, `eval/step<t>_<i>_<task>.json` and the W&B log (`cl/op`, `cl/bwt`, `eval/<task>`, `time/*`, `perf/*`).
  - At the end: a checkpoint (`--save_checkpoint final|every_task|none`), `summary.json` (OP, BWT, R, timings), then `DONE`.
  - Defaults (confirmed with the user): LR 1e-5, 1 epoch per task, optimizer reset at each boundary, W&B project `trace_sparse_cl`.
  - Phase 3 hooks are marked in the code: build the sparsifier after the optimizer, and call `sparsifier.step()` after `optimizer.step()`.
- [x] **T2.2** GPU smoke test in `tests/test_train.py`: 2 tasks, SmolLM2. The loss falls within each task, R has the right shape, OP/BWT match R, and the outputs plus DONE exist.

### Phase 3 — RigL integration
- [x] **T3.1** `src/sparse_utils.py`, adapted from `language/`, RigL only:
  - Targets every `nn.Linear` under `model.layers.` (q/k/v/o and gate/up/down). Embeddings, the tied `lm_head` and the norms stay dense.
  - Coverage:
    - SmolLM2-135M: 210 tensors, 106,168,320 of 134,515,008 weights (78.9%);
    - Qwen2.5-0.5B: 168 tensors, 357,826,560 of 494,032,768 weights (72.4%).

    Both counts match the numbers computed from each model's config.
  - `flatten_sparse_state_dict` also drops RigL's dense-gradient buffers.
  - Initial mask: sparsimony's `prepare()` magnitude-prunes the current (pretrained) weights, because `random_mask_init` defaults to False. So T3.2 needs no extra pruning step.
  - The distribution is ERK by default (as in the GPT-2 sweep); `uniform` is also available.
- [x] **T3.2** New config fields:
  - `--sparsifier rigl --sparsity s` (default 0.1);
  - `--sparse_distribution erk|uniform`;
  - `--num_mask_updates` (100), `--t_end_ratio` (0.8), `--pruning_ratio` (0.3, cosine-decayed).

  `prepare()` magnitude-prunes the pretrained weights. Tested at s ∈ {0.05, 0.1, 0.2} with both distributions: mask sparsity is exact; with uniform, every pruned weight is smaller than every kept weight in its layer; embeddings and norms are untouched. RigL combined with `compile` is rejected, so it runs eager.
- [x] **T3.3** Global schedule: `train.total_run_steps` sums the steps over all cumulative tasks before training starts, and `t_end = 0.8·total`, `delta_t = t_end // num_mask_updates`. `sparsifier.step()` runs after each `optimizer.step()`; ITOP is updated on topology changes. `dst/*` metrics are logged, and the sparsity stats at each boundary go into `summary.json` timings.
- [x] **T3.4** Invariants in the 2-task RigL run:
  - the schedule covers both tasks (48 steps, `t_end` 38, `delta_t` 4);
  - mask sparsity is 0.1 ± 1e-3 at every boundary;
  - ITOP > 0.9, so the topology really changes;
  - `exp_avg` and `exp_avg_sq` are 0 for every masked weight (210 tensors);
  - the loss falls;
  - the checkpoint loads as a plain HF model, is zero where the masks are zero, and gives the same logits as the sparse model.

  Evaluation runs inside `parametrize.cached()`, so mask × weight is computed once per evaluation, not per generated token.

### Phase 4 — Profiling on GPU (both models)
- [ ] **T4.1** In one GPU job, for dense and RigL:
  - per-task token-length statistics and the cumulative tokens per task;
  - tokens/s and peak memory at `max_len` 1024/1536, with and without gradient checkpointing;
  - generation seconds per test example for each task.
- [x] **T4.2** Cost model and decisions: see the status section at the top. No tasks are dropped.
- [ ] **T4.3** Resume from a task boundary. Optional once we move to a cluster with longer job limits (see the top of this file), but still useful insurance. Save the model, masks, sparsifier step count, optimizer state, the score matrix and the RNG state after each task; `--resume` picks up at the next task.

### Phase 5 — First sweep
- [x] **T5.1** `scripts/run_trace.sh`, `sweep.conf`, `sweep.sh` and `sync_wandb.sh`: a Slurm array with 1 GPU per run, W&B offline, the account taken from `TRACE_ACCOUNT`. `./sweep.sh` lists the runs, `--submit` queues them. No paths are hardcoded anywhere outside the gitignored `scripts/local.sh`. `README.md` documents all of it.
  - `--epochs_per_task` now also takes one value per task, for TRACE's `5,3,7,5,3,5,5,7`.
  - Default sweep: dense plus RigL at 5/10/20%, with `grow_init=previous` and `drop_fraction_schedule=per_task`, on Qwen2.5-0.5B.
- [ ] **T5.2** Runs: dense cumulative, and RigL cumulative × s ∈ {5%, 10%, 20%}.
- [ ] **T5.3** Analysis: OP, BWT, per-task curves, ITOP and mask drift across tasks.

## Verification
- `pytest cl_ft_language/tests` runs in the new venv. Tests that need a GPU run on `salloc`.
- Before the sweep: 2-task, few-step dense and RigL runs each produce an R-matrix, OP/BWT and exact sparsity in the logs.
