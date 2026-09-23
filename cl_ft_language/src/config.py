"""Configuration for train.py.

Every field becomes a CLI flag accepting both spellings (--max_prompt_len and
--max-prompt-len). Empty paths are taken from the environment set up by
scripts/env.sh (TRACE_DATA_DIR, TRACE_OUTPUT_DIR, TRACE_WANDB_DIR).
"""

import argparse
from copy import deepcopy

from data import SUBSETS, TASKS

CONFIG = {
    # ---- Model / data ----
    'model': 'HuggingFaceTB/SmolLM2-135M',
    'data_dir': '',                 # empty = $TRACE_DATA_DIR
    'subset': 5000,                 # 5000 | 1000 | 500 train examples per task
    'tasks': ','.join(TASKS),       # comma-separated, in training order
    'max_train_per_task': 0,        # 0 = all; else a seeded random subset
    'max_eval_per_task': 0,         # 0 = all; else a seeded random subset of the eval split
    'eval_split': 'test',           # test | eval
    'max_prompt_len': 1024,
    'max_ans_len': 512,

    # ---- Optimisation ----
    # Epochs over the cumulative data of each task: one number for every task,
    # or a comma-separated list, one per task (TRACE's own: 5,3,7,5,3,5,5,7).
    'epochs_per_task': '1',
    'max_steps_per_task': 0,        # 0 = no cap; else cap optimizer steps per task
    # TRACE trains at an effective batch of 128 (per_device 2 x accum 8 x 8
    # GPUs). On one GPU the same effective batch is micro-batch x accum, and
    # only the effective number matters to the optimizer: 4 x 32 = 128.
    # Micro-batch 4 rather than 8 because fp32 at 8 peaked at 61.4GB and hit an
    # allocator OOM on Py150; at 4 the heaviest arm (fp32 + RigL) peaks at 38.2GB.
    'batch_size': 4,                # micro-batch (memory-bound)
    'gradient_accumulation_steps': 32,
    'learning_rate': 1e-5,          # TRACE's full fine-tuning LR
    # TRACE uses get_constant_schedule_with_warmup with --num_warmup_steps 0,
    # i.e. a flat 1e-5 for the whole run. 'cosine' is our own variant.
    'lr_schedule': 'constant',      # constant | cosine
    'min_lr_ratio': 0.1,            # cosine only: decays to learning_rate * min_lr_ratio
    'warmup_ratio': 0.0,            # of each task's steps; the schedule restarts per task
    # Fraction of optimizer steps allowed to be skipped for non-finite
    # gradients before the run is declared broken. DeepSpeed skips such steps
    # for TRACE; a handful is normal in bf16, a flood means it is not training.
    'max_skipped_ratio': 0.05,
    'weight_decay': 0.0,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,
    'reset_optimizer': True,        # clear AdamW state at every task boundary
    # Batch examples of similar length together: mixing FOMC-length and
    # MeetingBank-length examples left ~68% of every batch as padding.
    'length_grouped': True,

    # ---- Sparse training (sparsimony) ----
    # The pretrained weights are magnitude-pruned to `sparsity` at the start
    # (sparsimony's prepare()); RigL then prunes/regrows every delta_t steps
    # until t_end_ratio of the whole run (one schedule across all tasks).
    'sparsifier': 'dense',          # dense | rigl | set
    'sparsity': 0.1,
    # Which weights get a mask. Embeddings, the tied lm_head, the norms and
    # (by default) attention are never touched. Note that `sparsity` is
    # measured *within* this set, so the same value prunes less of the whole
    # model as the set narrows:
    #   all_linear  attention + MLP   72% of Qwen2.5-0.5B
    #   mlp         gate + up + down  64%   <- the experiment's default
    #   up_down     up + down only    42%  (gate_proj stays dense too)
    #   gate        gate_proj alone   21%
    'sparse_targets': 'mlp',        # all_linear | mlp | up_down | gate
    'sparse_distribution': 'erk',   # erk | uniform (per-layer split of the sparsity)
    'num_mask_updates': 600,        # topology updates over the whole run
    # Startup check: delta_t is derived from the whole run, but the
    # drop-fraction schedule is per-task, so a coarse delta_t can leave a short
    # task with almost no topology updates. Fail loudly instead of silently
    # training a near-static mask. 0 disables the check.
    'min_updates_per_task': 10,
    't_end_ratio': 0.8,             # stop updating the topology after this fraction of steps
    'pruning_ratio': 0.3,           # fraction of active weights swapped per update
    # How the drop fraction (pruning_ratio) is scheduled, for rigl and set:
    #   global   - one cosine decay over t_end_ratio of the whole run. The
    #              topology then freezes partway through the last task.
    #   per_task - the cosine restarts at every task boundary and decays over
    #              t_end_ratio of that task's steps, so every task, including
    #              the last, gets topology updates. delta_t (the update cadence)
    #              stays global, so longer tasks get more updates.
    #   constant - held at pruning_ratio until t_end (global horizon)
    'drop_fraction_schedule': 'global',   # global | per_task | constant
    # What a regrown weight starts from:
    #   zero     - 0 (sparsimony's default)
    #   previous - the value it had when it was last pruned, or its pretrained
    #              value if it was never active
    'grow_init': 'zero',            # zero | previous

    # ---- System ----
    # bfloat16 overflows in the backward through Qwen2.5's tied embedding
    # (vocab 151,936 over hidden 896, 27% of the model), skipping 17% of steps
    # on MeetingBank and 42% on Py150. float32 skips none at ~5-15% less
    # throughput. TRACE used bf16 on Llama-2-7B, whose embedding is ~2%.
    'dtype': 'float32',             # autocast dtype; master weights stay fp32
    'gradient_checkpointing': False,
    'compile': False,
    'num_workers': 2,
    'seed': 0,

    # ---- Evaluation ----
    'eval_batch_size': 16,
    # Score every task with the untrained model before task 0: the reference
    # point for forward transfer, and impossible to reconstruct later.
    'eval_zero_shot': True,
    # Teacher-forced loss per task alongside the generation score: one forward
    # pass per batch, a few seconds per task, and smoother than exact match.
    'eval_loss': True,
    # Also score the *next* task at each boundary, i.e. R[t][t+1]. Forward
    # transfer needs it (each task scored just before it is trained), at the
    # cost of one extra task evaluation per boundary (~20 min per Qwen run).
    'eval_lookahead': True,

    # ---- Output / logging ----
    'out_root': '',                 # empty = $TRACE_OUTPUT_DIR
    'run_name': '',                 # empty = derived from the config
    'save_checkpoint': 'final',     # final | every_task | none
    'log_interval': 10,             # optimizer steps between log lines
    'wandb_mode': 'offline',        # offline | online | disabled
    'wandb_dir': '',                # empty = $TRACE_WANDB_DIR
    'wandb_project': 'dst_trace_benchmark',
    'wandb_entity': '',             # empty = account default
    'comment': '',
}

SPARSIFIERS = ('dense', 'rigl', 'set')
# Mirrors sparse_utils.TARGET_SETS, spelled out here so config.py stays free of
# torch and sparsimony imports; tests/test_sparse_utils.py checks they agree.
SPARSE_TARGETS = ('all_linear', 'mlp', 'up_down', 'gate')
LR_SCHEDULES = ('constant', 'cosine')
DROP_FRACTION_SCHEDULES = ('global', 'per_task', 'constant')
GROW_INITS = ('zero', 'previous')
WANDB_MODES = ('offline', 'online', 'disabled')
SAVE_MODES = ('final', 'every_task', 'none')


class Config:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def as_dict(self):
        return dict(self.__dict__)

    def task_list(self):
        return [t.strip() for t in self.tasks.split(',') if t.strip()]

    def epochs_list(self):
        """Epochs per task, one entry per task in task_list()."""
        values = [float(e) for e in str(self.epochs_per_task).split(',') if e.strip()]
        tasks = self.task_list()
        if len(values) == 1:
            return values * len(tasks)
        if len(values) != len(tasks):
            raise ValueError(f"epochs_per_task has {len(values)} values for {len(tasks)} tasks")
        return values

    def __repr__(self):
        return f"Config({self.__dict__})"

    def print(self):
        print("-" * 40)
        for key, value in sorted(self.__dict__.items()):
            print(f"{key}: {value}")
        print("-" * 40)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def build_parser(defaults=None):
    defaults = CONFIG if defaults is None else defaults
    parser = argparse.ArgumentParser(description="Continual fine-tuning on TRACE")
    for key, value in defaults.items():
        names = [f"--{key}"]
        dashed = f"--{key.replace('_', '-')}"
        if dashed != names[0]:
            names.append(dashed)
        kind = str2bool if isinstance(value, bool) else type(value)
        parser.add_argument(*names, dest=key, type=kind, default=value,
                            help=f"default: {value}")
    return parser


def validate(cfg):
    for field, allowed in (('sparsifier', SPARSIFIERS), ('wandb_mode', WANDB_MODES),
                           ('save_checkpoint', SAVE_MODES), ('eval_split', ('test', 'eval')),
                           ('lr_schedule', LR_SCHEDULES)):
        if getattr(cfg, field) not in allowed:
            raise ValueError(f"{field} must be one of {allowed}, got {getattr(cfg, field)!r}")
    if cfg.subset not in SUBSETS:
        raise ValueError(f"subset must be one of {tuple(SUBSETS)}, got {cfg.subset}")
    unknown = [t for t in cfg.task_list() if t not in TASKS]
    if unknown or not cfg.task_list():
        raise ValueError(f"unknown or empty tasks {unknown}; choose from {TASKS}")
    if any(e <= 0 for e in cfg.epochs_list()):
        raise ValueError(f"epochs_per_task must be positive, got {cfg.epochs_per_task!r}")
    if cfg.dtype not in ('bfloat16', 'float16', 'float32'):
        raise ValueError(f"dtype must be bfloat16, float16 or float32, got {cfg.dtype!r}")
    if cfg.sparse_distribution not in ('erk', 'uniform'):
        raise ValueError(f"sparse_distribution must be erk or uniform, got {cfg.sparse_distribution!r}")
    if cfg.sparse_targets not in SPARSE_TARGETS:
        raise ValueError(f"sparse_targets must be one of {SPARSE_TARGETS}, got {cfg.sparse_targets!r}")
    if cfg.drop_fraction_schedule not in DROP_FRACTION_SCHEDULES:
        raise ValueError(f"drop_fraction_schedule must be one of {DROP_FRACTION_SCHEDULES}, "
                         f"got {cfg.drop_fraction_schedule!r}")
    if cfg.grow_init not in GROW_INITS:
        raise ValueError(f"grow_init must be one of {GROW_INITS}, got {cfg.grow_init!r}")
    if cfg.grow_init != 'zero' and cfg.sparsifier not in ('rigl', 'set'):
        raise ValueError(f"grow_init={cfg.grow_init!r} only applies to rigl and set, which regrow")
    if cfg.sparsifier != 'dense' and not 0 < cfg.sparsity < 1:
        raise ValueError(f"sparsity must be in (0, 1), got {cfg.sparsity}")
    if cfg.sparsifier != 'dense' and cfg.compile:
        # RigL's dense-gradient backward hooks make compiled training slower
        # (measured in the GPT-2 sweep), so keep it eager.
        raise ValueError("compile=True is not supported with a sparsifier; RigL runs eager")
    return cfg


def get_config(argv=None, **overrides):
    args = build_parser().parse_args(argv)
    d = deepcopy(vars(args))
    d.update(overrides)
    return validate(Config(d))
