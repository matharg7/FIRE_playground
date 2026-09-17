"""Configuration for train_sparse.py.

Defaults match train.py. Every field becomes a CLI flag accepting both
spellings, so --c0_dataset and --c0-dataset both work.
"""
import argparse
from copy import deepcopy

CONFIG = {
    # ---- I/O ----
    'eval_interval': 2000,        # iterations between evaluations
    'log_interval': 1,
    'eval_iters': 200,            # batches averaged per loss estimate
    # Empty means "take it from the environment" (FIRE_DATA_DIR / FIRE_OUTPUT_DIR
    # / FIRE_WANDB_DIR, set by bash_scripts/env.sh), else a path next to the cwd.
    # Keeping these out of the code is what makes the setup cluster-agnostic.
    'data_dir': '',
    'save_checkpoint': True,
    'save_checkpoint_periodically': True,
    # Checkpoints exist so a crashed run is not a total loss. On success the
    # W&B run is the record, so delete them rather than keep ~19GB per run.
    'delete_checkpoints_on_success': False,
    'out_root': '',

    # ---- Data / batch ----
    'seed': 1337,
    'gradient_accumulation_steps': 8,
    'batch_size': 60,             # micro-batch, per GPU
    'block_size': 1024,

    # ---- Model ----
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
    'dropout': 0.0,
    'bias': False,
    'vocab_size': 50304,          # GPT-2's 50257 rounded up for efficiency

    # ---- Optimizer ----
    'learning_rate': 6e-4,
    'weight_decay': 1e-1,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,

    # ---- LR schedule (restarts every chunk) ----
    'decay_lr': True,
    'warmup_ratio': 0.1,
    'max_warmup_iters': 2000,
    'min_lr': 6e-5,

    # ---- System ----
    'backend': 'nccl',
    'device': 'cuda',
    'dtype': 'float16',
    'compile': True,

    # ---- Continual setup: two chunks played back to back ----
    'c0_dataset': 'wikitext',
    'c0_subset_ratio': 1.0,
    'c0_data_replay_ratio': 400,
    'c1_dataset': 'openwebtext',
    'c1_subset_ratio': 1.0,
    'c1_data_replay_ratio': 2,

    # ---- Intervention ----
    # vanilla     : no intervention
    # full_reset  : train chunk 1 only, from a fresh model
    # fire        : Frobenius-isometry reinitialization
    # snp         : shrink and perturb
    'method': 'vanilla',
    'fire_iteration': 5,
    'snp_shrink_coef': 0.8,
    'snp_init_load_path': '',
    # Where the intervention is applied:
    #   True  -> between chunk 0 and chunk 1, in this process (D1)
    #   False -> when loading warm_start_load_path (train.py's mode)
    'intervene_at_boundary': True,
    'warm_start_load_path': '',
    'reset_optimizer': True,

    # ---- Sparse training (sparsimony) ----
    'sparsifier': 'dense',        # dense | static | gmp | set | rigl
    'sparsity': 0.9,
    'num_mask_updates': 500,
    't_end_ratio': 0.8,
    'pruning_ratio': 0.3,         # rigl / set
    't_accel_ratio': 0.2,         # gmp
    'initial_sparsity': 0.0,      # gmp: sparsity before t_accel
    'accelerated_sparsity': 0.7,  # gmp: sparsity jumped to at t_accel
    # sparsimony's cubic ramp divides by t_end instead of (t_end - t_accel), so
    # GMP stops ~0.3pp short of the target. True uses the corrected span.
    'gmp_correct_cubic': True,

    # ---- Logging ----
    'wandb_log': True,
    # Compute nodes have no internet, so runs log offline and are pushed later
    # with bash_scripts/sync_wandb.sh from a login node.
    'wandb_mode': 'offline',      # offline | online | disabled
    'wandb_dir': '',              # empty = $SCRATCH/fire_output/wandb
    'wandb_project': 'warm_start_nanoGPT',
    'wandb_entity': '',           # empty = account default
    'comment': '',
}

METHODS = ('vanilla', 'full_reset', 'fire', 'snp')
SPARSIFIERS = ('dense', 'static', 'gmp', 'set', 'rigl')


class Config:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, dictionary):
        self.__dict__.update(dictionary)

    def as_dict(self):
        return dict(self.__dict__)

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
    parser = argparse.ArgumentParser(description="Continual pre-training with sparse training")
    for key, value in defaults.items():
        names = [f"--{key}"]
        dashed = f"--{key.replace('_', '-')}"
        if dashed != names[0]:
            names.append(dashed)
        kind = str2bool if isinstance(value, bool) else type(value)
        parser.add_argument(*names, dest=key, type=kind, default=value,
                            help=f"default: {value}")
    return parser


WANDB_MODES = ('offline', 'online', 'disabled')


def validate(cfg):
    if cfg.wandb_mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of {WANDB_MODES}, got {cfg.wandb_mode!r}")
    if cfg.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {cfg.method!r}")
    if cfg.sparsifier not in SPARSIFIERS:
        raise ValueError(f"sparsifier must be one of {SPARSIFIERS}, got {cfg.sparsifier!r}")
    if cfg.method == 'snp' and not cfg.snp_init_load_path:
        raise ValueError("method=snp needs --snp_init_load_path (the initial weights to shrink toward)")
    if not cfg.intervene_at_boundary and cfg.method in ('fire', 'snp') and not cfg.warm_start_load_path:
        raise ValueError(
            f"method={cfg.method} with intervene_at_boundary=False needs --warm_start_load_path"
        )
    return cfg


def get_config(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return validate(Config(deepcopy(vars(args))))
