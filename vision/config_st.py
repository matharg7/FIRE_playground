class Config:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, dictionary):
        self.__dict__.update(dictionary)

    def __repr__(self):
        return f"Config({self.__dict__})"

    def print(self):
        print("-" * 30)
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
        print("-" * 30)


CONFIG = {
    # ---- Optimizer ----
    'optimizer': 'adam',         # adam
    'lr': 1e-3,
    'clip_grad_norm': 0.5,       # 0 to disable

    # ---- Task / Model ----
    'task':  'CIFAR10',          # CIFAR10 | CIFAR100 | TinyImageNet
    'model': 'RESNET18',         # RESNET18 | TinyViT | VGG16

    # ---- Benchmark ----
    'benchmark': 'continual',    # warm_start | continual | class_incremental
    'warm_start_subset_ratio': 10,

    # ---- Misc ----
    'log_every': 1,
    'seed': 0,
    'batch_size': 256, 
    'disable_wandb': False,  # True | False
    'wandb_project': '',          # override wandb project name (empty = auto)

    # ---- Sparsifier ----
    # Which algorithm to use.  'dense' runs without any sparsifier (baseline).
    'sparsifier': 'dense',       # dense | rigl | set | gmp | static

    # Shared sparse params
    'sparsity': 0.9,             # target sparsity for all sparse methods
    'num_mask_updates': 500,     # delta_t = total_steps // num_mask_updates
    't_end_ratio': 0.8,          # t_end  = t_end_ratio  * total_steps

    # RigL / SET
    'pruning_ratio': 0.3,        # fraction of nnz weights to prune per update

    # GMP only
    't_accel_ratio': 0.2,        # t_accel = t_accel_ratio * total_steps
    'initial_sparsity': 0.0,     # sparsity at step 0 (dense start)
}


import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_config():
    parser = argparse.ArgumentParser(description='Sparse Training Config')

    def add_arguments(parser, config, prefix=''):
        for key, value in config.items():
            arg_name = f"--{prefix}{key}".replace('_', '-')
            if isinstance(value, dict):
                add_arguments(parser, value, prefix=f"{prefix}{key}-")
            else:
                arg_type = type(value)
                if arg_type == bool:
                    parser.add_argument(arg_name, type=str2bool, default=value,
                                        help=f"Default: {value}")
                elif value is None:
                    parser.add_argument(arg_name, default=value,
                                        help=f"Default: {value}")
                else:
                    parser.add_argument(arg_name, type=arg_type, default=value,
                                        help=f"Default: {value}")

    from copy import deepcopy
    default_config = deepcopy(CONFIG)
    add_arguments(parser, default_config)

    args = parser.parse_args()

    def update_config(config, args, prefix=''):
        for key, value in config.items():
            arg_name = f"{prefix}{key}"
            if isinstance(value, dict):
                update_config(value, args, prefix=f"{prefix}{key}_")
            else:
                if hasattr(args, arg_name):
                    config[key] = getattr(args, arg_name)

    update_config(default_config, args)
    return Config(default_config)
