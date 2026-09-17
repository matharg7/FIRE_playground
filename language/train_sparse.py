"""Continual pre-training of GPT-2 with plasticity interventions and sparse training.

Same logic as train.py, reorganised into importable functions so it can be unit
tested, plus sparse training via sparsimony. One script covers every arm:

    dense   : --method vanilla
    FIRE    : --method fire
    SNP     : --method snp  --snp_init_load_path ...
    sparse  : --sparsifier {static,gmp,set,rigl} --sparsity 0.9

Single GPU:  python train_sparse.py --method vanilla --compile False
Multi  GPU:  torchrun --standalone --nproc_per_node=4 train_sparse.py
"""
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from interventions.fire import fire
from interventions.snp import shrink_and_perturb
from model import GPT, GPTConfig

# Architecture fields that a checkpoint dictates: the weights only fit a model
# built with the same shape, so these override the command line on resume.
_CKPT_MODEL_ARGS = ('n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size')


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

@dataclass
class Dist:
    """Who am I in the process group? One process owns one GPU."""
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    master: bool = True
    device: str = 'cpu'


def adopt_slurm_env():
    """Let `srun --ntasks=N` work like torchrun.

    On DRAC each task is bound to its own GPU, so torchrun cannot see them all;
    multi-GPU jobs are launched with srun instead. This fills in the variables
    torch.distributed expects. Single-task steps are left alone.
    """
    if 'RANK' in os.environ or int(os.environ.get('SLURM_NTASKS', 1)) <= 1:
        return
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', '0')
    os.environ.setdefault('MASTER_PORT', '29500')
    if 'MASTER_ADDR' not in os.environ:
        nodelist = os.environ.get('SLURM_JOB_NODELIST', '')
        head = nodelist
        if nodelist and ('[' in nodelist or ',' in nodelist):
            import subprocess
            head = subprocess.run(['scontrol', 'show', 'hostnames', nodelist],
                                  capture_output=True, text=True,
                                  check=False).stdout.split()[0]
        os.environ['MASTER_ADDR'] = head or '127.0.0.1'


def setup_distributed(cfg):
    """Join the process group when launched by torchrun or srun, else run standalone.

    Also divides gradient accumulation across ranks so the global batch stays
    the same however many GPUs are used.
    """
    adopt_slurm_env()
    if int(os.environ.get('RANK', -1)) == -1:
        return Dist(enabled=False, device=cfg.device), cfg.gradient_accumulation_steps

    init_process_group(backend=cfg.backend)
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # Under srun each task usually sees only its own GPU, so every task's device
    # is cuda:0 even though their LOCAL_RANKs differ.
    visible = torch.cuda.device_count()
    if visible and local_rank >= visible:
        local_rank = local_rank % visible
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    if cfg.gradient_accumulation_steps % world_size != 0:
        raise ValueError(
            f"gradient_accumulation_steps ({cfg.gradient_accumulation_steps}) must divide "
            f"world size ({world_size})"
        )
    dist = Dist(True, rank, local_rank, world_size, rank == 0, device)
    return dist, cfg.gradient_accumulation_steps // world_size


# ---------------------------------------------------------------------------
# Chunk schedule
# ---------------------------------------------------------------------------

def chunk_configs(cfg):
    """The two chunks of training, played back to back."""
    return [
        {'dataset': cfg.c0_dataset, 'ratio': cfg.c0_subset_ratio,
         'replay': cfg.c0_data_replay_ratio},
        {'dataset': cfg.c1_dataset, 'ratio': cfg.c1_subset_ratio,
         'replay': cfg.c1_data_replay_ratio},
    ]


def count_chunk_steps(cfg, chunk, dataset_len, tokens_per_iter):
    """Turn (dataset size, subset ratio, replay ratio) into a training schedule.

    replay is the epoch-equivalent: the chunk trains on
    replay * ratio * dataset_len tokens, sampled with replacement.
    Returns None for an empty chunk (ratio or replay of 0), which is skipped.
    """
    num_tokens = int(chunk['replay'] * dataset_len * chunk['ratio'])
    if num_tokens <= 0:
        return None
    num_iters = num_tokens // tokens_per_iter
    return {
        'num_tokens': num_tokens,
        'num_iters': num_iters,
        'warmup_iters': min(int(cfg.warmup_ratio * num_iters), cfg.max_warmup_iters),
        'decay_iters': num_iters,
        # At least 20 evaluations per chunk. Guard the floor: train.py divides by
        # this, so a chunk shorter than 20 iterations would divide by zero.
        'eval_interval': max(1, min(cfg.eval_interval, num_iters // 20)),
    }


def get_lr(cfg, it, warmup_iters, decay_iters):
    """Linear warmup then cosine decay to min_lr. Restarts every chunk."""
    if not cfg.decay_lr:
        return cfg.learning_rate
    if it < warmup_iters:
        return cfg.learning_rate * (it + 1) / (warmup_iters + 1)
    if it > decay_iters:
        return cfg.min_lr
    decay_ratio = (it - warmup_iters) / max(1, decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def build_run_name(cfg):
    """A name that distinguishes every arm, so runs never share an output dir."""
    parts = [cfg.method]
    if cfg.method == 'fire':
        parts.append(f"it{cfg.fire_iteration}")
    elif cfg.method == 'snp':
        parts.append(f"coef{cfg.snp_shrink_coef}")
    if cfg.sparsifier != 'dense':
        parts.append(f"{cfg.sparsifier}_s{cfg.sparsity}")
        if cfg.sparsifier in ('set', 'rigl'):
            parts.append(f"pr{cfg.pruning_ratio}")
        if cfg.sparsifier in ('set', 'rigl', 'gmp'):
            parts.append(f"nmu{cfg.num_mask_updates}")
    else:
        parts.append("dense")
    parts.append(f"seed{cfg.seed}")
    parts.append(f"{cfg.c0_dataset}_{cfg.c0_subset_ratio}_{cfg.c0_data_replay_ratio}")
    parts.append(f"{cfg.c1_dataset}_{cfg.c1_subset_ratio}_{cfg.c1_data_replay_ratio}")
    if not cfg.reset_optimizer:
        parts.append("keep_optim")
    if cfg.comment:
        parts.append(cfg.comment)
    return "_".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class TokenData:
    """Reads the pre-tokenized .bin files.

    Each file is a flat uint16 array of token ids; np.memmap keeps it on disk and
    the OS pages in only the windows actually read. The memmap is recreated per
    batch to stop touched pages accumulating against the process (see train.py).
    """

    def __init__(self, cfg, device, device_type, data_root='data'):
        self.cfg = cfg
        self.device = device
        self.device_type = device_type
        self.data_root = data_root

    def _resolve(self, dataset_name):
        # wiki_owt is not a file: it mixes the two datasets in proportion to size.
        if dataset_name == 'wiki_owt':
            return 'wikitext' if random.random() < 0.1 / 9.1 else 'openwebtext'
        return dataset_name

    def _memmap(self, dataset_name, split):
        path = os.path.join(self.data_root, dataset_name, f'{split}.bin')
        return np.memmap(path, dtype=np.uint16, mode='r')

    def dataset_len(self, dataset_name):
        if dataset_name == 'wiki_owt':
            return sum(len(self._memmap(n, 'train')) for n in ('wikitext', 'openwebtext'))
        return len(self._memmap(dataset_name, 'train'))

    def get_batch(self, dataset_name, split, ratio=None):
        cfg = self.cfg
        data = self._memmap(self._resolve(dataset_name), split)
        # A subset ratio takes the front of the file, so chunk 1 at ratio 1.0
        # contains everything chunk 0 saw at a smaller ratio.
        end_index = int(len(data) * ratio) if ratio is not None else len(data)
        end_index -= cfg.block_size
        ix = torch.randint(end_index, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
        # y is x shifted one token left: every position predicts the next token.
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
        if self.device_type == 'cuda':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def model_args_from_cfg(cfg):
    return dict(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                block_size=cfg.block_size, bias=cfg.bias, vocab_size=cfg.vocab_size,
                dropout=cfg.dropout)


def load_model(cfg, ckpt_path, device):
    """Rebuild a model from a checkpoint, with the checkpoint's architecture."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = model_args_from_cfg(cfg)
    for k in _CKPT_MODEL_ARGS:
        model_args[k] = checkpoint['model_args'][k]
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint['model']
    # torch.compile wraps the module, so checkpoints saved from a compiled model
    # carry this prefix. We never save it (see save_checkpoint) but older ones have it.
    prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model, checkpoint, model_args


def build_model(cfg, device):
    """Either resume from a warm-start checkpoint or start from scratch."""
    if cfg.warm_start_load_path:
        print(f"Loading warm-start weights from {cfg.warm_start_load_path}")
        model, ckpt, model_args = load_model(cfg, cfg.warm_start_load_path, device)
        return model, ckpt, model_args
    print(f"Initializing a new model from scratch (vocab_size={cfg.vocab_size})")
    model_args = model_args_from_cfg(cfg)
    return GPT(GPTConfig(**model_args)), None, model_args


def spectrum_spread(model, max_layers=4):
    """Singular-value std/mean for a few weight matrices. 0 means isometric.

    FIRE aims to make weights isometric, but Newton-Schulz needs enough
    iterations to get there: on real GPT-2 weights, iteration=5 leaves the
    768x768 attention projections almost untouched. Logging this records what
    the intervention actually achieved instead of what it intended.
    """
    out = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not name.startswith('transformer.h'):
            continue
        weight = module.weight.detach().float()
        if weight.ndim != 2:
            continue
        svals = torch.linalg.svdvals(weight)
        svals = svals[svals > 1e-9]
        if svals.numel():
            out[name] = (svals.std() / svals.mean()).item()
        if len(out) >= max_layers:
            break
    return out


def apply_intervention(cfg, model, device):
    """Apply FIRE or shrink-and-perturb to the weights, in place."""
    metrics = {}
    if cfg.method == 'fire':
        print(f"Applying FIRE (iteration={cfg.fire_iteration})")
        before = spectrum_spread(model)
        fire(model, iteration=cfg.fire_iteration)
        after = spectrum_spread(model)
        for name in sorted(before):
            print(f"FIRE_SPECTRUM {name} before {before[name]:.4f} "
                  f"after {after.get(name, float('nan')):.4f}", flush=True)
            metrics[f"fire/{name}/spread_before"] = before[name]
            metrics[f"fire/{name}/spread_after"] = after.get(name, float('nan'))
    elif cfg.method == 'snp':
        print(f"Applying shrink-and-perturb (coef={cfg.snp_shrink_coef})")
        init_model, _, _ = load_model(cfg, cfg.snp_init_load_path, device)
        init_model.to(device)
        shrink_and_perturb(model, init_model, shrink_coef=cfg.snp_shrink_coef)
    return metrics


# ---------------------------------------------------------------------------
# Evaluation and checkpoints
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(cfg, model, data, ctx, dataset_name, train_ratio=None):
    out = {}
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            ratio = train_ratio if split == 'train' else None
            X, Y = data.get_batch(dataset_name, split, ratio=ratio)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path, raw_model, optimizer, model_args, cfg, **extra):
    """Write a checkpoint. Only the master process should call this.

    raw_model must be the uncompiled, unwrapped model so the keys stay clean.
    A sparse model is flattened back to plain GPT keys (masks folded into the
    weights, and stored separately), so any checkpoint loads into a plain GPT.
    """
    from sparse_utils import flatten_sparse_state_dict
    model_state, masks = flatten_sparse_state_dict(raw_model.state_dict())
    checkpoint = {
        'model': model_state,
        'masks': masks,
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'config': cfg.as_dict(),
        **extra,
    }
    torch.save(checkpoint, path)
    print(f"saved checkpoint {path}")


def resolve_dir(explicit, env_var, fallback):
    """Config value, else the environment, else a path relative to the cwd.

    bash_scripts/env.sh exports the FIRE_* variables per cluster, so no path is
    baked into the code and the same command works anywhere.
    """
    if explicit:
        return explicit
    return os.environ.get(env_var) or fallback


def resolve_wandb_dir(cfg):
    """Where W&B run directories go. They can get large, so prefer scratch."""
    return resolve_dir(cfg.wandb_dir or os.environ.get('WANDB_DIR', ''),
                       'FIRE_WANDB_DIR', 'wandb')


def write_done_marker(out_dir, run_name, wandb_run_dir, wandb_mode):
    """Mark a run as finished cleanly.

    bash_scripts/sync_wandb.sh looks for these: a run without DONE either failed
    or is still going, and must not be pushed to W&B.
    """
    path = os.path.join(out_dir, 'DONE')
    fields = {
        'run_name': run_name,
        'out_dir': os.path.abspath(out_dir),
        'wandb_run_dir': wandb_run_dir or '',
        'wandb_mode': wandb_mode,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(path, 'w') as f:
        for key, value in fields.items():
            f.write(f"{key}={value}\n")
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main(cfg):
    dist, grad_accum_steps = setup_distributed(cfg)
    device = dist.device
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'

    run_name = build_run_name(cfg)
    out_dir = os.path.join(resolve_dir(cfg.out_root, 'FIRE_OUTPUT_DIR', 'output'), run_name)
    if dist.master:
        os.makedirs(out_dir, exist_ok=True)

    # Each rank draws different batches; seeding random too makes the wiki_owt
    # mixture reproducible (train.py left it unseeded).
    torch.manual_seed(cfg.seed + dist.rank)
    random.seed(cfg.seed + dist.rank)
    np.random.seed(cfg.seed + dist.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
               'float16': torch.float16}[cfg.dtype]
    ctx = (nullcontext() if device_type == 'cpu'
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    tokens_per_iter = grad_accum_steps * dist.world_size * cfg.batch_size * cfg.block_size
    if dist.master:
        print(f"tokens per iteration: {tokens_per_iter:,}")
        cfg.print()

    data = TokenData(cfg, device, device_type,
                     data_root=resolve_dir(cfg.data_dir, 'FIRE_DATA_DIR', 'data'))
    chunks = chunk_configs(cfg)

    # ---- model ----------------------------------------------------------
    model, warm_ckpt, model_args = build_model(cfg, device)
    # train.py applies the intervention when loading the checkpoint; with
    # intervene_at_boundary it happens between the chunks instead (see plan D1).
    intervention_metrics = {}
    if cfg.warm_start_load_path and not cfg.intervene_at_boundary:
        intervention_metrics = apply_intervention(cfg, model, device)
    model.to(device)

    scaler = torch.amp.GradScaler(device_type, enabled=(cfg.dtype == 'float16'))
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate,
                                           (cfg.beta1, cfg.beta2), device_type)
    if cfg.warm_start_load_path and not cfg.reset_optimizer:
        print("Loading optimizer state from checkpoint")
        optimizer.load_state_dict(warm_ckpt['optimizer'])
    warm_ckpt = None

    # full_reset trains chunk 1 only; a warm start already covers chunk 0.
    skip_first_chunk = cfg.method == 'full_reset' or bool(cfg.warm_start_load_path)

    # ---- sparsifier -----------------------------------------------------
    # Must be built before compile/DDP: it reparametrizes the weight tensors.
    sparsifier = None
    itop_tracker = None
    if cfg.sparsifier != 'dense':
        from sparse_utils import ITOPTracker, build_sparsifier
        # The schedule spans every chunk that actually trains (plan D2). A
        # skipped chunk never calls sparsifier.step(), so counting it would put
        # t_end beyond the end of the run and the topology would never freeze.
        total_steps = 0
        for index, chunk in enumerate(chunks):
            if index == 0 and skip_first_chunk:
                continue
            sched = count_chunk_steps(cfg, chunk, data.dataset_len(chunk['dataset']),
                                      tokens_per_iter)
            if sched is not None:
                total_steps += sched['num_iters'] + 1
        sparsifier = build_sparsifier(cfg, model, optimizer, total_steps)
        itop_tracker = ITOPTracker(sparsifier)

    raw_model = model  # keep a handle to the unwrapped module for saving
    if cfg.compile:
        print("compiling the model...")
        model = torch.compile(model)
    # Evaluation runs on rank 0 only, so it must not touch the DDP wrapper:
    # sparsimony registers each mask as a module buffer, and DDP's default
    # broadcast_buffers=True would make every eval forward issue a collective
    # that the other ranks never match, deadlocking the run.
    eval_model = model
    if dist.enabled:
        model = DDP(model, device_ids=[dist.local_rank], broadcast_buffers=False)

    # ---- logging --------------------------------------------------------
    use_wandb = cfg.wandb_log and dist.master
    wandb_run_dir = ''
    if use_wandb:
        import wandb
        wandb_dir = resolve_wandb_dir(cfg)
        os.makedirs(wandb_dir, exist_ok=True)
        run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity or None,
                         name=run_name, config=cfg.as_dict(),
                         mode=cfg.wandb_mode, dir=wandb_dir)
        # run.dir is the run's files/ subdirectory; wandb sync wants its parent.
        wandb_run_dir = os.path.dirname(run.dir) if getattr(run, 'dir', None) else ''
        print(f"wandb {cfg.wandb_mode} run dir: {wandb_run_dir}")

    global_iter_num = 0
    global_logging_step = 0
    global_learned_token = 0
    running_mfu = -1.0
    # Checkpoints are deleted after a successful run, so W&B is the record:
    # keep the headline numbers in the run summary, not just the history.
    summary = dict(intervention_metrics)
    run_started = time.time()
    if cfg.save_checkpoint and dist.master:
        save_checkpoint(os.path.join(out_dir, 'init_ckpt.pt'), raw_model, optimizer,
                        model_args, cfg)

    for chunk_index, chunk in enumerate(chunks):
        dataset_name = chunk['dataset']
        dataset_len = data.dataset_len(dataset_name)
        sched = count_chunk_steps(cfg, chunk, dataset_len, tokens_per_iter)
        if sched is None:
            print(f"chunk {chunk_index}: empty, skipping")
            continue
        skip_this_chunk = chunk_index == 0 and skip_first_chunk
        print(f"chunk {chunk_index}: {dataset_name} ({dataset_len:,} tokens) -> "
              f"{sched['num_iters']:,} iters, warmup {sched['warmup_iters']:,}, "
              f"eval every {sched['eval_interval']:,}"
              + (" [SKIPPED]" if skip_this_chunk else ""))

        # The intervention lands between the chunks: chunk 0 has saturated the
        # model, chunk 1 is the new data.
        if chunk_index == 1 and cfg.intervene_at_boundary and cfg.method in ('fire', 'snp'):
            intervention_metrics = apply_intervention(cfg, raw_model, device)

        if cfg.reset_optimizer:
            optimizer.state.clear()

        if skip_this_chunk:
            # Keep the global counters aligned with a run that trained this chunk,
            # so curves from different arms line up on a shared x-axis.
            global_iter_num += sched['num_iters'] + 1
            global_learned_token += (sched['num_iters'] + 1) * tokens_per_iter
            continue

        X, Y = data.get_batch(dataset_name, 'train', ratio=chunk['ratio'])
        t0 = time.time()
        best_val_loss = float('inf')
        last_val_loss = last_train_loss = float('nan')
        pbar = tqdm(total=sched['num_iters'] + 1, disable=not dist.master)

        for local_iter_num in range(sched['num_iters'] + 1):
            lr = get_lr(cfg, local_iter_num, sched['warmup_iters'], sched['decay_iters'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if local_iter_num % sched['eval_interval'] == 0 and dist.master:
                losses = estimate_loss(cfg, eval_model, data, ctx, dataset_name,
                                       train_ratio=chunk['ratio'])
                last_val_loss, last_train_loss = losses['val'], losses['train']
                best_val_loss = min(best_val_loss, losses['val'])
                print(f"EVAL chunk {chunk_index} global_iter {global_iter_num} "
                      f"local_iter {local_iter_num} train {losses['train']:.6f} "
                      f"val {losses['val']:.6f}", flush=True)
                sparse_log = {}
                if sparsifier is not None:
                    from sparse_utils import sparse_metrics
                    sparse_log = sparse_metrics(raw_model, sparsifier, itop_tracker)
                    print("SPARSE " + " ".join(f"{k.split('/')[-1]} {v:.6f}"
                                               for k, v in sorted(sparse_log.items())),
                          flush=True)
                if use_wandb:
                    wandb.log({
                        'global_learned_token': global_learned_token,
                        'chunk_index': chunk_index,
                        'global_iter': global_iter_num,
                        'local_iter': local_iter_num,
                        'train/loss': losses['train'],
                        'val/loss': losses['val'],
                        'lr': lr,
                        'mfu': running_mfu * 100,
                        **sparse_log,
                    }, step=global_logging_step)
                global_logging_step += 1

                if cfg.save_checkpoint:
                    if (cfg.save_checkpoint_periodically
                            and local_iter_num % (sched['eval_interval'] * 5) == 0):
                        save_checkpoint(
                            os.path.join(out_dir, f'chunk{chunk_index}_iter{global_iter_num}_ckpt.pt'),
                            raw_model, optimizer, model_args, cfg,
                            chunk_index=chunk_index, global_iter=global_iter_num,
                            local_iter=local_iter_num, val_loss=losses['val'])
                    if losses['val'] <= best_val_loss:
                        save_checkpoint(
                            os.path.join(out_dir, f'best_chunk{chunk_index}_ckpt.pt'),
                            raw_model, optimizer, model_args, cfg,
                            chunk_index=chunk_index, global_iter=global_iter_num,
                            local_iter=local_iter_num, val_loss=losses['val'])

            # Gradient accumulation: several small forward/backward passes add
            # into .grad, then one optimizer step, so the effective batch is
            # grad_accum * world_size * batch_size sequences.
            for micro_step in range(grad_accum_steps):
                if dist.enabled:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                with ctx:
                    _, loss = model(X, Y)
                    loss = loss / grad_accum_steps
                X, Y = data.get_batch(dataset_name, 'train', ratio=chunk['ratio'])
                scaler.scale(loss).backward()

            if cfg.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if sparsifier is not None:
                # step() reports whether the mask topology changed this step.
                if sparsifier.step():
                    itop_tracker.update()

            t1 = time.time()
            dt, t0 = t1 - t0, t1
            if global_iter_num % cfg.log_interval == 0 and dist.master:
                lossf = loss.item() * grad_accum_steps
                if global_iter_num >= 5:
                    mfu = raw_model.estimate_mfu(cfg.batch_size * grad_accum_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                pbar.set_description(
                    f"chunk {chunk_index} iter {local_iter_num}/{sched['num_iters']}: "
                    f"loss {lossf:.4f}, {dt * 1000:.1f}ms, mfu {running_mfu * 100:.1f}%")

            global_iter_num += 1
            global_learned_token += tokens_per_iter
            pbar.update(1)

        pbar.close()
        summary[f'chunk{chunk_index}/best_val_loss'] = best_val_loss
        summary[f'chunk{chunk_index}/final_val_loss'] = last_val_loss
        summary[f'chunk{chunk_index}/final_train_loss'] = last_train_loss
        summary[f'chunk{chunk_index}/iters'] = sched['num_iters'] + 1
        if cfg.save_checkpoint and dist.master:
            save_checkpoint(os.path.join(out_dir, f'chunk{chunk_index}_ckpt.pt'),
                            raw_model, optimizer, model_args, cfg,
                            chunk_index=chunk_index, global_iter=global_iter_num)

    if use_wandb:
        import wandb
        summary['run/hours'] = (time.time() - run_started) / 3600
        summary['run/tokens'] = global_learned_token
        summary['run/iters'] = global_iter_num
        if sparsifier is not None:
            from sparse_utils import sparse_metrics
            summary.update(sparse_metrics(raw_model, sparsifier, itop_tracker))
        wandb.run.summary.update(summary)
        wandb.finish()
    if dist.master:
        write_done_marker(out_dir, run_name, wandb_run_dir, cfg.wandb_mode)
        if cfg.delete_checkpoints_on_success:
            # Only after DONE: a run that failed keeps its checkpoints so it can
            # be inspected or resumed. W&B holds everything needed for analysis.
            freed = 0
            for name in sorted(os.listdir(out_dir)):
                if name.endswith('.pt'):
                    path = os.path.join(out_dir, name)
                    freed += os.path.getsize(path)
                    os.remove(path)
            print(f"deleted checkpoints in {out_dir} ({freed / 1e9:.1f} GB freed)")
    if dist.enabled:
        destroy_process_group()
    return out_dir


if __name__ == '__main__':
    from config_sparse import get_config
    main(get_config())
