"""Measure what each arm costs, on the real GPT-2 (124M), with real data.

    python profile_sparse.py --sparsifier rigl --compile True --steps 30

Writes one JSON file per configuration into profiling/ so the cost model (T4.3)
can read them. Reports:
  * tokens/sec and MFU against the H100's real peak (model.py hardcodes the
    A100's 312 TFLOPS, which overstates MFU on an H100 by ~3.2x)
  * peak memory
  * ordinary step time vs mask-update step time
  * evaluation and checkpoint cost
"""
import argparse
import json
import os
import time

import torch

from config_sparse import CONFIG, Config
from model import GPT, GPTConfig
from train_sparse import TokenData, adopt_slurm_env, estimate_loss, save_checkpoint

H100_BF16_TFLOPS = 989.0  # dense, no sparsity, per NVIDIA's H100 SXM datasheet


def build_cfg(args):
    cfg = Config(dict(CONFIG))
    cfg.update({
        'sparsifier': args.sparsifier, 'sparsity': args.sparsity,
        'compile': args.compile, 'dtype': args.dtype,
        'batch_size': args.batch_size, 'block_size': args.block_size,
        'gradient_accumulation_steps': args.grad_accum,
        'eval_iters': args.eval_iters, 'num_mask_updates': args.num_mask_updates,
        'device': 'cuda',
    })
    return cfg


def profile(args):
    cfg = build_cfg(args)
    device_type = 'cuda'

    # Multi-rank when launched by srun --ntasks=N (see plan: torchrun cannot see
    # all GPUs on rorqual). Throughput is then reported for the whole job.
    adopt_slurm_env()
    ddp = int(os.environ.get('RANK', -1)) != -1
    rank, world_size, local_rank = 0, 1, 0
    if ddp:
        import torch.distributed as dist
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK']) % max(1, torch.cuda.device_count())
        torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    torch.manual_seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
               'float16': torch.float16}[cfg.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model = GPT(GPTConfig(n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd,
                          block_size=cfg.block_size, bias=cfg.bias,
                          vocab_size=cfg.vocab_size, dropout=cfg.dropout)).to(device)
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate,
                                           (cfg.beta1, cfg.beta2), device_type)
    scaler = torch.amp.GradScaler(device_type, enabled=(cfg.dtype == 'float16'))

    sparsifier = None
    if cfg.sparsifier != 'dense':
        from sparse_utils import build_sparsifier
        # A short profile still needs a realistic schedule, so derive delta_t
        # from the full run length rather than the profiling run.
        sparsifier = build_sparsifier(cfg, model, optimizer, args.total_steps)

    raw_model = model
    compile_seconds = 0.0
    if cfg.compile:
        t0 = time.time()
        model = torch.compile(model)
        compile_seconds = time.time() - t0
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # broadcast_buffers=False: sparsimony keeps masks (which are buffers) in
        # sync itself, and broadcasting them every forward deadlocks eval.
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    data = TokenData(cfg, device, device_type, data_root=args.data_root)
    tokens_per_step = (cfg.gradient_accumulation_steps * cfg.batch_size
                       * cfg.block_size * world_size)

    # --reuse-batch answers "is get_batch on the critical path?": CUDA is async
    # and the batch is fetched after the forward is launched, so its CPU cost may
    # already be hidden behind GPU compute.
    cached_batch = data.get_batch(args.dataset, 'train', ratio=args.ratio) if args.reuse_batch else None

    def fetch():
        if cached_batch is not None:
            return cached_batch
        return data.get_batch(args.dataset, 'train', ratio=args.ratio)

    def one_step():
        """One optimizer step. Returns (total_s, mask_update_s, data_s, did_update)."""
        t_data = 0.0
        t_d0 = time.time()
        X, Y = fetch()
        t_data += time.time() - t_d0
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(cfg.gradient_accumulation_steps):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / cfg.gradient_accumulation_steps
            t_d0 = time.time()
            X, Y = fetch()
            t_data += time.time() - t_d0
            scaler.scale(loss).backward()
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t_train = time.time()
        did_update = False
        if sparsifier is not None:
            did_update = bool(sparsifier.step())
            torch.cuda.synchronize()
        t_end = time.time()
        return t_end - t0, t_end - t_train, t_data, did_update

    for _ in range(args.warmup):
        one_step()
    torch.cuda.reset_peak_memory_stats()

    plain, updates, data_times = [], [], []
    for _ in range(args.steps):
        total_s, mask_s, data_s, did_update = one_step()
        (updates if did_update else plain).append((total_s, mask_s))
        data_times.append(data_s)

    step_times = [t for t, _ in plain] or [t for t, _ in updates]
    step_s = sum(step_times) / len(step_times)
    data_s = sum(data_times) / len(data_times)
    compute_s = max(1e-9, step_s - data_s)
    tokens_per_s = tokens_per_step / step_s
    # 6ND for forward+backward, plus attention; matches model.estimate_mfu's formula.
    n_params = raw_model.get_num_params()
    flops_per_token = 6 * n_params + 12 * cfg.n_layer * cfg.n_head * (
        cfg.n_embd // cfg.n_head) * cfg.block_size
    mfu = flops_per_token * tokens_per_s / (H100_BF16_TFLOPS * 1e12)

    torch.cuda.synchronize()
    t0 = time.time()
    estimate_loss(cfg, model, data, ctx, args.dataset, train_ratio=1.0)
    torch.cuda.synchronize()
    eval_s = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f'_probe_ckpt_{rank}.pt')
    t0 = time.time()
    save_checkpoint(ckpt_path, raw_model, optimizer, {}, cfg)
    ckpt_s = time.time() - t0
    ckpt_bytes = os.path.getsize(ckpt_path)
    os.remove(ckpt_path)

    result = {
        'world_size': world_size,
        'sparsifier': cfg.sparsifier, 'sparsity': cfg.sparsity,
        'compile': bool(cfg.compile), 'dtype': cfg.dtype,
        'batch_size': cfg.batch_size, 'block_size': cfg.block_size,
        'grad_accum': cfg.gradient_accumulation_steps,
        'tokens_per_step': tokens_per_step,
        'step_seconds': step_s,
        'data_seconds': data_s,
        'compute_seconds': compute_s,
        'tokens_per_second': tokens_per_s,
        'compute_tokens_per_second': tokens_per_step / compute_s,
        'mfu_h100': mfu,
        'compute_mfu_h100': flops_per_token * (tokens_per_step / compute_s) / (H100_BF16_TFLOPS * 1e12),
        'data_fraction': data_s / step_s,
        'mask_update_steps': len(updates),
        'mask_update_seconds': (sum(m for _, m in updates) / len(updates)) if updates else 0.0,
        'mask_update_step_seconds': (sum(t for t, _ in updates) / len(updates)) if updates else 0.0,
        'peak_memory_gb': torch.cuda.max_memory_allocated() / 1e9,
        'compile_seconds': compile_seconds,
        'eval_seconds': eval_s, 'eval_iters': cfg.eval_iters,
        'checkpoint_seconds': ckpt_s, 'checkpoint_gb': ckpt_bytes / 1e9,
        'n_params': n_params,
    }
    name = f"{cfg.sparsifier}_compile{int(bool(cfg.compile))}_b{cfg.batch_size}_n{world_size}"
    if rank == 0:
        with open(os.path.join(args.out_dir, f"{name}.json"), 'w') as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sparsifier', default='dense')
    p.add_argument('--sparsity', type=float, default=0.9)
    p.add_argument('--compile', type=lambda v: v.lower() in ('1', 'true', 'yes'), default=False)
    p.add_argument('--dtype', default='float16')
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--block-size', type=int, default=1024)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--eval-iters', type=int, default=20)
    p.add_argument('--num-mask-updates', type=int, default=500)
    p.add_argument('--total-steps', type=int, default=134_049, help='full-run length, for the schedule')
    p.add_argument('--dataset', default='wikitext')
    p.add_argument('--reuse-batch', action='store_true',
                   help='reuse one batch, so get_batch costs nothing')
    p.add_argument('--ratio', type=float, default=1.0,
                   help='subset ratio to sample from; small values stay in page cache')
    p.add_argument('--data-root', default=os.environ.get('FIRE_DATA_DIR', 'data'))
    p.add_argument('--out-dir', default='profiling')  # repo-relative: results are committed
    profile(p.parse_args())


if __name__ == '__main__':
    main()
