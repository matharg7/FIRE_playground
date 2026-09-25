"""Continual fine-tuning on TRACE with full cumulative replay.

For each task t in order: train on the union of the train splits of tasks
0..t, then evaluate tasks 0..t by generation and record R[t][:] in the score
matrix. One process, one GPU, plain PyTorch (torch AdamW, fp32 master weights
under autocast), no DeepSpeed.

    python src/train.py --model Qwen/Qwen2.5-0.5B --learning_rate 5e-5

Outputs in <out_root>/<run_name>/: config.json, scores.json (the matrix),
eval/step<t>_<task>.json (predictions), summary.json, checkpoints, and a DONE
marker written last (scripts/sync_wandb.sh only syncs runs that have one).
"""

import json
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

import data
import evaluate
import sparse_utils
import wsc as wsc_mod
from cl_metrics import ScoreMatrix
from config import get_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_dir(explicit, env_var, fallback):
    """Config value, else the environment (scripts/env.sh), else a local path."""
    return explicit or os.environ.get(env_var) or fallback


def build_run_name(cfg):
    """A name that distinguishes the runs of a sweep from each other.

    Under a W&B sweep every run is launched from the same command, so the name
    has to carry whatever the sweep varies; a timestamp alone would leave two
    runs differing only in sparsity looking identical.
    """
    if cfg.run_name:
        return cfg.run_name
    parts = [cfg.model.split('/')[-1], cfg.sparsifier]
    if cfg.sparsifier != 'dense':
        parts += [f"s{cfg.sparsity:g}", cfg.sparse_targets, cfg.sparse_distribution,
                  cfg.grow_init, cfg.drop_fraction_schedule,
                  f"pr{cfg.pruning_ratio:g}", f"nmu{cfg.num_mask_updates}"]
    # The CL method is orthogonal to the sparsifier, so it has to appear too:
    # without it a dense+WSC run is named identically to a plain dense run.
    if cfg.cl_method != 'none':
        parts.append(cfg.cl_method)
        if cfg.cl_method == 'wsc':
            parts += [f"pat{cfg.wsc_patience}", f"ret{cfg.wsc_retain_percent:g}"]
    parts += [f"lr{cfg.learning_rate:g}", f"seed{cfg.seed}",
              time.strftime('%Y%m%d-%H%M%S')]
    return "_".join(parts)


def get_lr(cfg, step, total_steps):
    """Learning rate for this step of a task.

    'cosine'   linear warmup then cosine decay to learning_rate * min_lr_ratio,
               restarting each task.
    'constant' warmup then a flat learning_rate, which is what TRACE uses:
               get_constant_schedule_with_warmup with --num_warmup_steps 0.
    """
    warmup = max(1, int(cfg.warmup_ratio * total_steps))
    peak, floor = cfg.learning_rate, cfg.learning_rate * cfg.min_lr_ratio
    if step < warmup:
        return peak * (step + 1) / warmup
    if cfg.lr_schedule == 'constant':
        return peak
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return floor + 0.5 * (1.0 + math.cos(math.pi * progress)) * (peak - floor)


def steps_for_task(cfg, n_batches, t=0):
    """Optimizer steps for task t: its epochs over the cumulative loader, optionally capped."""
    epochs = cfg.epochs_list()[t]
    steps = math.ceil(epochs * n_batches / cfg.gradient_accumulation_steps)
    return min(steps, cfg.max_steps_per_task) if cfg.max_steps_per_task > 0 else steps


def per_task_run_steps(cfg, task_data):
    """Optimizer steps for each task, in order."""
    return [steps_for_task(cfg, math.ceil(len(data.cumulative_train(task_data, t))
                                          / cfg.batch_size), t)
            for t in range(len(task_data))]


def total_run_steps(cfg, task_data):
    """Optimizer steps over the whole run; the mask schedule spans all tasks."""
    return sum(per_task_run_steps(cfg, task_data))


def save_checkpoint(model, tokenizer, path):
    """A plain HF checkpoint (masks folded into the weights) plus masks.pt if sparse."""
    flat, masks = sparse_utils.flatten_sparse_state_dict(model.state_dict())
    model.save_pretrained(path, state_dict=flat)
    tokenizer.save_pretrained(path)
    if masks:
        torch.save({k: v.cpu() for k, v in masks.items()}, os.path.join(path, 'masks.pt'))


def position_ids_for(attention_mask):
    """Positions that start at 0 on the first real token of left-padded rows,
    matching what generate() uses (RoPE is relative, but keep them identical)."""
    return (attention_mask.cumsum(-1) - 1).clamp(min=0)


def loss_and_task_stats(model, batch, device, ctx, num_tasks):
    """Token-mean loss for backward, plus detached per-task token loss sums/counts."""
    input_ids = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    labels = batch['labels'].to(device, non_blocking=True)
    with ctx:
        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       position_ids=position_ids_for(attention_mask)).logits
    targets = labels[:, 1:]
    tok_loss = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), targets.flatten(),
                               ignore_index=-100, reduction='none').view_as(targets)
    mask = (targets != -100).float()
    loss = (tok_loss * mask).sum() / mask.sum().clamp(min=1)
    task_ids = batch['task_ids'].to(device)
    per_ex = (tok_loss.detach() * mask).sum(1)
    sums = torch.zeros(num_tasks, device=device).index_add_(0, task_ids, per_ex)
    counts = torch.zeros(num_tasks, device=device).index_add_(0, task_ids, mask.sum(1))
    return loss, sums, counts, int(attention_mask.sum())


def build_model(cfg, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32,
                                                 attn_implementation='sdpa')
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model.to(device)


def build_optimizer(cfg, model):
    """Plain torch AdamW (sparsimony checks the exact type). No decay on biases/norms."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{'params': decay, 'weight_decay': cfg.weight_decay},
              {'params': no_decay, 'weight_decay': 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
                             fused=torch.cuda.is_available())


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def write_done_marker(out_dir, run_name, wandb_run_dir, wandb_mode):
    """Mark a run as finished cleanly; runs without DONE are never synced."""
    path = os.path.join(out_dir, 'DONE')
    fields = {'run_name': run_name, 'out_dir': os.path.abspath(out_dir),
              'wandb_run_dir': wandb_run_dir or '', 'wandb_mode': wandb_mode,
              'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(path, 'w') as f:
        for key, value in fields.items():
            f.write(f"{key}={value}\n")
    return path


class NullLogger:
    run_dir = ''

    def log(self, metrics, step):
        pass

    def summary(self, metrics):
        pass

    def finish(self):
        pass


class WandbLogger:
    def __init__(self, cfg, run_name):
        import wandb
        wandb_dir = resolve_dir(cfg.wandb_dir, 'TRACE_WANDB_DIR', 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)
        self.wandb = wandb
        self.run = wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity or None,
                              name=run_name, config=cfg.as_dict(), mode=cfg.wandb_mode,
                              dir=wandb_dir)
        # run.dir is the run's files/ subdirectory; wandb sync wants its parent.
        self.run_dir = os.path.dirname(self.run.dir) if getattr(self.run, 'dir', None) else ''

    def log(self, metrics, step):
        self.wandb.log(metrics, step=step)

    def summary(self, metrics):
        self.run.summary.update(metrics)

    def finish(self):
        self.wandb.finish()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_task(cfg, t, model, optimizer, loader, device, ctx, num_tasks, logger, global_step,
               sparsifier=None, itop=None, wsc=None):
    """Train on one task's cumulative loader. Returns (global_step, history)."""
    total = steps_for_task(cfg, len(loader), t)
    accum = cfg.gradient_accumulation_steps
    history = []
    skipped = [0]   # steps dropped for non-finite gradients; list so it stays mutable
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # perf/max_mem_gb is per task
    model.train()
    it = iter(loader)
    epoch = 0
    win_sums = torch.zeros(num_tasks, device=device)
    win_counts = torch.zeros(num_tasks, device=device)
    win_tokens, win_start = 0, time.time()
    for step in range(total):
        if wsc is not None and wsc.swa_active:
            # SWALR owns the LR from here; get_lr would overwrite it every step.
            lr = optimizer.param_groups[0]['lr']
        else:
            lr = get_lr(cfg, step, total)
            for group in optimizer.param_groups:
                group['lr'] = lr
        step_loss = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                epoch += 1
                it = iter(loader)
                batch = next(it)
                if wsc is not None:
                    wsc.on_epoch_end(model, optimizer, epoch, t)
            loss, sums, counts, n_tok = loss_and_task_stats(model, batch, device, ctx, num_tasks)
            (loss / accum).backward()
            step_loss += loss.item() / accum
            win_sums += sums
            win_counts += counts
            win_tokens += n_tok
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # Skip the step when the gradient is not finite, as DeepSpeed's engine
        # does for TRACE. clip_grad_norm_ scales every parameter by
        # max_norm/total_norm, so a single inf gradient makes that factor NaN
        # and would otherwise turn all 290 tensors into NaN permanently -- one
        # bad micro-batch kills the whole run. Skipping discards this step's
        # gradients and carries on with the weights intact.
        if not torch.isfinite(grad_norm):
            skipped[0] += 1
            optimizer.zero_grad(set_to_none=True)
            if skipped[0] <= 5 or skipped[0] % 50 == 0:
                print(f"task {t} step {step + 1}/{total}: non-finite grad norm "
                      f"({grad_norm}), step skipped ({skipped[0]} so far)", flush=True)
            # A few skips are normal in bf16; a flood means the run is not
            # training and should be stopped rather than quietly degraded.
            if step + 1 >= 50 and skipped[0] > cfg.max_skipped_ratio * (step + 1):
                raise RuntimeError(
                    f"{skipped[0]} of {step + 1} steps skipped for non-finite gradients "
                    f"(> {cfg.max_skipped_ratio:.0%}). Training is not converging; "
                    f"try --dtype float32 or a smaller --learning_rate.")
            continue
        if wsc is not None:
            wsc.on_optimizer_step(model)     # EMA of |g| and g^2, before grads are cleared
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # Once per optimizer step, after it (never per micro-step): advances the
        # schedule and, every delta_t steps, prunes and regrows.
        if sparsifier is not None and sparsifier.step() and itop is not None:
            itop.update()
        global_step += 1
        history.append(step_loss)

        if (step + 1) % cfg.log_interval == 0 or step + 1 == total:
            elapsed = time.time() - win_start
            metrics = {'train/loss': step_loss, 'train/lr': lr, 'train/grad_norm': float(grad_norm),
                       'train/skipped_steps': skipped[0],
                       'train/task': t, 'train/epoch': epoch,
                       'perf/tokens_per_s': win_tokens / max(elapsed, 1e-9),
                       'perf/max_mem_gb': torch.cuda.max_memory_allocated() / 2**30
                       if torch.cuda.is_available() else 0.0}
            per_task = (win_sums / win_counts.clamp(min=1)).tolist()
            for i in range(t + 1):
                if win_counts[i] > 0:
                    metrics[f'train_loss_by_task/{i}'] = per_task[i]
            metrics.update(sparse_utils.sparse_metrics(model, sparsifier, itop))
            logger.log(metrics, global_step)
            print(f"task {t} step {step + 1}/{total} loss {step_loss:.4f} lr {lr:.2e} "
                  f"tok/s {metrics['perf/tokens_per_s']:.0f} mem {metrics['perf/max_mem_gb']:.1f}GB",
                  flush=True)
            win_sums.zero_()
            win_counts.zero_()
            win_tokens, win_start = 0, time.time()
    if wsc is not None:
        # The last epoch never exhausted the iterator, so its end was not
        # signalled; process it so SWA averages every epoch it should.
        wsc.finish_epochs(model, optimizer, t, cfg.epochs_list()[t])
    return global_step, history


def main(cfg):
    cfg.print()
    torch.manual_seed(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ptdtype = getattr(torch, cfg.dtype)
    ctx = (torch.autocast(device_type='cuda', dtype=ptdtype)
           if device == 'cuda' and cfg.dtype != 'float32' else nullcontext())

    run_name = build_run_name(cfg)
    out_dir = os.path.join(resolve_dir(cfg.out_root, 'TRACE_OUTPUT_DIR', 'out'), run_name)
    os.makedirs(os.path.join(out_dir, 'eval'), exist_ok=True)
    write_json(os.path.join(out_dir, 'config.json'), cfg.as_dict())
    print(f"output: {out_dir}")

    tasks = cfg.task_list()
    data_root = resolve_dir(cfg.data_dir, 'TRACE_DATA_DIR', 'data')
    task_data = data.load_tasks(data_root, tasks=tasks, subset=cfg.subset,
                                max_train=cfg.max_train_per_task or None,
                                max_eval=cfg.max_eval_per_task or None,
                                max_test=cfg.max_eval_per_task or None, seed=cfg.seed)
    tokenizer = data.load_tokenizer(cfg.model)

    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    # After the optimizer (it hooks AdamW to mask moments), before compile.
    per_task_steps = per_task_run_steps(cfg, task_data)
    total_steps = sum(per_task_steps)
    sparsifier = sparse_utils.build_sparsifier(cfg, model, optimizer, total_steps)
    # Before training, not three tasks in: confirm every task actually gets
    # topology updates under this delta_t.
    sparse_utils.check_update_cadence(
        cfg, per_task_steps, sparse_utils.sparsifier_schedule(cfg, total_steps), tasks)

    # ---- Weight Space Consolidation (optional) ----
    wsc_ctl = None
    if cfg.cl_method == 'wsc':
        wsc_mod.check_wsc_schedule(cfg, cfg.epochs_list(), tasks)
        # Validation loss for the plateau check: the cumulative EVAL split of the
        # tasks seen so far, which is a held-out split distinct from the `test`
        # split used for scoring. Rebuilt at each boundary by _wsc_val_loss.
        _wsc_state = {'t': 0}

        def _wsc_val_loss():
            from torch.utils.data import ConcatDataset
            t_now = _wsc_state['t']
            splits = [ev for _, ev, _ in list(task_data.values())[:t_now + 1]]
            ds = ConcatDataset(splits)
            n = min(len(ds), cfg.wsc_val_examples)
            idx = list(range(len(ds)))[:n] if n == len(ds) else \
                  __import__('random').Random(cfg.seed).sample(range(len(ds)), n)
            vl = data.loss_loader(torch.utils.data.Subset(ds, idx), tokenizer,
                                  cfg.eval_batch_size, cfg.max_prompt_len,
                                  cfg.max_ans_len, num_workers=0)
            tot_loss, tot_tok = 0.0, 0
            with torch.no_grad():
                for b in vl:
                    l, _, _, _ = loss_and_task_stats(train_model, b, device, ctx, len(tasks))
                    ntok = int((b['labels'][:, 1:] != -100).sum())
                    tot_loss += float(l) * ntok
                    tot_tok += ntok
            return tot_loss / max(tot_tok, 1)

        wsc_ctl = wsc_mod.WSCController(cfg, model, _wsc_val_loss)
    itop = sparse_utils.ITOPTracker(sparsifier) if sparsifier is not None else None
    train_model = torch.compile(model) if cfg.compile else model
    print(f"total optimizer steps over {len(tasks)} tasks: {total_steps}")

    logger = WandbLogger(cfg, run_name) if cfg.wandb_mode != 'disabled' else NullLogger()
    scores = ScoreMatrix(tasks)
    global_step = 0
    histories = []
    timings = []

    def run_eval(upto):
        with parametrize.cached():
            return evaluate.evaluate_tasks(model, tokenizer, task_data, upto, cfg.max_prompt_len,
                                           split=cfg.eval_split, batch_size=cfg.eval_batch_size,
                                           device=device, max_ans_len=cfg.max_ans_len,
                                           with_loss=cfg.eval_loss)

    if cfg.eval_zero_shot:
        # The untrained model on every task: the reference point for forward
        # transfer, and a record of what the model could already do.
        start = time.time()
        zero = run_eval(len(tasks) - 1)
        scores.record_baselines({name: r['score'] for name, r in zero.items()})
        if cfg.eval_loss:
            scores.record_losses(-1, {name: r['loss'] for name, r in zero.items()})
        scores.save(os.path.join(out_dir, 'scores.json'))
        for i, (name, r) in enumerate(zero.items()):
            write_json(os.path.join(out_dir, 'eval', f'zero_shot_{i}_{name}.json'), r)
        metrics = {f'eval_zero_shot/{name}': r['score'] for name, r in zero.items()}
        metrics.update({f'eval_loss_zero_shot/{name}': r['loss'] for name, r in zero.items()
                        if 'loss' in r})
        logger.log(metrics, 0)
        print("== zero-shot: " + " ".join(f"{n}={r['score']:.3f}" for n, r in zero.items())
              + f" | {time.time() - start:.0f}s", flush=True)

    for t, task in enumerate(tasks):
        if t > 0 and cfg.reset_optimizer:
            # Clear state but keep the optimizer object, so hooks registered on
            # it (sparsimony's momentum masking, Phase 3) survive the boundary.
            optimizer.state.clear()
        train_set = data.cumulative_train(task_data, t)
        loader = data.train_loader(train_set, tokenizer, cfg.batch_size, cfg.max_prompt_len,
                                   cfg.max_ans_len, seed=cfg.seed + t, num_workers=cfg.num_workers,
                                   length_grouped=cfg.length_grouped)
        task_steps = steps_for_task(cfg, len(loader), t)
        # per_task: restart the drop-fraction cosine over this task's steps, so
        # the topology keeps updating in every task instead of freezing at
        # t_end_ratio of the whole run.
        new_t_end = sparse_utils.restart_drop_fraction_schedule(cfg, sparsifier, task_steps)
        if new_t_end is not None:
            print(f"[sparsifier] task {t}: drop-fraction cosine restarted, t_end={new_t_end}")
        print(f"== task {t} ({task}): {len(train_set)} cumulative examples, "
              f"{task_steps} optimizer steps", flush=True)
        if wsc_ctl is not None:
            _wsc_state['t'] = t
            wsc_ctl.begin_task(model, t)
        start = time.time()
        global_step, history = train_task(cfg, t, train_model, optimizer, loader, device, ctx,
                                          len(tasks), logger, global_step, sparsifier, itop,
                                          wsc=wsc_ctl)
        if wsc_ctl is not None:
            wsc_ctl.end_task(model, t)
        train_seconds = time.time() - start
        histories.append(history)
        sparse_at_boundary = sparse_utils.sparse_metrics(model, sparsifier, itop)

        start = time.time()
        # run_eval uses parametrize.cached(): compute each mask * weight once for
        # the whole evaluation, not on every generated token (no-op if dense).
        # eval_lookahead also scores task t+1, which forward transfer needs.
        results = run_eval(min(t + 1, len(tasks) - 1) if cfg.eval_lookahead else t)
        eval_seconds = time.time() - start
        scores.record(t, {name: r['score'] for name, r in results.items()})
        if cfg.eval_loss:
            scores.record_losses(t, {name: r['loss'] for name, r in results.items()})
        scores.save(os.path.join(out_dir, 'scores.json'))
        for i, (name, r) in enumerate(results.items()):
            write_json(os.path.join(out_dir, 'eval', f'step{t}_{i}_{name}.json'), r)

        summary = scores.summary(t)
        metrics = {f'eval/{name}': r['score'] for name, r in results.items()}
        metrics.update({f'eval_loss/{name}': r['loss'] for name, r in results.items()
                        if 'loss' in r})
        metrics.update({f'cl/{k}': v for k, v in summary.items()})
        metrics.update({'time/train_s': train_seconds, 'time/eval_s': eval_seconds})
        logger.log(metrics, global_step)
        # Final training loss on this task's cumulative data, as the mean over
        # the last 10% of its steps. The over-training comparison is read at
        # MATCHED training loss -- a sparse arm that merely fits less has not
        # generalised better -- so this belongs in the output rather than only
        # in the log lines.
        tail = history[max(1, int(0.9 * len(history))):] or history
        timings.append({'task': task, 'train_s': train_seconds, 'eval_s': eval_seconds,
                        'examples': len(train_set), 'steps': len(history),
                        'train_loss_final': sum(tail) / len(tail),
                        'train_loss_first': history[0] if history else None,
                        **sparse_at_boundary})
        print(f"== after task {t}: " + " ".join(f"{n}={r['score']:.3f}" for n, r in results.items())
              + " | " + " ".join(f"{k.upper()} {v:.4f}" for k, v in summary.items())
              + f" "
              f"| train {train_seconds:.0f}s eval {eval_seconds:.0f}s", flush=True)

        if cfg.save_checkpoint == 'every_task' or (cfg.save_checkpoint == 'final' and t == len(tasks) - 1):
            save_checkpoint(model, tokenizer, os.path.join(out_dir, f'checkpoint_task{t}'))

    final = {'op': scores.op(), 'bwt': scores.bwt(), 'global_steps': global_step,
             'total_steps_planned': total_steps, 'timings': timings, **scores.to_dict()}
    if wsc_ctl is not None:
        final['wsc_events'] = wsc_ctl.events
    if scores.can_fwt():
        final['fwt'] = scores.fwt()
    write_json(os.path.join(out_dir, 'summary.json'), final)
    logger.summary({f'final/{k}': final[k] for k in ('op', 'bwt', 'fwt') if k in final})
    logger.finish()
    write_done_marker(out_dir, run_name, logger.run_dir, cfg.wandb_mode)
    print(f"done: OP {final['op']:.4f} BWT {final['bwt']:.4f} -> {out_dir}")
    return {'out_dir': out_dir, 'scores': scores, 'histories': histories, 'model': model,
            'sparsifier': sparsifier, **final}


if __name__ == '__main__':
    main(get_config())
