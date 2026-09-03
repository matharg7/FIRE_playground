import gc
import hashlib
import math
import sys
import time
import os
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import numpy as np

from models import get_resnet18_CIFAR10, get_TinyViT_CIFAR100, get_VGG16_TinyImageNet
from task import TASKS
from dst_log_utils import ITOPTracker, get_sparsity_stats, get_current_pruning_ratio

# Add the bundled sparsimony repo to sys.path once at import time.
_sparsimony_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparsimony')
if os.path.isdir(_sparsimony_path) and _sparsimony_path not in sys.path:
    sys.path.insert(0, _sparsimony_path)


# ---------------------------------------------------------------------------
# Model / task / optimizer helpers
# ---------------------------------------------------------------------------

def get_optimizer(model, cfg):
    OPTIMIZERS = {'adam': torch.optim.Adam}
    if cfg.optimizer in OPTIMIZERS:
        return OPTIMIZERS[cfg.optimizer](model.parameters(), lr=cfg.lr)
    print(f"Optimizer '{cfg.optimizer}' not supported. Falling back to adam.")
    return torch.optim.Adam(model.parameters(), lr=cfg.lr)


def build_model(cfg):
    if cfg.model == 'RESNET18':
        assert cfg.task == 'CIFAR10'
        return get_resnet18_CIFAR10()
    elif cfg.model == 'TinyViT':
        assert cfg.task == 'CIFAR100'
        return get_TinyViT_CIFAR100()
    elif cfg.model == 'VGG16':
        assert cfg.task == 'TinyImageNet'
        return get_VGG16_TinyImageNet()
    raise ValueError(f"Invalid model: {cfg.model}")


def get_task(cfg):
    if cfg.benchmark == 'class_incremental':
        assert cfg.model in ('TinyViT', 'VGG16')
    benchmark_settings = {
        'warm_start':        {'n_epochs': 100, 'n_chunks': 2,  'mode': 'sample'},
        'continual':         {'n_epochs': 100, 'n_chunks': 10, 'mode': 'sample'},
        'class_incremental': {'n_epochs': 100, 'n_chunks': 20, 'mode': 'class'},
    }
    s = benchmark_settings[cfg.benchmark]
    task = TASKS[cfg.task](
        mode=s['mode'],
        n_chunks=s['n_chunks'],
        make_test_loader=True,
        access=cfg.access,
        test_access='same',
        seed=cfg.seed,
        warm_start_subset_ratio=cfg.warm_start_subset_ratio,
    )
    cfg.n_epochs = s['n_epochs']
    return task


# ---------------------------------------------------------------------------
# Gradient-step counter 
# ---------------------------------------------------------------------------

def compute_total_gradient_steps(cfg, task):
    """Return the total number of optimizer.step() calls for the full run.

    Uses actual chunk sizes from the task object so the count is exact
    regardless of benchmark / dataset / batch_size.  This value is used to
    derive sparsifier hyperparameters (t_end, delta_t) so they scale
    correctly without manual tuning.
    """
    total = 0
    for i_iter in range(task.n_chunks):
        if cfg.benchmark == 'warm_start' and i_iter == 0:
            log_every = 100 // cfg.warm_start_subset_ratio
        else:
            log_every = cfg.log_every
        real_epochs = cfg.n_epochs * log_every
        chunk_size = len(task._train_datasets[i_iter])
        steps_per_epoch = math.ceil(chunk_size / cfg.batch_size)
        total += real_epochs * steps_per_epoch
    return total


# ---------------------------------------------------------------------------
# Sparsifier factory
# ---------------------------------------------------------------------------

def build_sparsifier(cfg, model, optimizer, total_steps):
    """Create, configure, and prepare the requested sparsifier.

    Derived hyperparameters (printed for reproducibility):
        t_end   = cfg.t_end_ratio  * total_steps          (all sparse methods)
        delta_t = total_steps // cfg.num_mask_updates      (all sparse methods)
        t_accel = cfg.t_accel_ratio * total_steps          (GMP only)

    Returns None when cfg.sparsifier == 'dense'.

    Supported values of cfg.sparsifier:
        'dense'  – no sparsification (baseline)
        'rigl'   – Rig-L  (ERK distribution, cosine decay schedule)
        'set'    – SET    (uniform distribution, constant schedule)
        'gmp'    – GMP*   (uniform distribution, accelerated cubic schedule)
        'static' – Static magnitude pruning (one-shot, no regrowth)
    """
    if cfg.sparsifier == 'dense':
        return None

    from sparsimony import rigl, gmp, static
    from sparsimony import set as sp_set  # avoid shadowing Python's built-in

    t_end = int(cfg.t_end_ratio * total_steps)

    if cfg.sparsifier == 'rigl':
        delta_t = max(1, t_end // cfg.num_mask_updates)
        sparsifier = rigl(
            optimizer,
            sparsity=cfg.sparsity,
            t_end=t_end,
            delta_t=delta_t,
            pruning_ratio=cfg.pruning_ratio,
        )

    elif cfg.sparsifier == 'set':
        delta_t = max(1, t_end // cfg.num_mask_updates)
        sparsifier = sp_set(
            optimizer,
            sparsity=cfg.sparsity,
            t_end=t_end,
            delta_t=delta_t,
            pruning_ratio=cfg.pruning_ratio,
        )

    elif cfg.sparsifier == 'gmp':
        t_accel = int(cfg.t_accel_ratio * total_steps)
        delta_t = max(1, (t_end - t_accel) // cfg.num_mask_updates)
        sparsifier = gmp(
            optimizer,
            t_accel=t_accel,
            t_end=t_end,
            delta_t=delta_t,
            initial_sparsity=cfg.initial_sparsity,
            final_sparsity=cfg.sparsity,
        )

    elif cfg.sparsifier == 'static':
        sparsifier = static(
            optimizer,
            sparsity=cfg.sparsity,
        )

    else:
        raise ValueError(
            f"Unknown sparsifier '{cfg.sparsifier}'. "
            "Choose from: dense, rigl, set, gmp, static"
        )

    # Prepare: reparametrize all Conv2d and Linear weight tensors
    sparse_config = [
        {"tensor_fqn": f"{fqn}.weight"}
        for fqn, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
    ]
    sparsifier.prepare(model, sparse_config)

    
    print(
        f"[Sparsifier] {cfg.sparsifier} | sparsity={cfg.sparsity} | "
        f"total_steps={total_steps} | t_end={t_end} "#| delta_t={delta_t}"
    )
  
    return sparsifier


# ---------------------------------------------------------------------------
# W&B run-name builder
# ---------------------------------------------------------------------------

def build_run_name(cfg, sparsifier) -> str:
    base = f"{cfg.model}_{cfg.task}_{cfg.access}"
    if cfg.sparsifier == 'dense':
        return f"{base}_dense"
    # delta_t is stored on the scheduler for all non-static methods
    dt = getattr(getattr(sparsifier, 'scheduler', None), 'delta_t', None)
    if cfg.sparsifier in ('rigl', 'set'):
        return (
            f"{base}_dst"
            f"_sparsity_{cfg.sparsity}"
            f"_pruning_ratio_{cfg.pruning_ratio}"
            f"_delta_t_{dt}"
        )
    if cfg.sparsifier == 'gmp':
        return (
            f"{base}_gmp"
            f"_accel_sparsity_{cfg.initial_sparsity}"
            f"_final_sparsity_{cfg.sparsity}"
            f"_delta_t_{dt}"
        )
    if cfg.sparsifier == 'static':
        return f"{base}_static_sparsity_{cfg.sparsity}"
    return f"{base}_{cfg.sparsifier}"


# ---------------------------------------------------------------------------
# Checkpointing  (added for the class-incremental campaign)
#
# WHY: class_incremental + access=full is ~206,100 optimizer steps per run, and the
# gpu partition caps a job at 6h (scontrol show partition -> MaxTime=06:00:00). A run
# therefore spans several jobs and must be resumable.
#
# WHAT IS SAVED, and why exactly these things:
#   model.state_dict()  - weights AND the 74 sparsimony mask buffers. The masks live
#                         under ...parametrizations.weight.0.mask, so saving the model
#                         state dict saves the sparse topology. Verified 2026-09-03:
#                         all 74 masks and the logits come back bit-identical.
#   optimizer.state_dict() - Adam moments. The optimizer is rebuilt every stage, so
#                         this belongs to the stage being resumed into.
#   scheduler.state_dict() - otherwise a mid-stage resume restarts the cosine LR.
#   sparsifier._step_count - the sparsifier's own clock; drives delta_t / t_end.
#   RNG states           - so data order after a resume matches an uninterrupted run.
#
# WHAT MUST NOT BE SAVED:
#   the sparsifier object / its state_dict() -> holds references to parametrized
#     modules; raises "Serialization of parametrized modules is only supported
#     through state_dict()".
#   sparsifier._global_step -> it is a METHOD, not an int.
#
# Training maths is untouched: same data, same order, same optimizer, same schedule.
# ---------------------------------------------------------------------------

def _rng_state():
    return {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(d):
    if d is None:
        return
    if d.get('torch') is not None:
        torch.set_rng_state(d['torch'].cpu() if hasattr(d['torch'], 'cpu') else d['torch'])
    if d.get('numpy') is not None:
        np.random.set_state(d['numpy'])
    if d.get('cuda') is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([t.cpu() if hasattr(t, 'cpu') else t for t in d['cuda']])
        except Exception as e:              # different GPU count on the new node
            print(f"[ckpt] could not restore cuda RNG ({e}); continuing")


def save_checkpoint(path, *, model, optimizer, scheduler, sparsifier,
                    i_iter, epoch, real_epochs, global_epoch, global_step,
                    wandb_run_id):
    """Atomic: write to <path>.tmp then rename, so a kill mid-write cannot corrupt."""
    if not path:
        return
    ckpt = {
        'format': 1,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'sched': scheduler.state_dict() if scheduler is not None else None,
        'sp_step_count': int(sparsifier._step_count) if sparsifier is not None else None,
        'i_iter': int(i_iter),
        'epoch': int(epoch),
        'real_epochs': int(real_epochs),
        'global_epoch': int(global_epoch),
        'global_step': int(global_step),
        'rng': _rng_state(),
        'wandb_run_id': wandb_run_id,
    }
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device):
    if not path or not os.path.exists(path):
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get('format') != 1:
        raise ValueError(f"unrecognised checkpoint format in {path}")
    return ck


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(cfg):
    cfg.print()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
    print(f"Using device: {device}")

    # Build task first so cfg.n_epochs is populated before step counting
    task = get_task(cfg)
    model = build_model(cfg).to(device)
    optimizer = get_optimizer(model, cfg)

    total_steps = compute_total_gradient_steps(cfg, task)
    # init DDP AFTER reparametrization if using distributed training
    sparsifier = build_sparsifier(cfg, model, optimizer, total_steps)
    itop_tracker = ITOPTracker(sparsifier) if sparsifier is not None else None

    criterion = nn.CrossEntropyLoss()
    initial_lr = 0.0
    warmup_rate = 0.1

    wandb_project = cfg.wandb_project or f"{cfg.benchmark}_{cfg.task}_{cfg.model}"
    _run_name = build_run_name(cfg, sparsifier)
    # Deterministic id so a resumed run continues the SAME W&B run instead of
    # creating a duplicate. Overridable with --wandb-run-id.
    _wandb_id = cfg.wandb_run_id or hashlib.md5(
        f"{_run_name}|seed{cfg.seed}|{cfg.benchmark}|{cfg.access}".encode()).hexdigest()[:16]
    wandb.init(
        # entity="ucalgary",
        project= wandb_project,
        name=_run_name,
        id=_wandb_id,
        resume="allow",
        config=cfg.__dict__,
        mode="disabled" if cfg.disable_wandb else "online",
    )

    global_epoch = 0
    global_step = 0
    _last_ckpt_t = time.time()

    # ---- resume, if asked and a checkpoint exists ----
    start_iter, start_epoch = 0, 0
    _ck = load_checkpoint(cfg.ckpt_path, device) if cfg.resume else None
    if _ck is not None:
        model.load_state_dict(_ck['model'])
        if sparsifier is not None and _ck.get('sp_step_count') is not None:
            sparsifier._step_count = int(_ck['sp_step_count'])
        global_epoch = _ck['global_epoch']
        global_step  = _ck['global_step']
        start_iter, start_epoch = _ck['i_iter'], _ck['epoch'] + 1
        if start_epoch >= _ck['real_epochs']:      # that stage was finished
            start_iter, start_epoch = start_iter + 1, 0
        _restore_rng(_ck.get('rng'))
        print(f"[ckpt] resumed {cfg.ckpt_path}: stage {start_iter}, epoch {start_epoch}, "
              f"global_step {global_step}, sparsifier step {_ck.get('sp_step_count')}")
        if start_iter >= task.n_chunks:
            print("[ckpt] checkpoint says the run is already complete; exiting.")
            wandb.finish()
            return
    elif cfg.resume:
        print(f"[ckpt] --resume set but no checkpoint at {cfg.ckpt_path}; starting fresh")

    for i_iter in range(task.n_chunks):
        if i_iter < start_iter:                    # already done in an earlier job
            continue
        if cfg.benchmark == 'warm_start' and i_iter == 0:
            log_every = 100 // cfg.warm_start_subset_ratio
        else:
            log_every = cfg.log_every

        trainloader = task.set_level(i_iter, batch_size=cfg.batch_size)
        real_epochs = cfg.n_epochs * log_every
        pbar = tqdm(range(real_epochs), leave=True)

        # Reset optimizer at every iteration; hand the new instance to sparsifier
        optimizer = get_optimizer(model, cfg)
        target_lr = [pg['lr'] for pg in optimizer.param_groups]

        # Create the LR scheduler for this chunk
        if cfg.use_cosine_lr:
            T_max = cfg.cosine_T_max_epochs if cfg.cosine_T_max_epochs > 0 else real_epochs
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=cfg.cosine_eta_min)

        if sparsifier is not None:
            sparsifier.optimizer = optimizer
            if hasattr(sparsifier, 'zero_inactive_param_momentum_buffers'):
                sparsifier.zero_inactive_param_momentum_buffers()

        # Resuming into a partially-done stage: the optimizer and LR scheduler were
        # just rebuilt, so put back the state they had when the checkpoint was taken.
        if _ck is not None and i_iter == start_iter and start_epoch > 0:
            optimizer.load_state_dict(_ck['opt'])
            if cfg.use_cosine_lr and _ck.get('sched') is not None:
                cosine_scheduler.load_state_dict(_ck['sched'])
            print(f"[ckpt] restored optimizer/scheduler into stage {i_iter} "
                  f"at epoch {start_epoch}")

        for epoch in pbar:
            # epochs already completed before the checkpoint
            if _ck is not None and i_iter == start_iter and epoch < start_epoch:
                continue
            pbar.set_description(f'Iter {i_iter} | Epoch {epoch}')
            do_logging = global_epoch % log_every == 0

            if cfg.use_cosine_lr:
                cosine_scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
            else:
                # Warmup LR scheduling (from https://arxiv.org/abs/2406.02596)
                ls = global_step % cfg.n_epochs
                we = cfg.n_epochs * warmup_rate
                remain = (epoch + 1) / cfg.n_epochs - int((epoch + 1) / cfg.n_epochs)
                for i, pg in enumerate(optimizer.param_groups):
                    if ls < we:
                        current_lr = initial_lr + (target_lr[i] - initial_lr) * remain * (10 // log_every)
                    else:
                        current_lr = target_lr[i]
                    pg['lr'] = current_lr

            total = correct = 0
            for inputs, labels, _orig_idx, _chunk_idx in trainloader:
                model.train()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()

                if cfg.clip_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)

                optimizer.step()
                if sparsifier is not None:
                    if sparsifier.step():
                        itop_tracker.update()

            train_acc = correct / total

            if do_logging:
                global_step += 1
                log_dict = {
                    'train/acc': train_acc,
                    'train/lr': current_lr,
                    'level': i_iter,
                    'global_step': global_step,
                    'global_epoch': global_epoch,
                    'iter': i_iter,
                }
                p_fix = {'acc': train_acc, 'lr': current_lr}

                test_acc, _ = task.test(model, device)
                log_dict['test/acc'] = test_acc
                p_fix['test_acc'] = test_acc

                if cfg.benchmark == 'class_incremental':
                    acc_full, _ = task.test(model, device, full=True)
                    log_dict['test/acc_full'] = acc_full
                    p_fix['test_acc_full'] = acc_full

                if sparsifier is not None:
                    sparsity_stats = get_sparsity_stats(model)
                    log_dict['dst/mask_sparsity']  = sparsity_stats['mask_sparsity']
                    log_dict['dst/weight_sparsity'] = sparsity_stats['weight_sparsity']
                    log_dict['dst/itop_rate']       = itop_tracker.compute()
                    pruning_ratio = get_current_pruning_ratio(sparsifier)
                    if pruning_ratio is not None:
                        log_dict['dst/pruning_ratio'] = pruning_ratio

                wandb.log(log_dict, step=global_step)
                pbar.set_postfix(**p_fix)

            global_epoch += 1

            # Periodic checkpoint. Worst case lost on a wall clock kill is
            # ckpt_every_epochs epochs. Written atomically (.tmp + rename).
            # Trigger on ELAPSED TIME, not epoch count: early stages run an epoch
            # in ~2.4 s, so an epoch-count trigger would write 91 MB every few
            # seconds x 44 processes onto NFS. Always save at a stage boundary.
            _due = (time.time() - _last_ckpt_t) >= cfg.ckpt_min_interval_sec
            if cfg.ckpt_path and (_due or epoch == real_epochs - 1):
                _last_ckpt_t = time.time()
                save_checkpoint(
                    cfg.ckpt_path, model=model, optimizer=optimizer,
                    scheduler=cosine_scheduler if cfg.use_cosine_lr else None,
                    sparsifier=sparsifier, i_iter=i_iter, epoch=epoch,
                    real_epochs=real_epochs, global_epoch=global_epoch,
                    global_step=global_step, wandb_run_id=_wandb_id)

        del trainloader
        torch.cuda.empty_cache()
        gc.collect()

    wandb.finish()


if __name__ == "__main__":
    from config_st import get_config
    cfg = get_config()
    main(cfg)
