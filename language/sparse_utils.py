"""Sparse training for GPT, on top of the vendored sparsimony.

Three jobs:
  * pick which tensors get a mask (get_sparse_targets)
  * build and prepare a sparsifier (build_sparsifier)
  * report what the masks are doing (sparse_metrics, ITOPTracker)

sparsimony works by reparametrizing a weight: the dense tensor moves to
`module.parametrizations.weight.original` and `module.weight` becomes
mask * original, recomputed on every access. So the sparsifier must be built
BEFORE torch.compile and DDP wrap the model.
"""
import os
import sys

import torch
import torch.nn as nn

# sparsimony is vendored under vision/, not language/.
_SPARSIMONY_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vision", "sparsimony"
)
if os.path.isdir(_SPARSIMONY_REPO) and _SPARSIMONY_REPO not in sys.path:
    sys.path.insert(0, _SPARSIMONY_REPO)

from sparsimony import rigl, static  # noqa: E402
from sparsimony import set as sp_set  # noqa: E402
from sparsimony.distributions.base import UniformDistribution  # noqa: E402
from sparsimony.dst.gmp import GMP  # noqa: E402
from sparsimony.schedulers.base import AcceleratedCubicScheduler  # noqa: E402
from sparsimony.utils import get_mask  # noqa: E402

# The Linear layers inside the transformer blocks. Everything else stays dense:
# lm_head shares its weight with the token embedding (model.py:138), so masking
# it would sparsify the output projection but not the embedding lookup that
# reads the same tensor.
BLOCK_PREFIX = "transformer.h"


def get_sparse_targets(model, prefix=BLOCK_PREFIX):
    """Return sparsimony configs for the block Linear weights (plan D3).

    For GPT-2 small this is 48 tensors holding 84,934,656 of 124,354,560
    weights: c_attn, attn.c_proj, mlp.c_fc and mlp.c_proj in each of 12 blocks.
    """
    return [
        {"tensor_fqn": f"{name}.weight"}
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.startswith(prefix)
    ]


def count_sparse_params(model, targets=None):
    """How many weights the given targets cover."""
    targets = get_sparse_targets(model) if targets is None else targets
    fqns = {t["tensor_fqn"] for t in targets}
    return sum(p.numel() for n, p in model.named_parameters() if n in fqns)


class CorrectedAcceleratedCubicScheduler(AcceleratedCubicScheduler):
    """GMP's cubic ramp, spanning t_accel..t_end.

    sparsimony divides by t_end where the schedule should span (t_end - t_accel),
    so pruning stops short of the target: with t_accel/t_end = 0.25 it ends at
    0.8969 instead of 0.9. Set cfg.gmp_correct_cubic=False for sparsimony's
    original behaviour (what vision/train_st.py uses).
    """

    def __call__(self, step):
        if step > self.t_end:
            return None
        if step % self.delta_t != 0:
            return None
        if step < self.t_accel:
            return self.initial_sparsity
        span = max(1, self.t_end - self.t_accel)
        progress = min(1.0, (step - self.t_accel) / span)
        return (self.final_sparsity
                + (self.accelerated_sparsity - self.final_sparsity) * (1 - progress) ** 3)


def sparsifier_schedule(cfg, total_steps):
    """Derive the mask-update schedule from the total number of optimizer steps.

    Spans both chunks (plan D2), so topology updates stop at t_end_ratio of the
    whole run, not of each chunk.
    """
    t_end = int(cfg.t_end_ratio * total_steps)
    t_accel = int(cfg.t_accel_ratio * total_steps)
    if cfg.sparsifier == "gmp":
        delta_t = max(1, (t_end - t_accel) // cfg.num_mask_updates)
    else:
        delta_t = max(1, t_end // cfg.num_mask_updates)
    return {"t_end": t_end, "t_accel": t_accel, "delta_t": delta_t}


def build_sparsifier(cfg, model, optimizer, total_steps):
    """Create, prepare and return a sparsifier. None when cfg.sparsifier == 'dense'.

    Must be called after the optimizer exists (the sparsifier hooks it to zero
    momentum for masked weights) and before torch.compile / DDP.
    """
    if cfg.sparsifier == "dense":
        return None

    sched = sparsifier_schedule(cfg, total_steps)
    t_end, t_accel, delta_t = sched["t_end"], sched["t_accel"], sched["delta_t"]

    if cfg.sparsifier == "rigl":
        sparsifier = rigl(optimizer, sparsity=cfg.sparsity, t_end=t_end,
                          delta_t=delta_t, pruning_ratio=cfg.pruning_ratio)
    elif cfg.sparsifier == "set":
        sparsifier = sp_set(optimizer, sparsity=cfg.sparsity, t_end=t_end,
                            delta_t=delta_t, pruning_ratio=cfg.pruning_ratio)
    elif cfg.sparsifier == "gmp":
        scheduler_cls = (CorrectedAcceleratedCubicScheduler
                         if getattr(cfg, "gmp_correct_cubic", True)
                         else AcceleratedCubicScheduler)
        sparsifier = GMP(
            scheduler=scheduler_cls(
                t_end=t_end, delta_t=delta_t, t_accel=t_accel,
                initial_sparsity=cfg.initial_sparsity,
                accelerated_sparsity=cfg.accelerated_sparsity,
                final_sparsity=cfg.sparsity,
            ),
            distribution=UniformDistribution(),
            optimizer=optimizer,
        )
    elif cfg.sparsifier == "static":
        sparsifier = static(optimizer, sparsity=cfg.sparsity)
    else:
        raise ValueError(f"unknown sparsifier {cfg.sparsifier!r}")

    targets = get_sparse_targets(model)
    if not targets:
        raise ValueError(
            f"no sparsifiable layers found under {BLOCK_PREFIX!r}; is this a GPT model?"
        )
    # Count before prepare: reparametrization renames weight -> parametrizations.weight.original
    n_params = count_sparse_params(model, targets)
    sparsifier.prepare(model, targets)
    print(f"[sparsifier] {cfg.sparsifier} sparsity={cfg.sparsity} "
          f"tensors={len(targets)} params={n_params:,} "
          f"total_steps={total_steps:,} t_end={t_end:,} delta_t={delta_t:,}"
          + (f" t_accel={t_accel:,}" if cfg.sparsifier == "gmp" else ""))
    return sparsifier


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def get_sparsity_stats(model):
    """Mask sparsity (zeros in the mask) and weight sparsity (zeros in the
    tensor the model actually uses) over all reparametrized layers."""
    total = zero_mask = zero_weight = 0
    modules = 0
    for _, module in model.named_modules():
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            modules += 1
            with torch.no_grad():
                mask = get_mask(module, "weight")
                n = module.parametrizations.weight.original.numel()
                total += n
                zero_mask += n - int(mask.sum().item())
                zero_weight += n - int(torch.count_nonzero(module.weight).item())
    if total == 0:
        return {"mask_sparsity": 0.0, "weight_sparsity": 0.0, "num_sparse_modules": 0}
    return {
        "mask_sparsity": zero_mask / total,
        "weight_sparsity": zero_weight / total,
        "num_sparse_modules": modules,
    }


def get_current_pruning_ratio(sparsifier):
    """The scheduler's value at the most recent update step, or None."""
    if sparsifier is None:
        return None
    scheduler = getattr(sparsifier, "scheduler", None)
    if scheduler is None or not hasattr(scheduler, "delta_t"):
        return None
    delta_t = scheduler.delta_t
    step = sparsifier._step_count
    last_step = max(delta_t, (step // delta_t) * delta_t)
    return scheduler(last_step)


class ITOPTracker:
    """In-Time Over-Parameterization: the fraction of weights that have been
    active at any point. 1.0 means training has explored every weight.

    Create after prepare(); call update() whenever step() reports a topology change.
    """

    def __init__(self, sparsifier):
        self._sparsifier = sparsifier
        self._union = []
        self._total = 0
        for config in sparsifier.groups:
            mask = get_mask(config["module"], config["tensor_name"])
            self._union.append(mask.detach().cpu().clone())
            self._total += mask.numel()

    def update(self):
        for i, config in enumerate(self._sparsifier.groups):
            mask = get_mask(config["module"], config["tensor_name"])
            self._union[i] |= mask.detach().cpu()

    def compute(self):
        if self._total == 0:
            return 0.0
        return sum(int(m.sum().item()) for m in self._union) / self._total


def sparse_metrics(model, sparsifier, itop_tracker=None):
    """Everything worth logging about the masks, ready for wandb."""
    if sparsifier is None:
        return {}
    stats = get_sparsity_stats(model)
    metrics = {
        "dst/mask_sparsity": stats["mask_sparsity"],
        "dst/weight_sparsity": stats["weight_sparsity"],
        "dst/num_sparse_modules": stats["num_sparse_modules"],
    }
    if itop_tracker is not None:
        metrics["dst/itop_rate"] = itop_tracker.compute()
    pruning_ratio = get_current_pruning_ratio(sparsifier)
    if pruning_ratio is not None:
        metrics["dst/pruning_ratio"] = pruning_ratio
    return metrics


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def flatten_sparse_state_dict(state_dict):
    """Turn a reparametrized state dict back into a plain GPT one.

    sparsimony stores `X.parametrizations.weight.original` plus a mask buffer at
    `X.parametrizations.weight.0.mask`, which a plain GPT cannot load. This
    folds the mask into the weight (so the saved tensor is genuinely sparse) and
    returns the masks separately, keyed by the weight they belong to.

    Returns (model_state_dict, masks). masks is empty for a dense model.
    """
    masks = {}
    for key, value in state_dict.items():
        if ".parametrizations." in key and key.endswith(".mask"):
            # X.parametrizations.weight.0.mask -> X.weight
            prefix, rest = key.split(".parametrizations.", 1)
            tensor_name = rest.split(".", 1)[0]
            masks[f"{prefix}.{tensor_name}"] = value

    flat = {}
    for key, value in state_dict.items():
        if ".parametrizations." not in key:
            flat[key] = value
            continue
        if key.endswith(".mask"):
            continue
        if key.endswith(".original"):
            prefix, rest = key.split(".parametrizations.", 1)
            tensor_name = rest.split(".", 1)[0]
            weight_key = f"{prefix}.{tensor_name}"
            mask = masks.get(weight_key)
            flat[weight_key] = value * mask if mask is not None else value
    return flat, masks
