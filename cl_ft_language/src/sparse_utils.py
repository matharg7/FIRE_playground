"""Sparse training for HuggingFace decoder LLMs, on top of sparsimony.

Adapted from language/sparse_utils.py (GPT-2): same schedule, metrics and
checkpoint folding, but targeting the Linear layers of HF decoder blocks
(model.layers.*: q/k/v/o_proj and gate/up/down_proj) and, for now, RigL only.

Four jobs:
  * pick which tensors get a mask (get_sparse_targets)
  * build and prepare a sparsifier (build_sparsifier)
  * report what the masks are doing (sparse_metrics, ITOPTracker)
  * save a plain, loadable HF state dict with the masks folded in
    (flatten_sparse_state_dict)

sparsimony reparametrizes each target weight: the dense tensor moves to
`module.parametrizations.weight.original` and `module.weight` becomes
mask * original, recomputed on every access. So the sparsifier must be built
after the optimizer exists (it hooks the optimizer to zero AdamW moments of
masked weights) and before torch.compile wraps the model.

Initial mask: sparsimony magnitude-prunes the current (pretrained) weights at
prepare(), per layer, with the sparsity split across layers by the
distribution (ERK by default, as in the GPT-2 sweep).
"""

import torch
import torch.nn as nn
from sparsimony.distributions.base import ERKDistribution, UniformDistribution
from sparsimony.dst.gmp import GMP
from sparsimony.dst.rigl import RigL
from sparsimony.dst.set import SET
from sparsimony.dst.static import StaticMagnitudeSparsifier
from sparsimony.schedulers.base import (AcceleratedCubicScheduler, ConstantScheduler,
                                        CosineDecayScheduler)
from sparsimony.utils import get_mask

# The decoder blocks of Llama/Qwen2-style HF models. Embeddings, lm_head (tied
# to the embeddings in both SmolLM2 and Qwen2.5) and norms stay dense: they are
# never candidates, whichever target set is chosen.
BLOCK_PREFIX = "model.layers."

# Which Linear weights inside a decoder block get a mask. Both SmolLM2 (Llama)
# and Qwen2.5 use these names and a SwiGLU MLP, where gate_proj and up_proj
# both map hidden -> intermediate and down_proj maps back.
#
#   all_linear  attention and MLP        72% of Qwen2.5-0.5B, 79% of SmolLM2
#   mlp         the whole MLP            64% / 59%
#   up_down     up and down only, so     42% / 40%
#               gate_proj stays dense too
#   gate        gate_proj alone          21% / 20%  (the SwiGLU gate: it
#               decides how much of each up_proj channel passes through)
TARGET_SETS = {
    "all_linear": ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "up_down": ("up_proj", "down_proj"),
    "gate": ("gate_proj",),
}
# The experiment prunes the MLP only: attention, the embeddings and the tied
# lm_head stay dense. gate_proj is included because it is an up-projection by
# shape (hidden -> intermediate), the twin of up_proj in the SwiGLU pair.
DEFAULT_TARGETS = "mlp"

DISTRIBUTIONS = {"erk": ERKDistribution, "uniform": UniformDistribution}


def get_sparse_targets(model, target_set=DEFAULT_TARGETS, prefix=BLOCK_PREFIX):
    """sparsimony configs for the Linear weights the target set selects.

    Matching is on the module's own name (the last component of its FQN), so
    only the named projections inside the decoder blocks are ever returned.
    """
    if target_set not in TARGET_SETS:
        raise ValueError(f"sparse_targets must be one of {tuple(TARGET_SETS)}, got {target_set!r}")
    suffixes = TARGET_SETS[target_set]
    return [
        {"tensor_fqn": f"{name}.weight"}
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.startswith(prefix)
        and name.rsplit(".", 1)[-1] in suffixes
    ]


def count_sparse_params(model, targets=None, target_set=DEFAULT_TARGETS):
    """How many weights the given targets cover."""
    targets = get_sparse_targets(model, target_set) if targets is None else targets
    fqns = {t["tensor_fqn"] for t in targets}
    return sum(p.numel() for n, p in model.named_parameters() if n in fqns)


def sparsifier_schedule(cfg, total_steps):
    """Mask-update schedule over the whole run (all tasks), from the total
    number of optimizer steps: updates every delta_t steps until t_end."""
    t_end = int(cfg.t_end_ratio * total_steps)
    delta_t = max(1, t_end // cfg.num_mask_updates)
    return {"t_end": t_end, "delta_t": delta_t}


def updates_per_task(cfg, per_task_steps, delta_t):
    """How many topology updates each task actually receives.

    'per_task' restarts the schedule with t_end = t_end_ratio * that task's
    steps, while delta_t stays global -- so a delta_t derived from the whole
    run can be larger than a short task's entire update window.
    'global' runs one horizon across the run, which starves the early tasks in
    the same way.
    """
    out, global_step = [], 0
    t_end_global = int(cfg.t_end_ratio * sum(per_task_steps))
    for steps in per_task_steps:
        if cfg.drop_fraction_schedule == 'per_task':
            t_end = max(1, int(cfg.t_end_ratio * steps))
            out.append(len(range(delta_t, t_end + 1, delta_t)))
        else:
            n = sum(1 for s in range(global_step + 1, global_step + steps + 1)
                    if s % delta_t == 0 and s <= t_end_global)
            out.append(n)
        global_step += steps
    return out


def check_update_cadence(cfg, per_task_steps, sched, task_names=None):
    """Fail at startup if any task would get too few topology updates.

    delta_t comes from the *whole run*, but the drop-fraction schedule is
    per-task, so changing the effective batch (which changes the step count)
    silently changes how much DST each task sees. That showed up once as a
    rising dst/pruning_ratio three tasks in; it should be a startup error.
    """
    if cfg.sparsifier == 'dense' or cfg.min_updates_per_task <= 0:
        return
    delta_t = sched['delta_t']
    counts = updates_per_task(cfg, per_task_steps, delta_t)
    names = task_names or [f"task {i}" for i in range(len(counts))]
    print("[sparsifier] topology updates per task: "
          + ", ".join(f"{n}={c}" for n, c in zip(names, counts)))
    starved = [(n, c) for n, c in zip(names, counts) if c < cfg.min_updates_per_task]
    if not starved:
        return
    # What would give the shortest task enough updates?
    shortest = min(per_task_steps)
    want_delta_t = max(1, int(cfg.t_end_ratio * shortest) // cfg.min_updates_per_task)
    suggested = max(1, int(cfg.t_end_ratio * sum(per_task_steps)) // want_delta_t)
    raise ValueError(
        f"delta_t={delta_t} is too coarse for these tasks: "
        + ", ".join(f"{n} gets {c}" for n, c in starved)
        + f" (need >= {cfg.min_updates_per_task}). delta_t comes from the whole "
          f"run ({sum(per_task_steps):,} steps), but drop_fraction_schedule="
          f"{cfg.drop_fraction_schedule!r} gives the shortest task an update "
          f"window of only {int(cfg.t_end_ratio * shortest)} steps. "
          f"Use --num_mask_updates={suggested} (delta_t~{want_delta_t}), or lower "
          f"--min_updates_per_task to accept this.")


def restart_drop_fraction_schedule(cfg, sparsifier, task_steps):
    """Restart the drop-fraction cosine for a new task (drop_fraction_schedule
    == 'per_task'): the drop fraction warms back up to cfg.pruning_ratio and
    decays over t_end_ratio of this task's steps, so every task -- including
    the last -- gets topology updates. delta_t (the cadence) stays global.

    Returns the new t_end, or None when the schedule is not per-task.
    """
    if sparsifier is None or cfg.drop_fraction_schedule != "per_task":
        return None
    if cfg.sparsifier not in ("rigl", "set"):
        return None
    t_end = max(1, int(cfg.t_end_ratio * task_steps))
    sparsifier._step_count = 0
    sparsifier.scheduler.t_end = t_end
    return t_end


def build_sparsifier(cfg, model, optimizer, total_steps):
    """Create, prepare and return a sparsifier. None when cfg.sparsifier == 'dense'.

    Call after the optimizer exists and before torch.compile.
    """
    if cfg.sparsifier == "dense":
        return None
    if cfg.sparsifier not in ("rigl", "set", "gmp", "static"):
        raise ValueError(f"unknown sparsifier {cfg.sparsifier!r}")

    sched = sparsifier_schedule(cfg, total_steps)

    # --- the two arms that do not rewire ------------------------------------
    # They isolate what RigL/SET's regrowth is worth: `static` fixes the mask
    # after one magnitude prune, `gmp` ramps sparsity up on a cubic schedule
    # and prunes only (its grow_mask is a no-op).
    if cfg.sparsifier == "static":
        sparsifier = StaticMagnitudeSparsifier(
            optimizer=optimizer,
            distribution=DISTRIBUTIONS[cfg.sparse_distribution](),
            sparsity=cfg.sparsity,
            global_pruning=False,
        )
    elif cfg.sparsifier == "gmp":
        # Sparsity ramps 0 -> cfg.sparsity. t_accel is where the cubic starts,
        # as a fraction of t_end; accelerated_sparsity is the level it jumps to
        # there. Kurtic et al.'s GMP*.
        sparsifier = GMP(
            scheduler=AcceleratedCubicScheduler(
                t_end=sched["t_end"], delta_t=sched["delta_t"],
                t_accel=int(cfg.gmp_t_accel_ratio * sched["t_end"]),
                initial_sparsity=0.0,
                accelerated_sparsity=cfg.gmp_accel_sparsity * cfg.sparsity,
                final_sparsity=cfg.sparsity),
            distribution=DISTRIBUTIONS[cfg.sparse_distribution](),
            optimizer=optimizer,
            global_pruning=False,
        )
    else:
        sparsifier = _build_dst(cfg, optimizer, sched)
    return _prepare_sparsifier(cfg, model, sparsifier, sched, total_steps)


def _build_dst(cfg, optimizer, sched):
    """RigL / SET: prune and regrow every delta_t steps."""
    if cfg.drop_fraction_schedule == "constant":
        scheduler = ConstantScheduler(quantity=cfg.pruning_ratio, t_end=sched["t_end"],
                                      delta_t=sched["delta_t"])
    else:
        # 'per_task' starts from the global cosine; train.py retargets t_end at
        # every task boundary via restart_drop_fraction_schedule().
        scheduler = CosineDecayScheduler(quantity=cfg.pruning_ratio, t_end=sched["t_end"],
                                         delta_t=sched["delta_t"])
    cls = RigL if cfg.sparsifier == "rigl" else SET
    return cls(
        scheduler=scheduler,
        distribution=DISTRIBUTIONS[cfg.sparse_distribution](),
        optimizer=optimizer,
        sparsity=cfg.sparsity,
        global_pruning=False,
    )


def _prepare_sparsifier(cfg, model, sparsifier, sched, total_steps):
    """Attach masks to the targeted tensors and report what was covered."""
    target_set = getattr(cfg, "sparse_targets", DEFAULT_TARGETS)
    targets = get_sparse_targets(model, target_set)
    if not targets:
        raise ValueError(
            f"no Linear layers matched sparse_targets={target_set!r} "
            f"({TARGET_SETS[target_set]}) under {BLOCK_PREFIX!r}; is this an HF decoder?")
    # Count before prepare: reparametrization renames weight -> parametrizations.weight.original
    n_params = count_sparse_params(model, targets)
    n_model = sum(p.numel() for p in model.parameters())
    sparsifier.prepare(model, targets)
    # Read by DSTMixin.grow_mask: 'previous' keeps the value a regrown weight
    # already held (its pretrained value if it was never active).
    sparsifier.grow_init = cfg.grow_init
    if cfg.grow_init == "previous" and getattr(cfg, "weight_decay", 0.0) > 0:
        # AdamW's decoupled weight decay shrinks every parameter each step,
        # including masked ones (which get no gradient), so a regrown weight
        # would come back decayed rather than at the value it was pruned at.
        print(f"[sparsifier] WARNING: grow_init='previous' with weight_decay="
              f"{cfg.weight_decay}: masked weights keep decaying while inactive, "
              f"so regrown values are shrunken. Use weight_decay=0 for exact regrowth.")
    # cfg.sparsity is the sparsity *within the targeted tensors*, so narrowing
    # the target set lowers the fraction of the whole model that is pruned.
    # Print both, or a sweep over sparsity is easy to misread across target sets.
    coverage = n_params / n_model if n_model else 0.0
    print(f"[sparsifier] {cfg.sparsifier} sparsity={cfg.sparsity} "
          f"targets={target_set} ({'/'.join(TARGET_SETS[target_set])}) "
          f"tensors={len(targets)} params={n_params:,} "
          f"coverage={coverage:.1%} of {n_model:,} "
          f"effective_global_sparsity={cfg.sparsity * coverage:.2%}")
    print(f"[sparsifier] distribution={cfg.sparse_distribution} "
          f"total_steps={total_steps:,} t_end={sched['t_end']:,} delta_t={sched['delta_t']:,} "
          f"drop_fraction_schedule={cfg.drop_fraction_schedule} grow_init={cfg.grow_init}")
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
    """Turn a reparametrized state dict back into a plain HF one.

    sparsimony stores `X.parametrizations.weight.original` plus a mask buffer at
    `X.parametrizations.weight.0.mask` (and, for RigL, a dense-gradient buffer),
    which a plain model cannot load. This folds the mask into the weight (so the
    saved tensor is genuinely sparse), drops the other parametrization buffers,
    and returns the masks separately, keyed by the weight they belong to.

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
        if key.endswith(".original"):
            prefix, rest = key.split(".parametrizations.", 1)
            tensor_name = rest.split(".", 1)[0]
            weight_key = f"{prefix}.{tensor_name}"
            mask = masks.get(weight_key)
            flat[weight_key] = value * mask if mask is not None else value
    return flat, masks
