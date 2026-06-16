import sys
import os
import torch
import torch.nn as nn

_sparsimony_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparsimony')
if os.path.isdir(_sparsimony_path) and _sparsimony_path not in sys.path:
    sys.path.insert(0, _sparsimony_path)

from sparsimony.utils import get_mask


def get_sparsity_stats(model: nn.Module) -> dict:
    """
    Compute mask sparsity and effective weight sparsity for all
    sparsimony-reparametrized layers.

    mask_sparsity:   fraction of weights zeroed by the boolean mask.
    weight_sparsity: fraction of weights with zero value after FakeSparsity
                     multiplies the mask into the weight tensor.
    """
    total_params = 0
    total_zero_mask = 0
    total_zero_weight = 0
    num_sparse_modules = 0

    for _, module in model.named_modules():
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            num_sparse_modules += 1
            with torch.no_grad():
                mask = get_mask(module, "weight")
                weight = module.parametrizations.weight.original
                n = weight.numel()
                total_params += n
                total_zero_mask += n - mask.sum().item()
                total_zero_weight += n - torch.count_nonzero(module.weight).item()

    if total_params == 0:
        return {"mask_sparsity": 0.0, "weight_sparsity": 0.0, "num_sparse_modules": 0}

    return {
        "mask_sparsity": total_zero_mask / total_params,
        "weight_sparsity": total_zero_weight / total_params,
        "num_sparse_modules": num_sparse_modules,
    }


def get_current_pruning_ratio(sparsifier) -> float | None:
    """
    Return the scheduler's output at the most recent mask-update step.

    For RigL / SET  (CosineDecay / Constant scheduler): the fraction of
    active weights pruned per update — decays from pruning_ratio toward 0.
    For GMP (AcceleratedCubic scheduler): the target sparsity level at the
    current step (interpretation differs from RigL/SET).
    Returns None if the sparsifier has no scheduler with delta_t (e.g.
    static), or if all updates are past t_end.
    """
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
    """
    Tracks the ITOP (In-Time Over Parameters) rate: the fraction of total
    parameters that have been active (mask == True) at any point during
    training.

        ITOP = |union_{t} M_t| / N

    Initialise after sparsifier.prepare(); call update() whenever
    sparsifier.step() returns True (i.e. a mask topology change occurred).
    """

    def __init__(self, sparsifier):
        self._sparsifier = sparsifier
        self._union_masks: list[torch.Tensor] = []
        self._total_params: int = 0
        for config in sparsifier.groups:
            mask = get_mask(config["module"], config["tensor_name"])
            self._union_masks.append(mask.detach().cpu().clone())
            self._total_params += mask.numel()

    def update(self) -> None:
        """OR current masks into the cumulative union."""
        for i, config in enumerate(self._sparsifier.groups):
            mask = get_mask(config["module"], config["tensor_name"])
            self._union_masks[i] = self._union_masks[i] | mask.detach().cpu()

    def compute(self) -> float:
        """Return the current ITOP rate in [0, 1]."""
        if self._total_params == 0:
            return 0.0
        return sum(m.sum().item() for m in self._union_masks) / self._total_params
