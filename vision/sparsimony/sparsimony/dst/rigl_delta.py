from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from sparsimony.dst.rigl import RigL
from sparsimony.parametrization.dfsb import DFSB_RigL
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.distributions.base import BaseDistribution
from sparsimony.utils import get_mask, get_original_tensor

class RigLDelta(RigL):
    """
    RigL-Delta: A Dynamic Sparse Training method that combines:
    1. Dense Forward Pass (from SET-Delta/DFSB)
    2. Gradient-based Growth (from RigL)
    
    It keeps the forward pass dense, but pruning and growing happen on the mask.
    The backward pass is sparse (masked gradients flow to weights), but we also
    accumulate dense gradients to inform the growth step.
    """
    def __init__(
        self,
        scheduler: BaseScheduler,
        distribution: BaseDistribution,
        optimizer: torch.optim.Optimizer,
        defaults: Optional[Dict[str, Any]] = None,
        sparsity: float = 0.5,
        grown_weights_init: float = 0.0,
        init_method: Optional[str] = None,
        low_mem_mode: bool = False,
        *args,
        **kwargs,
    ):
        self._pretrained_weights: Dict[Tuple[int, str], torch.Tensor] = {}
        if defaults is None:
            defaults = dict(parametrization=DFSB_RigL)

        super().__init__(
            scheduler=scheduler,
            distribution=distribution,
            optimizer=optimizer,
            defaults=defaults,
            sparsity=sparsity,
            grown_weights_init=grown_weights_init,
            init_method=init_method,
            low_mem_mode=low_mem_mode,
            *args,
            **kwargs,
        )

    def _initialize_masks(self) -> None:
        for config in self.groups:
            module, tensor_name = config["module"], config["tensor_name"]
            key = (id(module), tensor_name)
            self._pretrained_weights[key] = get_original_tensor(module, tensor_name).detach().clone()
        super()._initialize_masks()

    def update_mask(
        self,
        module: nn.Module,
        tensor_name: str,
        sparsity: float,
        prune_ratio: float,
        dense_grads: torch.Tensor,
        **kwargs,
    ):
        mask = get_mask(module, tensor_name)
        if sparsity == 0:
            mask.data = torch.ones_like(mask)
        else:
            key = (id(module), tensor_name)
            pretrained = self._pretrained_weights[key]
            original_weights = get_original_tensor(module, tensor_name)
            weights = getattr(module, tensor_name)
            delta_weights = weights - pretrained
            target_sparsity = self.get_sparsity_from_prune_ratio(mask, prune_ratio)
            self.prune_mask(target_sparsity, mask, values=delta_weights)
            self.grow_mask(sparsity, mask, original_weights, values=dense_grads)
            self._assert_sparsity_level(mask, sparsity)
