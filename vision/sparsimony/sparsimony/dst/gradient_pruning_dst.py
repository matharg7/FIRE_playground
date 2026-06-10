from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from sparsimony.dst.rigl_delta import RigLDelta
from sparsimony.utils import get_mask

class GradientPruningDST(RigLDelta):
    """
    GradientPruningDST: A Dynamic Sparse Training method that behaves like RigL-Delta
    but prunes based on gradient magnitude instead of weight magnitude.
    
    1. Dense Forward Pass (from SET-Delta/DFSB)
    2. Gradient-based Growth (from RigL)
    3. Gradient-based Pruning (New)
    """
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
            original_weights = getattr(
                module.parametrizations, tensor_name
            ).original
            
            target_sparsity = self.get_sparsity_from_prune_ratio(
                mask, prune_ratio
            )
            # Prune based on dense_grads instead of weights
            self.prune_mask(target_sparsity, mask, values=dense_grads)
            self.grow_mask(sparsity, mask, original_weights, values=dense_grads)
            self._assert_sparsity_level(mask, sparsity)
