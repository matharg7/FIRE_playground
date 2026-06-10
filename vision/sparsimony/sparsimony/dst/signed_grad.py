from typing import Optional, Dict, Any

import torch
import torch.optim

from sparsimony.dst.rigl_delta import RigLDelta, DFSB_RigL
from sparsimony.schedulers.base import BaseScheduler
from sparsimony.distributions.base import BaseDistribution
from sparsimony.mask_calculators.unstructured import UnstructuredPruner
from sparsimony.mask_calculators.scorers import MagnitudeScorer, IdentityScorer


class SignedGradDST(RigLDelta):
    """Signed Gradient Dynamic Sparse Training.

    This method:
    1. Initializes with a dense mask (all ones).
    2. Prunes connections based on the signed value of sign(W) * Grad.
       (Prunes the most negative values).
    3. Grows connections based on gradient magnitude (standard RigL).
    4. Uses DFSB parametrization for dense forward / sparse backward.
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
        # Override pruner to use IdentityScorer for signed values
        self.pruner = UnstructuredPruner(
            scorer=IdentityScorer(), low_mem_mode=low_mem_mode
        )

    @torch.no_grad()
    def _initialize_masks(self):
        """Initializes masks to all ones (Dense)."""
        self._distribute_sparsity(self.sparsity)
        for config in self.groups:
            # Set mask to all ones
            mask = config["module"].parametrizations[config["tensor_name"]][0].mask
            if hasattr(mask, "is_leaf"):  # Check if it's a leaf tensor for safety
                pass
            mask.fill_(1.0)
            
            # Reset dense grad buffers to zero just in case
            if hasattr(config["module"].parametrizations[config["tensor_name"]][0], "dense_grad"):
                 config["module"].parametrizations[config["tensor_name"]][0].dense_grad.zero_()

    @torch.no_grad()
    def update_mask(
        self,
        module,
        tensor_name,
        sparsity,
        prune_ratio,
        dense_grads,
        **kwargs
    ):
        """Update the mask for a single layer.
        
        Pruning: sign(W) * Grad (IdentityScorer -> remove most negative)
        Growth: |Grad| (MagnitudeScorer -> add largest absolute)
        """
        mask = module.parametrizations[tensor_name][0].mask
        
        # 1. Calculate Pruning Score
        # Score = sign(Weight) * DenseGrad
        original_weights = getattr(module.parametrizations[tensor_name], "original")
        
        pruning_score = torch.sign(original_weights) * dense_grads
        
        # 2. Prune
        # Calculate target sparsity for this step (we prune down to this)
        # e.g. if we are at 50% sparsity and want to prune 10%, we prune down to 60% sparsity
        # (wait, RigL logic: prune_ratio is fraction of active weights to prune.
        # current sparsity is self.sparsity (final target). 
        # But RigL uses get_sparsity_from_prune_ratio to find how many to drop.
        
        target_sparsity = self.get_sparsity_from_prune_ratio(mask, prune_ratio)
        
        # Prune using IdentityScorer (prunes smallest algebraic values, i.e. most negative)
        # We call self.pruner directly or use self.prune_mask?
        # self.prune_mask calls self.pruner.calculate_mask.
        # Let's call self.pruner.calculate_mask directly to be explicit about 'values'.
        
        # Actually self.prune_mask is strictly better if it handles other things?
        # But RigL calls self.prune_mask(..., values=weights).
        # We will call self.prune_mask(..., values=pruning_score).
        
        self.prune_mask(
            target_sparsity,
            mask,
            values=pruning_score
        )
        
        # 3. Grow
        # Grow back to 'sparsity' (the target sparsity for the layer) using dense_grads.
        # RigL grows active weights based on magnitude of gradient.
        # self.grower uses MagnitudeScorer (default for RigL).
        
        self.grow_mask(
            sparsity,
            mask,
            values=dense_grads,
            original_weights=original_weights
        )
        
        # Ensure sparsity level is correct
        self._assert_sparsity_level(mask, sparsity)
