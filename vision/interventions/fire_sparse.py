"""FIRE for models with sparse (masked) weights.

fire() in fire.py replaces each selected weight matrix with the nearest
orthogonal matrix (Newton-Schulz), scaled by sqrt(d_out / d_in). It cannot be
used on a sparsified layer: there `module.weight` is recomputed as
mask * original on every access, so writing to it has no effect.

fire_sparse() orthogonalizes the matrix the layer actually uses
(mask * original), writes the result to `original` and keeps the mask. On a
dense model it gives the same weights as fire().
"""

import numpy as np
import torch
from torch import nn
import torch.nn.utils.parametrize as parametrize

from .fire import newton_schulz


def _weight_and_mask(module):
    """The trainable weight tensor of a module and its mask (None if dense)."""
    if not parametrize.is_parametrized(module, "weight"):
        return module.weight, None
    mask = None
    for p in module.parametrizations.weight:
        if hasattr(p, "mask"):
            mask = p.mask
    return module.parametrizations.weight.original, mask


def _newton_schulz(mat, iters):
    # An all-zero slice would divide by zero inside newton_schulz; leave it as is.
    if not torch.isfinite(mat).all() or mat.norm() == 0:
        return mat.clone()
    return newton_schulz(mat, num_iters=iters)


def _orthogonalize(w, iters):
    """Same steps and scale as fire(), for a 2-D or 4-D weight."""
    if w.ndim == 4:
        out = torch.zeros_like(w)
        for i in range(w.shape[2]):
            for j in range(w.shape[3]):
                out[:, :, i, j] = _newton_schulz(w[:, :, i, j], iters)
        kernel_size = w.shape[2] * w.shape[3]
    else:
        out = _newton_schulz(w, iters)
        kernel_size = 1.0
    return out * (np.sqrt(w.shape[0] / w.shape[1]) / kernel_size)


@torch.no_grad()
def fire_sparse(model, iteration=10, is_vit=False):
    """Apply FIRE in place and return the number of weight matrices changed.

    Selects the same layers as fire(): every Linear / Conv2d, or only to_q and
    to_k when is_vit is True.
    """
    changed = 0
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        if is_vit and name.split(".")[-1] not in ("to_q", "to_k"):
            continue
        weight, mask = _weight_and_mask(module)
        if mask is None:
            new = _orthogonalize(weight.data.clone(), iteration)
        else:
            new = _orthogonalize(weight.data * mask, iteration)
            # The mask keeps only a fraction `density` of the orthogonal
            # matrix, which shrinks the layer's output by about
            # sqrt(density). Scale back up to the gain fire() intends.
            density = mask.float().mean().clamp_min(1e-8)
            new = new / density.sqrt()
        weight.data.copy_(new)
        changed += 1
    return changed
