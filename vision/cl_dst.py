"""CL-DST parameter isolation (cl-dst/) driven by train_st.py's chunk loop.

Everything method-specific is cl-dst's own, imported unmodified from cl-dst/:
``Masking`` and ``DeathCosineDecay`` from core.py, ``set_optimizer`` from
utils.py.  This module only sequences them the way cl-dst/cifar100.py sequences
them over tasks, with a chunk (``i_iter``) playing the role of a task: fresh
optimizer, LR schedule, drop-fraction cosine and mask per chunk, with
``used_params`` accumulating across chunks.
"""

import contextlib
import importlib.util
import os
import sys
import types

import torch.nn as nn

_CL_DST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cl-dst')


def _load(name, filename, stubs):
    """Import a cl-dst module by path, without putting cl-dst/ on sys.path.

    cl-dst/utils.py opens with ``from models import ResNet18, ...``, so cl-dst/
    on sys.path would either shadow vision/models (breaking train_st.py) or fail
    to import.  ``stubs`` stands in for what the module imports but we never
    reach, and is removed again once the module is loaded.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_CL_DST_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for key, previous in saved.items():
            if previous is None:
                del sys.modules[key]
            else:
                sys.modules[key] = previous
    return module


# core.py's only cl-dst import is print_and_log, used exclusively in its
# train()/evaluate(), which we never call.
_utils_stub = types.ModuleType('utils')
_utils_stub.print_and_log = print
_core = _load('cl_dst_core', 'core.py', {'utils': _utils_stub})

# set_optimizer never builds a model, so the network classes need only exist.
_models_stub = types.ModuleType('models')
for _name in ('ResNet18', 'LeNet5', 'MobileNetV2', 'VGG16', 'MobileNetFunc'):
    setattr(_models_stub, _name, None)
_cl_utils = _load('cl_dst_utils', 'utils.py', {'models': _models_stub})

Masking = _core.Masking
DeathCosineDecay = _core.DeathCosineDecay
set_optimizer = _cl_utils.set_optimizer


class _Sink:
    """Swallows cl-dst's per-layer prints without allocating."""

    def write(self, *_args):
        pass

    def flush(self):
        pass


_SINK = _Sink()


def _quiet():
    # print_nonzero_counts() and fired_masks_update() print one line per layer on
    # every rewiring, which would bury the run log.  tqdm writes to stderr.
    return contextlib.redirect_stdout(_SINK)


# cfg.sparsifier -> cl-dst's (--init, --death, --growth); see METHOD.md section 3.
# 'set' reproduces cl-dst's shipped defaults.
_METHODS = {
    'rigl': ('ERK',     'magnitude', 'gradient'),
    'set':  ('uniform', 'magnitude', 'random'),
}


class CLDST:
    """Sparsity + DST + gradient isolation over train_st.py's chunks."""

    def __init__(self, model, cfg, chunk_steps, device):
        if cfg.sparsifier not in _METHODS:
            raise ValueError(
                f"use_cl_dst requires sparsifier in {sorted(_METHODS)}, got "
                f"'{cfg.sparsifier}'. CL-DST isolates a mask that DST keeps "
                "rewiring; 'dense' and 'static' have no such mask, and 'gmp' "
                "has no working call site in cl-dst."
            )

        self.model = model
        self.cfg = cfg
        self.device = device
        self.chunk_steps = chunk_steps
        self.density = 1 - cfg.sparsity
        self.init, self.death, self.growth = _METHODS[cfg.sparsifier]

        # Same delta_t as build_sparsifier(), so the rewiring cadence matches the
        # plain rigl/set runs.
        total_steps = sum(chunk_steps)
        t_end = int(cfg.t_end_ratio * total_steps)
        self.delta_t = max(1, t_end // cfg.num_mask_updates)

        self.mask = None
        self.lr_scheduler = None
        self.used_params = {}
        self._superposition = None
        # add_module() drops biases and BatchNorm from the mask but not
        # LayerNorm, so TinyViT's gains would be masked at cfg.sparsity.  Keep
        # every norm outside the mask, as cl-dst intends (METHOD.md 2.1).
        self._ln_params = {
            f"{fqn}.{name}"
            for fqn, module in model.named_modules() if isinstance(module, nn.LayerNorm)
            for name, _ in module.named_parameters(recurse=False)
        }

        # Both of cl-dst's schedules restart every task and neither is
        # configurable, so the matching train_st.py knobs do not apply.
        if cfg.drop_fraction_schedule != 'per_task':
            print(f"[CL-DST] drop_fraction_schedule="
                  f"'{cfg.drop_fraction_schedule}' ignored: cl-dst restarts the "
                  "drop-fraction cosine every chunk.")
        if cfg.get('use_cosine_lr', False):
            print("[CL-DST] use_cosine_lr ignored: cl-dst uses MultiStepLR "
                  "(x0.1 at 1/2 and 3/4 of each chunk's epochs).")

        print(f"[CL-DST] {cfg.sparsifier} -> init={self.init} death={self.death} "
              f"growth={self.growth} | density={self.density} | "
              f"delta_t={self.delta_t} | death_rate={cfg.pruning_ratio} | "
              f"lr={cfg.lr} MultiStepLR(1/2, 3/4, gamma=0.1) | "
              f"drop fraction: cosine restarted per chunk")

    def start_chunk(self, i_iter, real_epochs):
        """Rebuild both schedules and resample the mask (cifar100.py:84-95).

        Returns the optimizer cl-dst's set_optimizer built; its MultiStepLR is
        held here and advanced by lr_step().
        """
        if self.mask is not None:
            # U <- U + M_{i-1}, the expression from utils.freeze_used_params but
            # reading the previous mask from memory instead of ./masks/*.pt.
            self.used_params = {
                k: self.used_params.get(k, 0) + self.mask.masks.get(k, 0)
                for k in set(self.used_params) | set(self.mask.masks)
            }

        # cl-dst builds the optimizer and its MultiStepLR together, per task,
        # with milestones at 1/2 and 3/4 of that task's epochs (utils.py:364).
        optimizer, self.lr_scheduler = set_optimizer(
            self.cfg.optimizer, self.cfg.lr, self.cfg.get('momentum', 0.9),
            self.cfg.get('l2', 0), real_epochs, self.model)
        # One cosine half-period per chunk, over its batches, to eta_min=0.005.
        decay = DeathCosineDecay(self.cfg.pruning_ratio, self.chunk_steps[i_iter])
        self.mask = Masking(
            self.init, self.density, self.death, self.growth,
            self.cfg.pruning_ratio, real_epochs, self.delta_t,
            1.0, decay, optimizer, self.device)

        backup = ({n: p.detach().clone() for n, p in self.model.named_parameters()}
                  if self._ln_params else None)
        with _quiet():
            self.mask.add_module(self.model)
            if backup is not None:
                # add_module's init() has already masked the LayerNorms, so drop
                # them, restore every weight, and sample once more over what is
                # left.  Restoring first matters: without it the weights would be
                # masked twice and most of the new mask would start at zero.
                self.mask.remove_type(nn.LayerNorm)
                for name, param in self.model.named_parameters():
                    param.data.copy_(backup[name])
                self.mask.init(self.init, self.density)
        return optimizer

    def lr_step(self):
        """cl-dst steps MultiStepLR after each epoch (cifar100.py:103)."""
        self.lr_scheduler.step()

    def freeze_grads(self):
        """Zero the gradient on every weight an earlier chunk claimed (core.py:576-579).

        Called between backward() and the optimizer step.  It is also what lets
        'gradient' growth respect isolation for free: truncate_weights() reads
        weight.grad, which is exactly zero on U by then (METHOD.md 2.14).
        """
        if not self.used_params:
            return
        for name, params in self.model.named_parameters():
            if name in self.mask.masks and params.grad is not None:
                params.grad[self.used_params[name] != 0] = 0

    def step(self):
        """Masking.step() subsumes optimizer.step() (core.py:202-213)."""
        with _quiet():
            self.mask.step()

    def end_chunk(self):
        """Restore the weights this chunk's mask zeroed (cifar100.py:110-117).

        apply_mask() multiplies every masked weight by the current mask, so a
        weight an earlier chunk trained but this chunk's mask excludes is
        genuinely destroyed during the chunk; only this merge brings it back.
        """
        params = dict(self.model.named_parameters())
        if self._superposition is not None:
            for name, param in params.items():
                param.data[param == 0] = self._superposition[name][param == 0]
        self._superposition = {n: p.detach().clone() for n, p in params.items()}

    def stats(self):
        """Metrics under the keys train_st.py already logs, plus capacity ones."""
        masks = self.mask.masks
        params = dict(self.model.named_parameters())
        total = sum(m.numel() for m in masks.values())
        active = sum(int(m.count_nonzero()) for m in masks.values())
        nonzero = sum(int(params[n].count_nonzero()) for n in masks)
        fired = sum(int(m.count_nonzero()) for m in self.mask.fired_masks.values())
        # Of this chunk's own mask, the share already frozen by an earlier chunk.
        frozen = sum(int(((self.used_params[n] != 0) & (masks[n] != 0)).sum())
                     for n in masks if n in self.used_params)
        claimed = sum(int((self.used_params[n] != 0).sum())
                      for n in masks if n in self.used_params)
        return {
            'dst/mask_sparsity': 1 - active / total,
            'dst/weight_sparsity': 1 - nonzero / total,
            'dst/itop_rate': fired / total,
            'dst/pruning_ratio': self.mask.death_rate,
            'cl_dst/claimed': claimed / total,
            'cl_dst/frozen_active': frozen / active,
            'cl_dst/trainable': 1 - frozen / active,
        }
