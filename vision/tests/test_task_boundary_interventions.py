"""Tests for grow_init, FIRE (fire_sparse) and full_reset.

Everything runs on CPU with tiny models.
"""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


class TinyAttnNet(nn.Module):
    """Small model whose Linear layers use the TinyViT names to_q / to_k."""

    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(16, 16, bias=False)
        self.attn.to_k = nn.Linear(16, 16, bias=False)
        self.attn.to_v = nn.Linear(16, 16, bias=False)
        self.fc = nn.Linear(16, 32)
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        q, k, v = self.attn.to_q(x), self.attn.to_k(x), self.attn.to_v(x)
        return self.head(torch.relu(self.fc(q * k + v)))


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).mean(dim=(2, 3)))


def _cfg(**overrides):
    from config_st import Config
    d = {
        'optimizer': 'adam', 'lr': 1e-2, 'clip_grad_norm': 0.5,
        'task': 'CIFAR100', 'model': 'TinyViT', 'benchmark': 'continual',
        'seed': 0, 'sparsifier': 'rigl', 'sparsity': 0.5,
        'num_mask_updates': 20, 't_end_ratio': 0.8, 'pruning_ratio': 0.3,
        'drop_fraction_schedule': 'global', 't_accel_ratio': 0.2,
        'initial_sparsity': 0.0, 'grow_init': 'zero',
    }
    d.update(overrides)
    return Config(d)


def _sparse_model(**overrides):
    from train_st import build_sparsifier
    torch.manual_seed(0)
    model = TinyAttnNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sp = build_sparsifier(_cfg(**overrides), model, opt, chunk_steps=[40])
    return model, opt, sp


def _sparse_layers(model):
    """(name, trainable weight, mask) for every sparsified layer."""
    out = []
    for name, m in model.named_modules():
        if parametrize.is_parametrized(m, "weight"):
            mask = [p.mask for p in m.parametrizations.weight if hasattr(p, "mask")][0]
            out.append((name, m.parametrizations.weight.original, mask))
    return out


def _train_step(model, opt):
    x, y = torch.randn(8, 16), torch.randint(0, 4, (8,))
    opt.zero_grad()
    nn.CrossEntropyLoss()(model(x), y).backward()
    opt.step()


def _run_until_mask_update(model, opt, sp):
    """Train until RigL changes the masks. Returns the weights and masks as they
    were just before that update."""
    for _ in range(40):
        _train_step(model, opt)
        before = {n: (w.detach().clone(), m.clone()) for n, w, m in _sparse_layers(model)}
        if sp.step():
            return before
    raise AssertionError("no mask update happened")


# ---------------------------------------------------------------------------
# grow_init
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grow_init", ["zero", "previous"])
def test_grow_init(grow_init):
    model, opt, sp = _sparse_model(grow_init=grow_init)
    assert sp.grow_init == grow_init
    before = _run_until_mask_update(model, opt, sp)

    n_grown = 0
    for name, w, mask in _sparse_layers(model):
        w_before, mask_before = before[name]
        grown = mask & ~mask_before
        n_grown += int(grown.sum())
        if grow_init == "zero":
            # Grown weights are set to 0. RigL prunes before it grows, so a
            # weight pruned and regrown in the same update is also set to 0.
            changed = w != w_before
            assert (w[changed] == 0).all() and mask[changed].all()
            assert (w[grown] == 0).all()
        else:
            # A mask update does not touch the stored weights at all.
            assert torch.equal(w, w_before)
    assert n_grown > 0


def test_previous_values_are_not_zero():
    model, opt, sp = _sparse_model(grow_init="previous")
    grown_values = []
    before = _run_until_mask_update(model, opt, sp)
    for name, w, mask in _sparse_layers(model):
        grown_values.append(w[mask & ~before[name][1]])
    assert torch.cat(grown_values).abs().sum() > 0


def test_unknown_grow_init_raises():
    model, opt, sp = _sparse_model()
    sp.grow_init = "something_else"
    with pytest.raises(ValueError):
        _run_until_mask_update(model, opt, sp)


# ---------------------------------------------------------------------------
# FIRE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("net, is_vit", [(TinyAttnNet, True), (TinyConvNet, False)])
def test_fire_sparse_matches_fire_on_dense_models(net, is_vit):
    from interventions.fire import fire
    from interventions.fire_sparse import fire_sparse
    torch.manual_seed(0)
    a = net()
    b = copy.deepcopy(a)
    fire(a, iteration=10, is_vit=is_vit)
    n = fire_sparse(b, iteration=10, is_vit=is_vit)
    assert n == 2
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), name


def test_fire_sparse_on_sparse_model():
    from interventions.fire import fire
    from interventions.fire_sparse import fire_sparse
    model, opt, sp = _sparse_model()
    for _ in range(3):
        _train_step(model, opt)
    before = {n: (w.detach().clone(), m.clone()) for n, w, m in _sparse_layers(model)}

    assert fire_sparse(model, iteration=10, is_vit=True) == 2

    for name, w, mask in _sparse_layers(model):
        w_before, mask_before = before[name]
        assert torch.equal(mask, mask_before)
        if name.split(".")[-1] in ("to_q", "to_k"):
            # Independent check: run the original fire() on a dense layer that
            # holds mask * weight, then undo the density scaling.
            ref = nn.Module()
            ref.to_q = nn.Linear(w.shape[1], w.shape[0], bias=False)
            ref.to_q.weight.data = w_before * mask_before
            fire(ref, iteration=10, is_vit=True)
            density = mask_before.float().mean()
            assert torch.allclose(w, ref.to_q.weight / density.sqrt())
        else:
            assert torch.equal(w, w_before)


# ---------------------------------------------------------------------------
# full_reset
# ---------------------------------------------------------------------------

def test_full_reset_restores_weights_and_keeps_masks():
    from train_st import full_reset
    model, opt, sp = _sparse_model()
    init_model = copy.deepcopy(model)
    _run_until_mask_update(model, opt, sp)
    learned_masks = {n: m.clone() for n, _, m in _sparse_layers(model)}
    init_masks = {n: m.clone() for n, _, m in _sparse_layers(init_model)}
    assert any(not torch.equal(learned_masks[n], init_masks[n]) for n in learned_masks)

    full_reset(model, init_model)

    for (name, p), (_, p0) in zip(model.named_parameters(), init_model.named_parameters()):
        assert torch.equal(p, p0), name
    for name, _, mask in _sparse_layers(model):
        assert torch.equal(mask, learned_masks[name])
