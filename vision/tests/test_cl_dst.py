"""Integration tests for the CL-DST baseline (vision/cl_dst.py).

Coverage:
  1.  Import shim - core.py loads, vision/models is untouched
  2.  Config validation - method mapping; the LR and drop-fraction knobs are ignored
  3.  Mask resampling - fresh mask per chunk, at the target density
  4.  used_params - monotone, equals the sum of past masks
  5.  freeze_grads - gradients exactly zero on U
  6.  Isolation - a claimed weight never changes again
  7.  Merge - weights the current mask excludes survive the chunk
  8.  Growth - 'gradient' regrows into frozen slots less than 'random'
  9.  LayerNorm - outside the mask and left intact
  10. stats() - keys, ranges, capacity decay
  11. LR schedule (cl-dst MultiStepLR) and end-to-end mini loop over chunks
"""

import math
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tiny synthetic models (CPU-only, no dataset download)
# ---------------------------------------------------------------------------

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.fc(self.pool(x).flatten(1))


class TinyNormNet(nn.Module):
    """Has a LayerNorm, like TinyViT."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.norm(torch.relu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device('cpu')
CHUNK_STEPS = [100, 100, 100, 100]
EPOCHS = 20          # per chunk; MultiStepLR milestones land at 10 and 15


def _make_cfg(**overrides):
    from config_st import Config
    defaults = {
        'sparsifier': 'set', 'use_cl_dst': True,
        'sparsity': 0.5, 'pruning_ratio': 0.3,
        'num_mask_updates': 16, 't_end_ratio': 0.8,
        'drop_fraction_schedule': 'per_task',
        'n_epochs': 1, 'lr': 1e-2, 'optimizer': 'adam',
        'model': 'RESNET18', 'task': 'CIFAR10',
    }
    defaults.update(overrides)
    return Config(defaults)


def _build(model=None, epochs=EPOCHS, **overrides):
    """Return (cl_dst, model, optimizer) with the first chunk started."""
    from cl_dst import CLDST
    torch.manual_seed(0)
    model = TinyNet() if model is None else model
    cfg = _make_cfg(**overrides)
    cl = CLDST(model, cfg, CHUNK_STEPS, DEVICE)
    return cl, model, cl.start_chunk(0, epochs)


def _batch(model):
    if isinstance(model, TinyNormNet):
        return torch.randn(4, 16), torch.randint(0, 4, (4,))
    return torch.randn(4, 3, 8, 8), torch.randint(0, 4, (4,))


def _do_step(cl, model, optimizer):
    """One training step, in train_st.py's order."""
    model.train()
    x, y = _batch(model)
    optimizer.zero_grad()
    nn.CrossEntropyLoss()(model(x), y).backward()
    cl.freeze_grads()
    cl.step()


def _run_chunk(cl, model, i_iter, n_steps=25):
    optimizer = cl.start_chunk(i_iter, EPOCHS)
    for _ in range(n_steps):
        _do_step(cl, model, optimizer)
    cl.end_chunk()


def _snapshot(d):
    return {k: v.detach().clone() for k, v in d.items()}


# ---------------------------------------------------------------------------
# 1. Import shim
# ---------------------------------------------------------------------------

class TestImportShim:
    def test_core_objects_loaded(self):
        import cl_dst
        assert cl_dst.Masking.__module__ == 'cl_dst_core'
        assert hasattr(cl_dst.DeathCosineDecay, 'get_dr')

    def test_vision_models_not_shadowed(self):
        import cl_dst  # noqa: F401
        import models
        # cl-dst/models.py has ResNet18; vision/models is the package we need.
        assert hasattr(models, 'get_resnet18_CIFAR10')

    def test_utils_stub_removed(self):
        import sys
        import cl_dst  # noqa: F401
        stub = sys.modules.get('utils')
        assert stub is None or hasattr(stub, 'set_logger')


# ---------------------------------------------------------------------------
# 2. Config validation
# ---------------------------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize('sparsifier', ['dense', 'gmp', 'static'])
    def test_rejects_sparsifier(self, sparsifier):
        from cl_dst import CLDST
        with pytest.raises(ValueError, match='sparsifier'):
            CLDST(TinyNet(), _make_cfg(sparsifier=sparsifier), CHUNK_STEPS, DEVICE)

    @pytest.mark.parametrize('schedule', ['global', 'constant'])
    def test_schedule_knob_ignored_not_rejected(self, schedule, capsys):
        cl, _, _ = _build(drop_fraction_schedule=schedule)
        assert 'ignored' in capsys.readouterr().out
        # Still cl-dst's own per-chunk cosine, whatever the flag said.
        assert cl.mask.death_rate_decay.cosine_stepper.T_max == CHUNK_STEPS[0]

    def test_cosine_lr_knob_ignored(self, capsys):
        _build(use_cosine_lr=True)
        assert 'use_cosine_lr ignored' in capsys.readouterr().out

    @pytest.mark.parametrize('sparsifier,init,growth', [
        ('rigl', 'ERK', 'gradient'),
        ('set', 'uniform', 'random'),
    ])
    def test_method_mapping(self, sparsifier, init, growth):
        cl, _, _ = _build(sparsifier=sparsifier)
        assert (cl.init, cl.death, cl.growth) == (init, 'magnitude', growth)

    def test_delta_t_matches_build_sparsifier(self):
        cl, _, _ = _build()
        cfg = cl.cfg
        expected = max(1, int(cfg.t_end_ratio * sum(CHUNK_STEPS)) // cfg.num_mask_updates)
        assert cl.delta_t == expected

    def test_build_sparsifier_returns_none(self):
        from train_st import build_sparsifier
        cfg = _make_cfg()
        assert build_sparsifier(cfg, TinyNet(), None, CHUNK_STEPS) is None

    def test_run_name_is_distinct(self):
        from train_st import build_run_name
        cfg = _make_cfg()
        assert build_run_name(cfg, None).count('cldst') == 1
        cfg.use_cl_dst = False
        assert 'cldst' not in build_run_name(cfg, None)


# ---------------------------------------------------------------------------
# 3. Mask resampling
# ---------------------------------------------------------------------------

class TestMask:
    def test_biases_and_batchnorm_excluded(self):
        cl, _, _ = _build()
        assert set(cl.mask.masks) == {'conv.weight', 'fc.weight'}

    def test_density_on_target(self):
        cl, _, _ = _build(sparsity=0.5)
        for name, mask in cl.mask.masks.items():
            # init samples Bernoulli(density), so allow sampling noise.
            assert mask.mean().item() == pytest.approx(0.5, abs=0.12), name

    def test_weights_masked_immediately(self):
        cl, model, _ = _build()
        for name, mask in cl.mask.masks.items():
            weight = dict(model.named_parameters())[name]
            assert torch.all(weight[mask == 0] == 0)

    def test_mask_resampled_each_chunk(self):
        cl, model, _ = _build()
        first = _snapshot(cl.mask.masks)
        _run_chunk(cl, model, 1)
        assert any(not torch.equal(first[k], cl.mask.masks[k]) for k in first)


# ---------------------------------------------------------------------------
# 4. used_params
# ---------------------------------------------------------------------------

class TestUsedParams:
    def test_empty_on_first_chunk(self):
        cl, _, _ = _build()
        assert cl.used_params == {}

    def test_equals_sum_of_past_masks(self):
        cl, model, optimizer = _build()
        masks = [_snapshot(cl.mask.masks)]
        for _ in range(4):
            _do_step(cl, model, optimizer)
        cl.end_chunk()
        for i in (1, 2):
            masks.append(None)
            _run_chunk(cl, model, i, n_steps=4)
            masks[i] = _snapshot(cl.mask.masks)
        # After start_chunk(2), U is the sum of chunk 0's and chunk 1's masks.
        cl2 = cl.used_params
        for name in cl.mask.masks:
            assert torch.equal(cl2[name], masks[0][name] + masks[1][name])

    def test_monotone(self):
        cl, model, _ = _build()
        _run_chunk(cl, model, 1, n_steps=4)
        after_one = _snapshot(cl.used_params)
        _run_chunk(cl, model, 2, n_steps=4)
        for name, used in after_one.items():
            assert torch.all(cl.used_params[name] >= used)


# ---------------------------------------------------------------------------
# 5. freeze_grads
# ---------------------------------------------------------------------------

class TestFreezeGrads:
    def test_grads_zero_on_used(self):
        cl, model, optimizer = _build()
        for _ in range(4):
            _do_step(cl, model, optimizer)
        cl.end_chunk()

        optimizer = cl.start_chunk(1, EPOCHS)
        x, y = _batch(model)
        optimizer.zero_grad()
        nn.CrossEntropyLoss()(model(x), y).backward()
        assert any(p.grad[cl.used_params[n] != 0].abs().sum() > 0
                   for n, p in model.named_parameters() if n in cl.mask.masks)
        cl.freeze_grads()
        for name, param in model.named_parameters():
            if name in cl.mask.masks:
                assert torch.all(param.grad[cl.used_params[name] != 0] == 0), name

    def test_noop_on_first_chunk(self):
        cl, model, optimizer = _build()
        x, y = _batch(model)
        optimizer.zero_grad()
        nn.CrossEntropyLoss()(model(x), y).backward()
        before = {n: p.grad.clone() for n, p in model.named_parameters()}
        cl.freeze_grads()
        for name, param in model.named_parameters():
            assert torch.equal(before[name], param.grad)


# ---------------------------------------------------------------------------
# 6/7. Isolation and the merge
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_claimed_weights_never_change(self):
        """The guarantee METHOD.md section 4 says BWT == 0 cannot evidence."""
        cl, model, optimizer = _build()
        for _ in range(10):
            _do_step(cl, model, optimizer)
        cl.end_chunk()
        claimed = _snapshot(cl.mask.masks)
        frozen = {n: p.detach().clone() for n, p in model.named_parameters()}

        for i in (1, 2):
            _run_chunk(cl, model, i, n_steps=10)
            for name, param in model.named_parameters():
                if name in claimed:
                    keep = claimed[name] != 0
                    assert torch.equal(param.data[keep], frozen[name][keep]), \
                        f'{name} changed during chunk {i}'

    def test_merge_restores_shelved_weights(self):
        """A weight in M_0 but not M_1 is zero during chunk 1 and back after it."""
        cl, model, optimizer = _build()
        for _ in range(10):
            _do_step(cl, model, optimizer)
        cl.end_chunk()
        before = dict(model.named_parameters())['conv.weight'].detach().clone()
        m0 = cl.mask.masks['conv.weight'].clone()

        optimizer = cl.start_chunk(1, EPOCHS)
        shelved = (m0 != 0) & (cl.mask.masks['conv.weight'] == 0) & (before != 0)
        assert shelved.sum() > 0, 'no shelved weights to test'
        weight = dict(model.named_parameters())['conv.weight']
        assert torch.all(weight.data[shelved] == 0)

        for _ in range(10):
            _do_step(cl, model, optimizer)
        cl.end_chunk()
        assert torch.equal(weight.data[shelved], before[shelved])


# ---------------------------------------------------------------------------
# 8. Growth criterion
# ---------------------------------------------------------------------------

class TestGrowth:
    def _grown_into_frozen(self, sparsifier):
        """Count positions a rewiring grows that an earlier chunk already froze."""
        cl, model, optimizer = _build(sparsifier=sparsifier, num_mask_updates=80)
        for _ in range(12):
            _do_step(cl, model, optimizer)
        cl.end_chunk()

        optimizer = cl.start_chunk(1, EPOCHS)
        grown = 0
        for _ in range(30):
            before = _snapshot(cl.mask.masks)
            _do_step(cl, model, optimizer)
            for name, mask in cl.mask.masks.items():
                new = (mask != 0) & (before[name] == 0)
                grown += int((new & (cl.used_params[name] != 0)).sum())
        return grown

    def test_gradient_growth_avoids_frozen(self):
        # METHOD.md 2.14: truncate_weights reads weight.grad, already zeroed on
        # U, so gradient growth spends less budget on frozen slots than random.
        assert self._grown_into_frozen('rigl') <= self._grown_into_frozen('set')


# ---------------------------------------------------------------------------
# 9. LayerNorm
# ---------------------------------------------------------------------------

class TestLayerNorm:
    def test_excluded_from_mask(self):
        cl, _, _ = _build(model=TinyNormNet())
        assert set(cl.mask.masks) == {'fc1.weight', 'fc2.weight'}

    def test_gains_left_intact(self):
        torch.manual_seed(0)
        model = TinyNormNet()
        with torch.no_grad():  # move off the all-ones init so masking would show
            model.norm.weight.add_(torch.randn(16) * 0.1)
        before = model.norm.weight.detach().clone()
        _build(model=model)
        assert torch.equal(model.norm.weight.data, before)

    def test_no_layernorm_model_unaffected(self):
        cl, _, _ = _build()
        assert cl._ln_params == set()

    def test_weights_not_masked_twice(self):
        """Dropping the LayerNorms needs a second init(); the weights must not
        be masked by both, or most of the new mask would start at zero."""
        torch.manual_seed(0)
        model = TinyNormNet()
        cl, _, _ = _build(model=model)
        params = dict(model.named_parameters())
        for name, mask in cl.mask.masks.items():
            assert torch.all(params[name].data[mask != 0] != 0), name


# ---------------------------------------------------------------------------
# 10. stats()
# ---------------------------------------------------------------------------

class TestStats:
    KEYS = {'dst/mask_sparsity', 'dst/weight_sparsity', 'dst/itop_rate',
            'dst/pruning_ratio', 'cl_dst/claimed', 'cl_dst/frozen_active',
            'cl_dst/trainable'}

    def test_keys_and_ranges(self):
        cl, model, optimizer = _build()
        for _ in range(4):
            _do_step(cl, model, optimizer)
        stats = cl.stats()
        assert set(stats) == self.KEYS
        for key, value in stats.items():
            assert 0.0 <= value <= 1.0, key

    def test_first_chunk_has_nothing_frozen(self):
        cl, _, _ = _build()
        stats = cl.stats()
        assert stats['cl_dst/claimed'] == 0.0
        assert stats['cl_dst/trainable'] == 1.0

    def test_mask_sparsity_tracks_config(self):
        cl, _, _ = _build(sparsity=0.5)
        assert cl.stats()['dst/mask_sparsity'] == pytest.approx(0.5, abs=0.12)

    def test_capacity_decays(self):
        """claimed grows and trainable shrinks toward (1-d)^(t-1)."""
        cl, model, _ = _build(sparsity=0.5)
        claimed, trainable = [], []
        for i in (1, 2, 3):
            _run_chunk(cl, model, i, n_steps=6)
            claimed.append(cl.stats()['cl_dst/claimed'])
            trainable.append(cl.stats()['cl_dst/trainable'])
        assert claimed == sorted(claimed)
        assert trainable == sorted(trainable, reverse=True)
        # Entering chunk t, ~1-(1-d)^t of the network is claimed at d=0.5, and
        # only ~(1-d)^t of chunk t's own mask can still learn (METHOD.md 2.14).
        assert claimed[-1] == pytest.approx(1 - 0.5 ** 3, abs=0.15)
        assert trainable[-1] == pytest.approx(0.5 ** 3, abs=0.15)


# ---------------------------------------------------------------------------
# 11. End to end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.mark.parametrize('sparsifier', ['rigl', 'set'])
    def test_three_chunk_loop(self, sparsifier):
        cl, model, optimizer = _build(sparsifier=sparsifier)
        for _ in range(20):
            _do_step(cl, model, optimizer)
        cl.end_chunk()
        for i in (1, 2):
            _run_chunk(cl, model, i, n_steps=20)

        for name, param in model.named_parameters():
            assert torch.isfinite(param).all(), name
        assert cl.stats()['dst/mask_sparsity'] == pytest.approx(0.5, abs=0.12)

    def test_death_rate_decays_within_chunk(self):
        cl, model, optimizer = _build()
        start = cl.mask.death_rate
        for _ in range(40):
            _do_step(cl, model, optimizer)
        assert cl.mask.death_rate < start

    def test_drop_fraction_cosine_restarts_each_chunk(self):
        cl, model, optimizer = _build()
        for _ in range(40):
            _do_step(cl, model, optimizer)
        decayed = cl.mask.death_rate
        cl.end_chunk()
        _run_chunk(cl, model, 1, n_steps=1)
        assert cl.mask.death_rate > decayed

    def test_lr_schedule_is_cl_dsts(self):
        """MultiStepLR at 1/2 and 3/4 of the chunk's epochs, gamma 0.1."""
        cl, _, optimizer = _build(epochs=20)
        assert type(cl.lr_scheduler).__name__ == 'MultiStepLR'
        assert sorted(cl.lr_scheduler.milestones) == [10, 15]
        assert cl.lr_scheduler.gamma == 0.1
        assert optimizer.param_groups[0]['lr'] == cl.cfg.lr

    def test_lr_drops_at_both_milestones(self):
        cl, _, optimizer = _build(epochs=20)
        seen = []
        for _ in range(20):
            seen.append(optimizer.param_groups[0]['lr'])
            cl.lr_step()
        base = cl.cfg.lr
        assert seen[9] == pytest.approx(base)
        assert seen[10] == pytest.approx(base * 0.1)
        assert seen[15] == pytest.approx(base * 0.01)

    def test_lr_restarts_each_chunk(self):
        cl, model, optimizer = _build(epochs=20)
        for _ in range(16):
            cl.lr_step()
        assert optimizer.param_groups[0]['lr'] == pytest.approx(cl.cfg.lr * 0.01)
        cl.end_chunk()
        optimizer = cl.start_chunk(1, EPOCHS)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(cl.cfg.lr)

    def test_death_rate_stays_finite(self):
        cl, model, optimizer = _build()
        for _ in range(40):
            _do_step(cl, model, optimizer)
        assert math.isfinite(cl.mask.death_rate)
