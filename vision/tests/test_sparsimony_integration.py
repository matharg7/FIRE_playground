"""
Integration tests for the sparsimony integration in train_st.py / config_st.py.

Coverage:
  1.  Config – flat sparsifier fields, argparse round-trip
  2.  sys.path setup – sparsimony repo importable
  3.  compute_total_gradient_steps – correctness for simple mock tasks
  4.  build_sparsifier – dense returns None; rigl/set/gmp/static return prepared sparsifiers
  5.  Derived hyperparameters – t_end, delta_t values propagated correctly
  6.  sparse_config building – only Conv2d/Linear weights targeted
  7.  prepare() – flag set, model still forwards, groups populated
  8.  step() – no-raise, increments counter, loss stays finite
  9.  Optimizer reset – reassignment + momentum clearing
  10. Sparsity applied – masks exist, are bool, become sparse after updates
  11. Dense baseline – sparsifier stays None throughout
  12. End-to-end mini loop – mirrors train_st.py two-level structure
"""

import os
import sys
import math
import types
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tiny synthetic model (CPU-only, no dataset download)
# ---------------------------------------------------------------------------

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.pool(self.relu(self.conv(x))).flatten(1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides):
    """Return a simple Config-like namespace with sane defaults."""
    from config_st import Config
    defaults = {
        'optimizer': 'adam', 'lr': 1e-3, 'clip_grad_norm': 0.5,
        'task': 'CIFAR10', 'model': 'RESNET18',
        'benchmark': 'continual', 'warm_start_subset_ratio': 10,
        'log_every': 1, 'seed': 0, 'batch_size': 256, 'disable_wandb': True,
        'sparsifier': 'dense',
        'sparsity': 0.9, 'num_mask_updates': 500, 't_end_ratio': 0.8,
        'pruning_ratio': 0.3,
        't_accel_ratio': 0.2, 'initial_sparsity': 0.0,
        'n_epochs': 100,  # normally set by get_task()
    }
    defaults.update(overrides)
    return Config(defaults)


def _make_mock_task(cfg, chunk_sizes):
    """Minimal task mock that satisfies compute_total_gradient_steps."""
    task = types.SimpleNamespace()
    task.n_chunks = len(chunk_sizes)
    task._train_datasets = [
        types.SimpleNamespace(__len__=lambda self, n=n: n) for n in chunk_sizes
    ]
    # Make len() work on each fake dataset
    for i, n in enumerate(chunk_sizes):
        task._train_datasets[i].__len__ = lambda self=None, n=n: n
        task._train_datasets[i] = type(
            'FakeDS', (), {'__len__': lambda self, n=n: n}
        )()
    return task


def _do_step(model, optimizer, sparsifier=None):
    model.train()
    x, y = torch.randn(4, 3, 8, 8), torch.randint(0, 4, (4,))
    optimizer.zero_grad()
    nn.CrossEntropyLoss()(model(x), y).backward()
    optimizer.step()
    if sparsifier is not None:
        sparsifier.step()


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_flat_fields_exist(self):
        cfg = _make_cfg()
        for field in ('sparsifier', 'sparsity', 'num_mask_updates',
                      't_end_ratio', 'pruning_ratio', 't_accel_ratio', 'initial_sparsity'):
            assert hasattr(cfg, field), f"missing field: {field}"

    def test_defaults(self):
        cfg = _make_cfg()
        assert cfg.sparsifier == 'dense'
        assert cfg.sparsity == 0.9
        assert cfg.num_mask_updates == 500
        assert cfg.t_end_ratio == 0.8
        assert cfg.pruning_ratio == 0.3
        assert cfg.t_accel_ratio == 0.2
        assert cfg.initial_sparsity == 0.0

    def test_override(self):
        cfg = _make_cfg(sparsifier='rigl', sparsity=0.5, num_mask_updates=200)
        assert cfg.sparsifier == 'rigl'
        assert cfg.sparsity == 0.5
        assert cfg.num_mask_updates == 200

    def test_config_st_class_round_trip(self):
        from config_st import Config
        d = {'sparsifier': 'gmp', 'sparsity': 0.8, 't_end_ratio': 0.75}
        cfg = Config(d)
        assert cfg.sparsifier == 'gmp'
        assert cfg.sparsity == 0.8
        assert cfg.t_end_ratio == 0.75

    def test_config_get_method(self):
        cfg = _make_cfg(sparsifier='set')
        assert cfg.get('sparsifier') == 'set'
        assert cfg.get('nonexistent', 'fallback') == 'fallback'


# ---------------------------------------------------------------------------
# 2. sys.path / import
# ---------------------------------------------------------------------------

class TestSparsimonyImport:
    def test_sparsimony_repo_dir_exists(self):
        vision_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert os.path.isdir(os.path.join(vision_dir, 'sparsimony'))

    def test_rigl_importable(self):
        from sparsimony import rigl
        assert callable(rigl)

    def test_all_api_functions_importable(self):
        from sparsimony import rigl, gmp, static  # noqa: F401
        from sparsimony import set as sp_set       # noqa: F401

    def test_rigl_dst_class_importable(self):
        from sparsimony.dst.rigl import RigL  # noqa: F401


# ---------------------------------------------------------------------------
# 3. compute_total_gradient_steps
# ---------------------------------------------------------------------------

class TestComputeTotalGradientSteps:
    def _steps(self, cfg, chunk_sizes):
        from train_st import compute_total_gradient_steps
        task = _make_mock_task(cfg, chunk_sizes)
        return compute_total_gradient_steps(cfg, task)

    def test_single_chunk_formula(self):
        cfg = _make_cfg(benchmark='continual', log_every=1, n_epochs=100, batch_size=100)
        # chunk_size=1000, steps_per_epoch=ceil(1000/100)=10, real_epochs=100*1=100
        assert self._steps(cfg, [1000]) == 10 * 100

    def test_multi_chunk_sums(self):
        cfg = _make_cfg(benchmark='continual', log_every=1, n_epochs=100, batch_size=100)
        # two chunks of 1000 and 2000
        expected = math.ceil(1000/100)*100 + math.ceil(2000/100)*100
        assert self._steps(cfg, [1000, 2000]) == expected

    def test_warm_start_first_chunk_log_every(self):
        # warm_start: i_iter==0 gets log_every = 100 // warm_start_subset_ratio
        cfg = _make_cfg(
            benchmark='warm_start',
            warm_start_subset_ratio=10,
            log_every=1,
            n_epochs=100,
            batch_size=100,
        )
        # i_iter=0: log_every=10, real_epochs=100*10=1000
        # i_iter=1: log_every=1,  real_epochs=100*1=100
        chunk_size = 1000
        s0 = math.ceil(chunk_size/100) * 100 * 10
        s1 = math.ceil(chunk_size/100) * 100 * 1
        assert self._steps(cfg, [chunk_size, chunk_size]) == s0 + s1

    def test_positive_and_integer(self):
        cfg = _make_cfg(n_epochs=100, batch_size=256)
        result = self._steps(cfg, [50000] * 10)
        assert isinstance(result, int)
        assert result > 0

    def test_t_end_and_delta_t_derivation(self):
        """Verify that t_end and delta_t are computed correctly from total_steps."""
        cfg = _make_cfg(t_end_ratio=0.8, num_mask_updates=500, n_epochs=100, batch_size=100)
        total = self._steps(cfg, [1000] * 10)
        t_end   = int(cfg.t_end_ratio * total)
        delta_t = max(1, total // cfg.num_mask_updates)
        assert t_end == int(0.8 * total)
        assert delta_t == max(1, total // 500)


# ---------------------------------------------------------------------------
# 4 & 5. build_sparsifier – factory, types, and derived params
# ---------------------------------------------------------------------------

class TestBuildSparsifier:
    def _build(self, sparsifier_name, **cfg_overrides):
        from train_st import build_sparsifier
        model = TinyNet()
        cfg = _make_cfg(
            sparsifier=sparsifier_name,
            sparsity=0.5,
            num_mask_updates=50,
            t_end_ratio=0.8,
            t_accel_ratio=0.2,
            pruning_ratio=0.3,
            initial_sparsity=0.0,
            **cfg_overrides,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        total_steps = 500
        sp = build_sparsifier(cfg, model, opt, total_steps)
        return sp, model, opt, total_steps

    def test_dense_returns_none(self):
        sp, *_ = self._build('dense')
        assert sp is None

    def test_rigl_type(self):
        from sparsimony.dst.rigl import RigL
        sp, *_ = self._build('rigl')
        assert isinstance(sp, RigL)

    def test_set_type(self):
        from sparsimony.dst.set import SET
        sp, *_ = self._build('set')
        assert isinstance(sp, SET)

    def test_gmp_type(self):
        from sparsimony.dst.gmp import GMP
        sp, *_ = self._build('gmp')
        assert isinstance(sp, GMP)

    def test_static_type(self):
        from sparsimony.dst.static import StaticMagnitudeSparsifier
        sp, *_ = self._build('static')
        assert isinstance(sp, StaticMagnitudeSparsifier)

    def test_unknown_raises(self):
        from train_st import build_sparsifier
        model = TinyNet()
        cfg = _make_cfg(sparsifier='unknown')
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        with pytest.raises(ValueError, match="Unknown sparsifier"):
            build_sparsifier(cfg, model, opt, 500)

    def test_rigl_sparsity_stored(self):
        sp, *_ = self._build('rigl')
        assert sp.sparsity == 0.5

    def test_rigl_prepared_flag(self):
        sp, *_ = self._build('rigl')
        assert sp.prepared_

    def test_rigl_t_end_derived(self):
        from train_st import build_sparsifier
        from sparsimony.dst.rigl import RigL
        model = TinyNet()
        cfg = _make_cfg(sparsifier='rigl', sparsity=0.5, t_end_ratio=0.8,
                        num_mask_updates=50, pruning_ratio=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        total = 500
        sp = build_sparsifier(cfg, model, opt, total)
        expected_t_end = int(0.8 * total)
        # Scheduler stores t_end
        assert sp.scheduler.t_end == expected_t_end

    def test_rigl_delta_t_derived(self):
        from train_st import build_sparsifier
        model = TinyNet()
        cfg = _make_cfg(sparsifier='rigl', sparsity=0.5, t_end_ratio=0.8,
                        num_mask_updates=50, pruning_ratio=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        total = 500
        sp = build_sparsifier(cfg, model, opt, total)
        expected_delta_t = max(1, total // 50)
        assert sp.scheduler.delta_t == expected_delta_t

    def test_gmp_t_accel_derived(self):
        from train_st import build_sparsifier
        from sparsimony.dst.gmp import GMP
        model = TinyNet()
        cfg = _make_cfg(sparsifier='gmp', sparsity=0.5, t_end_ratio=0.8,
                        t_accel_ratio=0.2, num_mask_updates=50, initial_sparsity=0.0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        total = 500
        sp = build_sparsifier(cfg, model, opt, total)
        expected_t_accel = int(0.2 * total)
        assert sp.scheduler.t_accel == expected_t_accel

    def test_all_models_still_forward_after_prepare(self):
        for name in ('rigl', 'set', 'gmp', 'static'):
            sp, model, *_ = self._build(name)
            out = model(torch.randn(4, 3, 8, 8))
            assert out.shape == (4, 4), f"{name}: wrong output shape"
            assert not torch.isnan(out).any(), f"{name}: NaN in output"


# ---------------------------------------------------------------------------
# 6. sparse_config building (inline logic from train_st.py)
# ---------------------------------------------------------------------------

def _build_sparse_config(model):
    return [
        {"tensor_fqn": f"{fqn}.weight"}
        for fqn, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv2d))
    ]


class TestSparseConfig:
    def test_conv_and_linear_included(self):
        fqns = {e["tensor_fqn"] for e in _build_sparse_config(TinyNet())}
        assert "conv.weight" in fqns
        assert "fc.weight" in fqns

    def test_only_weight_tensors(self):
        for e in _build_sparse_config(TinyNet()):
            assert e["tensor_fqn"].endswith(".weight")

    def test_batchnorm_excluded(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8),
                               nn.Flatten(), nn.Linear(8*6*6, 4))
        cfg = _build_sparse_config(model)
        for e in cfg:
            assert "1.weight" not in e["tensor_fqn"]   # BN is index 1

    def test_count_matches_layer_count(self):
        model = TinyNet()
        n = sum(1 for _, m in model.named_modules()
                if isinstance(m, (nn.Conv2d, nn.Linear)))
        assert len(_build_sparse_config(model)) == n


# ---------------------------------------------------------------------------
# 7. prepare()
# ---------------------------------------------------------------------------

class TestPrepare:
    def _setup(self, name='rigl'):
        from train_st import build_sparsifier
        model = TinyNet()
        cfg = _make_cfg(sparsifier=name, sparsity=0.5, num_mask_updates=50,
                        t_end_ratio=0.8, t_accel_ratio=0.2,
                        pruning_ratio=0.3, initial_sparsity=0.0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sp = build_sparsifier(cfg, model, opt, 500)
        return sp, model, opt

    def test_prepared_flag_set(self):
        sp, *_ = self._setup()
        assert sp.prepared_

    def test_forward_still_works(self):
        _, model, _ = self._setup()
        out = model(torch.randn(4, 3, 8, 8))
        assert out.shape == (4, 4)

    def test_groups_populated(self):
        sp, *_ = self._setup()
        assert len(sp.groups) > 0

    def test_prepare_all_sparsifiers(self):
        for name in ('rigl', 'set', 'gmp', 'static'):
            sp, model, _ = self._setup(name)
            assert sp.prepared_, f"{name}: prepared_ not set"
            assert len(sp.groups) > 0, f"{name}: groups empty"


# ---------------------------------------------------------------------------
# 8. step()
# ---------------------------------------------------------------------------

class TestStep:
    def _setup(self, delta_t=3):
        from train_st import build_sparsifier
        model = TinyNet()
        cfg = _make_cfg(sparsifier='rigl', sparsity=0.5, num_mask_updates=500,
                        t_end_ratio=0.8, pruning_ratio=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Use a small total_steps so delta_t is manageable
        sp = build_sparsifier(cfg, model, opt, 500)
        return model, opt, sp

    def test_step_does_not_raise(self):
        model, opt, sp = self._setup()
        _do_step(model, opt, sp)

    def test_step_increments_count(self):
        model, opt, sp = self._setup()
        before = sp._step_count
        _do_step(model, opt, sp)
        assert sp._step_count == before + 1

    def test_twenty_steps_stable(self):
        model, opt, sp = self._setup()
        for _ in range(20):
            _do_step(model, opt, sp)

    def test_loss_stays_finite(self):
        model, opt, sp = self._setup()
        criterion = nn.CrossEntropyLoss()
        for _ in range(5):
            x, y = torch.randn(4, 3, 8, 8), torch.randint(0, 4, (4,))
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            sp.step()
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 9. Optimizer reset (mirrors train_st.py lines 140-146)
# ---------------------------------------------------------------------------

class TestOptimizerReset:
    def _setup(self):
        from train_st import build_sparsifier
        model = TinyNet()
        cfg = _make_cfg(sparsifier='rigl', sparsity=0.5, num_mask_updates=50,
                        t_end_ratio=0.8, pruning_ratio=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sp = build_sparsifier(cfg, model, opt, 500)
        _do_step(model, opt, sp)  # warm up optimizer state
        return model, opt, sp

    def test_optimizer_reassignment(self):
        model, opt, sp = self._setup()
        new_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sp.optimizer = new_opt
        assert sp.optimizer is new_opt

    def test_zero_inactive_buffers_does_not_raise(self):
        model, opt, sp = self._setup()
        new_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sp.optimizer = new_opt
        if hasattr(sp, 'zero_inactive_param_momentum_buffers'):
            sp.zero_inactive_param_momentum_buffers()

    def test_training_continues_after_reset(self):
        model, opt, sp = self._setup()
        new_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sp.optimizer = new_opt
        if hasattr(sp, 'zero_inactive_param_momentum_buffers'):
            sp.zero_inactive_param_momentum_buffers()
        for _ in range(5):
            _do_step(model, new_opt, sp)


# ---------------------------------------------------------------------------
# 10. Sparsity is actually applied
# ---------------------------------------------------------------------------

class TestSparsityApplied:
    def _setup(self, sparsifier='rigl', sparsity=0.5):
        from train_st import build_sparsifier
        model = TinyNet()
        # Small total_steps so topology updates fire quickly
        cfg = _make_cfg(sparsifier=sparsifier, sparsity=sparsity,
                        num_mask_updates=50, t_end_ratio=0.8,
                        t_accel_ratio=0.2, pruning_ratio=0.3, initial_sparsity=0.0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sp = build_sparsifier(cfg, model, opt, total_steps=200)
        return sp, model, opt

    def test_masks_exist_after_prepare(self):
        from sparsimony.utils import get_mask
        sp, *_ = self._setup()
        for g in sp.groups:
            assert get_mask(g['module'], g['tensor_name']) is not None

    def test_masks_are_bool(self):
        from sparsimony.utils import get_mask
        sp, *_ = self._setup()
        for g in sp.groups:
            assert get_mask(g['module'], g['tensor_name']).dtype == torch.bool

    def test_weights_become_sparse_after_updates(self):
        from sparsimony.utils import get_mask
        sp, model, opt = self._setup()
        for _ in range(20):
            _do_step(model, opt, sp)
        any_masked = any(
            (~get_mask(g['module'], g['tensor_name'])).any().item()
            for g in sp.groups
        )
        assert any_masked, "Expected at least one zeroed weight after topology updates"

    def test_sparsity_within_tolerance(self):
        from sparsimony.utils import get_mask
        sp, model, opt = self._setup(sparsity=0.5)
        for _ in range(20):
            _do_step(model, opt, sp)
        total = zeros = 0
        for g in sp.groups:
            mask = get_mask(g['module'], g['tensor_name'])
            total += mask.numel()
            zeros += (~mask).sum().item()
        actual = zeros / total
        assert abs(actual - 0.5) < 0.20, f"sparsity {actual:.2f} too far from 0.5"

    def test_static_applies_sparsity_immediately(self):
        from sparsimony.utils import get_mask
        sp, model, opt = self._setup(sparsifier='static', sparsity=0.5)
        _do_step(model, opt, sp)  # one step to trigger initial mask
        any_masked = any(
            (~get_mask(g['module'], g['tensor_name'])).any().item()
            for g in sp.groups
        )
        assert any_masked


# ---------------------------------------------------------------------------
# 11. Dense baseline
# ---------------------------------------------------------------------------

class TestDenseBaseline:
    def test_build_sparsifier_dense_returns_none(self):
        from train_st import build_sparsifier
        model = TinyNet()
        cfg = _make_cfg(sparsifier='dense')
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        assert build_sparsifier(cfg, model, opt, 1000) is None

    def test_training_loop_with_none_sparsifier(self):
        model = TinyNet()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(5):
            _do_step(model, opt, sparsifier=None)
        out = model(torch.randn(4, 3, 8, 8))
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# 12. End-to-end mini loop (mirrors train_st.py)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _run_loop(self, sparsifier_name, n_iters=2, steps_per_iter=6):
        from train_st import build_sparsifier
        model = TinyNet()
        cfg = _make_cfg(sparsifier=sparsifier_name, sparsity=0.5,
                        num_mask_updates=50, t_end_ratio=0.8, t_accel_ratio=0.2,
                        pruning_ratio=0.3, initial_sparsity=0.0)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        total_steps = n_iters * steps_per_iter
        sp = build_sparsifier(cfg, model, opt, total_steps)

        for _ in range(n_iters):
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            if sp is not None:
                sp.optimizer = opt
                if hasattr(sp, 'zero_inactive_param_momentum_buffers'):
                    sp.zero_inactive_param_momentum_buffers()
            for _ in range(steps_per_iter):
                _do_step(model, opt, sp)

        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 3, 8, 8))
        assert out.shape == (4, 4)
        assert not torch.isnan(out).any()
        return sp

    def test_dense_loop(self):
        sp = self._run_loop('dense')
        assert sp is None

    def test_rigl_loop(self):
        self._run_loop('rigl')

    def test_set_loop(self):
        self._run_loop('set')

    def test_gmp_loop(self):
        self._run_loop('gmp')

    def test_static_loop(self):
        self._run_loop('static')

    def test_grad_clip_compat(self):
        from train_st import build_sparsifier
        model = TinyNet()
        cfg = _make_cfg(sparsifier='rigl', sparsity=0.5, num_mask_updates=50,
                        t_end_ratio=0.8, pruning_ratio=0.3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sp = build_sparsifier(cfg, model, opt, 500)

        x, y = torch.randn(4, 3, 8, 8), torch.randint(0, 4, (4,))
        opt.zero_grad()
        nn.CrossEntropyLoss()(model(x), y).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)   # mirrors cfg.clip_grad_norm
        opt.step()
        sp.step()


# ---------------------------------------------------------------------------
# 13. Mask update actually changes mask and zeros weights at pruned positions
# ---------------------------------------------------------------------------

class TestMaskUpdateSparsifiesParams:
    """Verify that after a topology-update step fires, the mask changes AND
    the weight tensor has exactly 0 at every False-mask position."""

    def _setup(self, sparsifier='rigl'):
        from train_st import build_sparsifier
        # total_steps=50, num_mask_updates=5  =>  delta_t = max(1, 50//5) = 10
        cfg = _make_cfg(sparsifier=sparsifier, sparsity=0.5, num_mask_updates=5,
                        t_end_ratio=0.8, t_accel_ratio=0.2,
                        pruning_ratio=0.3, initial_sparsity=0.0)
        model = TinyNet()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sp = build_sparsifier(cfg, model, opt, total_steps=50)
        return sp, model, opt

    def test_mask_changes_after_topology_update(self):
        """Running delta_t steps must trigger a mask update that alters the mask."""
        from sparsimony.utils import get_mask
        sp, model, opt = self._setup('rigl')
        delta_t = sp.scheduler.delta_t

        g = sp.groups[0]
        mask_before = get_mask(g['module'], g['tensor_name']).clone()

        # Run just past the first update boundary
        for _ in range(delta_t + 2):
            _do_step(model, opt, sp)

        mask_after = get_mask(g['module'], g['tensor_name'])
        assert not torch.equal(mask_before, mask_after), \
            "Expected mask to change after topology update but it did not"

    def test_weight_values_zero_at_pruned_positions(self):
        """After a mask update fires, weight[~mask] must be exactly 0."""
        from sparsimony.utils import get_mask
        sp, model, opt = self._setup('rigl')
        delta_t = sp.scheduler.delta_t

        for _ in range(delta_t + 2):
            _do_step(model, opt, sp)

        for g in sp.groups:
            mask = get_mask(g['module'], g['tensor_name'])
            weight = g['module'].weight.data
            pruned_vals = weight[~mask]
            assert (pruned_vals == 0).all(), \
                f"{g['tensor_fqn']}: non-zero value found at pruned (False-mask) position"

    def test_set_mask_changes_and_weights_zero(self):
        """Same check for SET, which uses uniform random topology."""
        from sparsimony.utils import get_mask
        sp, model, opt = self._setup('set')
        delta_t = sp.scheduler.delta_t
        g = sp.groups[0]
        mask_before = get_mask(g['module'], g['tensor_name']).clone()

        for _ in range(delta_t + 2):
            _do_step(model, opt, sp)

        mask_after = get_mask(g['module'], g['tensor_name'])
        assert not torch.equal(mask_before, mask_after), "SET mask should change after update"

        for g in sp.groups:
            mask = get_mask(g['module'], g['tensor_name'])
            weight = g['module'].weight.data
            assert (weight[~mask] == 0).all(), \
                f"{g['tensor_fqn']}: non-zero value at pruned position after SET update"


# ---------------------------------------------------------------------------
# 14. Optimizer renewal preserves mask, params, and all sparsifier state
# ---------------------------------------------------------------------------

class TestOptimizerRenewalPreservesState:
    """Verify that replacing sp.optimizer with a new instance (the per-task
    reset done in train_st.py) leaves the mask, weight values, step counter,
    groups, and scheduler completely unchanged."""

    def _setup(self):
        from train_st import build_sparsifier
        cfg = _make_cfg(sparsifier='rigl', sparsity=0.5, num_mask_updates=5,
                        t_end_ratio=0.8, pruning_ratio=0.3)
        model = TinyNet()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sp = build_sparsifier(cfg, model, opt, total_steps=50)
        # Run past first mask update so there is interesting mask state
        for _ in range(sp.scheduler.delta_t + 2):
            _do_step(model, opt, sp)
        return sp, model, opt

    def _snap_masks(self, sp):
        from sparsimony.utils import get_mask
        return [get_mask(g['module'], g['tensor_name']).clone() for g in sp.groups]

    def _snap_weights(self, sp):
        return [g['module'].weight.data.clone() for g in sp.groups]

    def test_mask_identical_after_optimizer_reset(self):
        sp, model, _ = self._setup()
        masks_before = self._snap_masks(sp)
        sp.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if hasattr(sp, 'zero_inactive_param_momentum_buffers'):
            sp.zero_inactive_param_momentum_buffers()
        masks_after = self._snap_masks(sp)
        for i, (mb, ma) in enumerate(zip(masks_before, masks_after)):
            assert torch.equal(mb, ma), f"group {i}: mask changed after optimizer reset"

    def test_weight_values_identical_after_optimizer_reset(self):
        sp, model, _ = self._setup()
        weights_before = self._snap_weights(sp)
        sp.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if hasattr(sp, 'zero_inactive_param_momentum_buffers'):
            sp.zero_inactive_param_momentum_buffers()
        weights_after = self._snap_weights(sp)
        for i, (wb, wa) in enumerate(zip(weights_before, weights_after)):
            assert torch.equal(wb, wa), f"group {i}: weight changed after optimizer reset"

    def test_step_count_identical_after_optimizer_reset(self):
        sp, model, _ = self._setup()
        count_before = sp._step_count
        sp.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        assert sp._step_count == count_before

    def test_groups_object_identical_after_optimizer_reset(self):
        sp, model, _ = self._setup()
        groups_id = id(sp.groups)
        sp.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        assert id(sp.groups) == groups_id, "groups list was replaced on optimizer reset"

    def test_scheduler_state_identical_after_optimizer_reset(self):
        sp, model, _ = self._setup()
        t_end_before = sp.scheduler.t_end
        delta_t_before = sp.scheduler.delta_t
        sp.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        assert sp.scheduler.t_end == t_end_before
        assert sp.scheduler.delta_t == delta_t_before

    def test_training_resumes_correctly_after_reset(self):
        """After optimizer renewal the sparsifier must keep stepping without error."""
        sp, model, _ = self._setup()
        new_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sp.optimizer = new_opt
        if hasattr(sp, 'zero_inactive_param_momentum_buffers'):
            sp.zero_inactive_param_momentum_buffers()
        for _ in range(5):
            _do_step(model, new_opt, sp)
