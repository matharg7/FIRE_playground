"""Weight Space Consolidation: the trim, the plateau trigger, and the guards.

Ported from umamicode/weight-space-consolidation models/wsc.py (commit 026c54e9);
these tests pin the arithmetic so the port cannot drift from the reference.
"""

import pytest
import torch
import torch.nn as nn

import config
import wsc


def cfg_with(**kw):
    return config.get_config([], **kw)


class Tiny(nn.Module):
    def __init__(self, n=100):
        super().__init__()
        self.a = nn.Linear(n, n, bias=False)
        self.b = nn.Linear(n, 1, bias=False)

    def forward(self, x):
        return self.b(self.a(x))


# --- rank-based parameter reset ------------------------------------------

def test_trim_keeps_exactly_the_top_percent_untouched():
    torch.manual_seed(0)
    m = Tiny(10)
    prev = wsc.snapshot_params(m)
    with torch.no_grad():                      # move away from theta(t-1)
        for p in m.parameters():
            p.add_(1.0)
    cur = {n: p.detach().clone() for n, p in m.named_parameters()}
    # a strict ranking so "top 20%" is unambiguous
    scores = {n: torch.arange(p.numel(), dtype=torch.float32).reshape(p.shape)
              for n, p in m.named_parameters()}
    n_t, n_blend = wsc.pre_swa_trim(m, prev, scores, retain_percent=20.0)
    assert n_t == 2
    for name, p in m.named_parameters():
        n = p.numel()
        k = max(1, int(n * 0.2))
        s = scores[name].flatten()
        keep = s >= s.topk(k).values.min()
        flat_now, flat_cur = p.detach().flatten(), cur[name].flatten()
        flat_prev = prev[name].flatten()
        # kept weights are untouched
        assert torch.allclose(flat_now[keep], flat_cur[keep])
        # the rest are exactly half-way to theta(t-1)
        assert torch.allclose(flat_now[~keep],
                              0.5 * flat_cur[~keep] + 0.5 * flat_prev[~keep], atol=1e-6)
    total = sum(p.numel() for p in m.parameters())
    assert n_blend == total - sum(max(1, int(p.numel() * 0.2)) for p in m.parameters())


def test_trim_is_a_noop_without_a_previous_task():
    m = Tiny(8)
    before = [p.detach().clone() for p in m.parameters()]
    assert wsc.pre_swa_trim(m, None, {}, 20.0) == (0, 0)
    for p, b in zip(m.parameters(), before):
        assert torch.equal(p.detach(), b)


def test_retain_100_percent_changes_nothing():
    torch.manual_seed(0)
    m = Tiny(10)
    prev = wsc.snapshot_params(m)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    before = [p.detach().clone() for p in m.parameters()]
    scores = {n: torch.rand_like(p) for n, p in m.named_parameters()}
    wsc.pre_swa_trim(m, prev, scores, retain_percent=100.0)
    for p, b in zip(m.parameters(), before):
        assert torch.allclose(p.detach(), b)


# --- importance scoring --------------------------------------------------

def test_moment_scores_match_adam_bias_correction():
    torch.manual_seed(0)
    m = Tiny(4)
    tr = wsc.MomentTracker(m, beta1=0.9, beta2=0.999)
    g = {}
    for step in range(3):
        for n, p in m.named_parameters():
            p.grad = torch.full_like(p, 0.1 * (step + 1))
            g.setdefault(n, []).append(p.grad.clone())
        tr.update(m)
    assert tr.t == 3
    s = tr.scores()
    for n, p in m.named_parameters():
        mm = torch.zeros_like(p); vv = torch.zeros_like(p)
        for gi in g[n]:
            mm = 0.9 * mm + 0.1 * gi
            vv = 0.999 * vv + 0.001 * gi * gi
        want = (mm / (1 - 0.9 ** 3)).abs() * (vv / (1 - 0.999 ** 3))
        assert torch.allclose(s[n], want, atol=1e-7)


def test_moment_tracker_ignores_params_without_grad():
    m = Tiny(4)
    tr = wsc.MomentTracker(m)
    m.a.weight.grad = torch.ones_like(m.a.weight)   # only one has a grad
    tr.update(m)
    s = tr.scores()
    assert s["a.weight"].abs().sum() > 0
    assert s["b.weight"].abs().sum() == 0


# --- plateau detection ---------------------------------------------------

def test_plateau_fires_after_patience_non_improving_epochs():
    w = wsc.PlateauWatcher(patience=2, tol=1e-4)
    assert not w.step(1.0)          # sets the baseline
    assert not w.step(0.5)          # improved
    assert not w.step(0.5)          # stagnant 1
    assert w.step(0.5)              # stagnant 2 -> fire
    assert not w.step(0.5)          # fires once only
    assert w.fired


def test_improvement_smaller_than_tol_counts_as_stagnant():
    w = wsc.PlateauWatcher(patience=1, tol=1e-2)
    w.step(1.0)
    assert w.step(0.995)            # a 0.005 gain is inside tol -> stagnant
    assert w.best == 1.0


# --- the guard that stops a silent no-op --------------------------------

def test_schedule_guard_rejects_tasks_too_short_for_swa():
    c = cfg_with(cl_method="wsc", wsc_patience=4)
    with pytest.raises(ValueError, match="cannot trigger"):
        wsc.check_wsc_schedule(c, [3, 3, 5], ["FOMC", "ScienceQA", "NumGLUE-cm"])
    wsc.check_wsc_schedule(c, [10, 10, 20], ["a", "b", "c"])      # 4+2 <= 10, fine
    c1 = cfg_with(cl_method="wsc", wsc_patience=1)
    wsc.check_wsc_schedule(c1, [3, 3, 5], ["a", "b", "c"])        # 1+2 <= 3, fine


def test_guard_is_skipped_when_cl_method_is_none():
    wsc.check_wsc_schedule(cfg_with(cl_method="none"), [1, 1, 1])


def test_swa_lr_defaults_to_the_run_learning_rate():
    c = cfg_with(cl_method="wsc", learning_rate=1e-5, wsc_swa_lr=0.0)
    ctl = wsc.WSCController(c, Tiny(4), lambda: 0.0, log=lambda *a, **k: None)
    assert ctl.swa_lr == 1e-5       # NOT upstream's 0.1, which is an SGD/vision value
    c2 = cfg_with(cl_method="wsc", learning_rate=1e-5, wsc_swa_lr=3e-5)
    assert wsc.WSCController(c2, Tiny(4), lambda: 0.0, log=lambda *a, **k: None).swa_lr == 3e-5


def test_config_validates_cl_method():
    assert cfg_with(cl_method="wsc").cl_method == "wsc"
    with pytest.raises(ValueError):
        cfg_with(cl_method="nonsense")


# --- SWA end to end on a tiny model -------------------------------------

def test_swa_average_matches_the_mean_of_the_visited_weights():
    torch.manual_seed(0)
    m = Tiny(4)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    swa_model, swa_sched = wsc.make_swa(m, opt, swa_lr=0.1, anneal_epochs=2)
    seen = []
    for _ in range(3):
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        seen.append(m.a.weight.detach().clone())
        swa_model.update_parameters(m)
        swa_sched.step()
    want = torch.stack(seen).mean(0)
    assert torch.allclose(swa_model.module.a.weight, want, atol=1e-6)
    wsc.load_swa_into(m, swa_model)
    assert torch.allclose(m.a.weight, want, atol=1e-6)
