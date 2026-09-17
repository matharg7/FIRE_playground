"""T1.2/T1.3/T1.4/T1.7: layer selection, sparsifier construction, metrics."""
import pytest
import torch
import torch.nn as nn

from config_sparse import get_config
from sparse_utils import (
    CorrectedAcceleratedCubicScheduler,
    ITOPTracker,
    build_sparsifier,
    count_sparse_params,
    get_sparse_targets,
    get_sparsity_stats,
    sparse_metrics,
    sparsifier_schedule,
)

# GPT-2 small: 12 blocks x 4 Linear weights, holding 84,934,656 of 124,354,560 weights.
GPT2_TARGETS = 48
GPT2_SPARSE_PARAMS = 84_934_656


def _train_steps(model, optimizer, sparsifier, n, vocab=128, block=32, batch=2):
    """Run n optimizer steps, stepping the sparsifier like train_sparse.py does."""
    topo_changes = 0
    for _ in range(n):
        x = torch.randint(0, vocab, (batch, block))
        y = torch.randint(0, vocab, (batch, block))
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if sparsifier is not None and sparsifier.step():
            topo_changes += 1
    return topo_changes


class TestTargets:
    def test_tiny_model_selects_block_linears(self, tiny_gpt):
        targets = get_sparse_targets(tiny_gpt)
        assert len(targets) == 2 * 4  # 2 blocks x (c_attn, attn.c_proj, c_fc, mlp.c_proj)

    def test_all_targets_are_block_weights(self, tiny_gpt):
        for t in get_sparse_targets(tiny_gpt):
            assert t["tensor_fqn"].startswith("transformer.h")
            assert t["tensor_fqn"].endswith(".weight")

    def test_excludes_head_embeddings_and_norms(self, tiny_gpt):
        fqns = {t["tensor_fqn"] for t in get_sparse_targets(tiny_gpt)}
        assert not any("lm_head" in f for f in fqns)
        assert not any("wte" in f or "wpe" in f for f in fqns)
        assert not any("ln_" in f for f in fqns)

    def test_count_matches_named_parameters(self, tiny_gpt):
        targets = get_sparse_targets(tiny_gpt)
        fqns = {t["tensor_fqn"] for t in targets}
        expected = sum(p.numel() for n, p in tiny_gpt.named_parameters() if n in fqns)
        assert count_sparse_params(tiny_gpt, targets) == expected

    def test_raises_for_non_gpt(self):
        assert get_sparse_targets(nn.Sequential(nn.Linear(4, 4))) == []

    @pytest.mark.slow
    def test_gpt2_small_counts(self):
        """The numbers quoted in the plan (D3)."""
        from model import GPT, GPTConfig
        model = GPT(GPTConfig(block_size=1024, vocab_size=50304, n_layer=12,
                              n_head=12, n_embd=768, dropout=0.0, bias=False))
        targets = get_sparse_targets(model)
        assert len(targets) == GPT2_TARGETS
        assert count_sparse_params(model, targets) == GPT2_SPARSE_PARAMS


class TestSchedule:
    def test_rigl_delta_t(self):
        cfg = get_config(["--sparsifier=rigl", "--num_mask_updates=500"])
        sched = sparsifier_schedule(cfg, 134_049)
        assert sched["t_end"] == 107_239
        assert sched["delta_t"] == 107_239 // 500

    def test_gmp_delta_t_spans_accel_to_end(self):
        cfg = get_config(["--sparsifier=gmp", "--num_mask_updates=500"])
        sched = sparsifier_schedule(cfg, 134_049)
        assert sched["t_accel"] == 26_809
        assert sched["delta_t"] == (sched["t_end"] - sched["t_accel"]) // 500

    def test_delta_t_never_zero(self):
        cfg = get_config(["--sparsifier=set", "--num_mask_updates=500"])
        assert sparsifier_schedule(cfg, 10)["delta_t"] >= 1


class TestBuild:
    def test_dense_returns_none(self, tiny_gpt):
        cfg = get_config([])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        assert build_sparsifier(cfg, tiny_gpt, opt, 100) is None

    @pytest.mark.parametrize("name", ["static", "set", "rigl"])
    def test_reaches_target_sparsity_immediately(self, tiny_gpt, name):
        cfg = get_config([f"--sparsifier={name}", "--sparsity=0.9"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        build_sparsifier(cfg, tiny_gpt, opt, 100)
        assert get_sparsity_stats(tiny_gpt)["mask_sparsity"] == pytest.approx(0.9, abs=0.01)

    def test_gmp_starts_dense(self, tiny_gpt):
        """GMP ramps up: at step 0 it is still at initial_sparsity."""
        cfg = get_config(["--sparsifier=gmp", "--sparsity=0.9", "--initial_sparsity=0.0"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        build_sparsifier(cfg, tiny_gpt, opt, 100)
        assert get_sparsity_stats(tiny_gpt)["mask_sparsity"] == pytest.approx(0.0, abs=0.01)

    def test_only_targets_are_masked(self, tiny_gpt):
        cfg = get_config(["--sparsifier=static"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        build_sparsifier(cfg, tiny_gpt, opt, 100)
        assert not hasattr(tiny_gpt.lm_head, "parametrizations")
        assert get_sparsity_stats(tiny_gpt)["num_sparse_modules"] == 8


class TestTiedWeights:
    """D3: lm_head shares storage with the token embedding, so it stays dense."""

    def test_tie_survives_prepare(self, tiny_gpt):
        cfg = get_config(["--sparsifier=static"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        build_sparsifier(cfg, tiny_gpt, opt, 100)
        assert tiny_gpt.lm_head.weight is tiny_gpt.transformer.wte.weight

    def test_embedding_is_dense(self, tiny_gpt):
        cfg = get_config(["--sparsifier=static", "--sparsity=0.9"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        build_sparsifier(cfg, tiny_gpt, opt, 100)
        wte = tiny_gpt.transformer.wte.weight
        assert torch.count_nonzero(wte) == wte.numel()

    def test_model_still_runs(self, tiny_gpt):
        cfg = get_config(["--sparsifier=static"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        build_sparsifier(cfg, tiny_gpt, opt, 100)
        x = torch.randint(0, 128, (2, 32))
        logits, loss = tiny_gpt(x, x)
        assert torch.isfinite(loss)


class TestTraining:
    def test_masked_weights_stay_zero_after_steps(self, tiny_gpt):
        cfg = get_config(["--sparsifier=static", "--sparsity=0.9"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-2)
        sp = build_sparsifier(cfg, tiny_gpt, opt, 100)
        _train_steps(tiny_gpt, opt, sp, 5)
        assert get_sparsity_stats(tiny_gpt)["weight_sparsity"] == pytest.approx(0.9, abs=0.01)

    def test_optimizer_reset_keeps_sparsifier_working(self, tiny_gpt):
        """train_sparse.py clears optimizer state between chunks, keeping the
        same optimizer object, so the sparsifier's hook stays registered."""
        cfg = get_config(["--sparsifier=static", "--sparsity=0.9"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-2)
        sp = build_sparsifier(cfg, tiny_gpt, opt, 100)
        _train_steps(tiny_gpt, opt, sp, 3)
        opt.state.clear()
        _train_steps(tiny_gpt, opt, sp, 3)
        assert get_sparsity_stats(tiny_gpt)["weight_sparsity"] == pytest.approx(0.9, abs=0.01)

    @pytest.mark.parametrize("name", ["set", "rigl"])
    def test_topology_changes_over_time(self, tiny_gpt, name):
        cfg = get_config([f"--sparsifier={name}", "--sparsity=0.9",
                          "--num_mask_updates=40"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-2)
        sp = build_sparsifier(cfg, tiny_gpt, opt, 100)
        assert _train_steps(tiny_gpt, opt, sp, 12) > 0

    def test_gmp_sparsity_increases(self, tiny_gpt):
        cfg = get_config(["--sparsifier=gmp", "--sparsity=0.9", "--num_mask_updates=10"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-2)
        sp = build_sparsifier(cfg, tiny_gpt, opt, 60)
        before = get_sparsity_stats(tiny_gpt)["mask_sparsity"]
        _train_steps(tiny_gpt, opt, sp, 50)
        assert get_sparsity_stats(tiny_gpt)["mask_sparsity"] > before


class TestGMPCubicFix:
    """sparsimony's ramp undershoots the target; CorrectedAccelerated... does not."""

    def _final(self, cls, t_end=1000, t_accel=250):
        sched = cls(t_end=t_end, delta_t=1, t_accel=t_accel,
                    initial_sparsity=0.0, accelerated_sparsity=0.7, final_sparsity=0.9)
        return sched(t_end)

    def test_corrected_reaches_target(self):
        assert self._final(CorrectedAcceleratedCubicScheduler) == pytest.approx(0.9)

    def test_original_undershoots(self):
        from sparsimony.schedulers.base import AcceleratedCubicScheduler
        assert self._final(AcceleratedCubicScheduler) == pytest.approx(0.8969, abs=1e-3)


class TestMetrics:
    def _prepared(self, model, name="set", extra=()):
        cfg = get_config([f"--sparsifier={name}", "--sparsity=0.9", *extra])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        return build_sparsifier(cfg, model, opt, 100), opt

    def test_dense_returns_no_metrics(self, tiny_gpt):
        assert sparse_metrics(tiny_gpt, None) == {}

    def test_keys_present(self, tiny_gpt):
        sp, _ = self._prepared(tiny_gpt)
        metrics = sparse_metrics(tiny_gpt, sp, ITOPTracker(sp))
        for key in ("dst/mask_sparsity", "dst/weight_sparsity", "dst/itop_rate"):
            assert key in metrics

    def test_itop_starts_at_density(self, tiny_gpt):
        sp, _ = self._prepared(tiny_gpt)
        assert ITOPTracker(sp).compute() == pytest.approx(0.1, abs=0.01)

    def test_itop_grows_as_topology_changes(self, tiny_gpt):
        sp, opt = self._prepared(tiny_gpt, "set", ["--num_mask_updates=40"])
        tracker = ITOPTracker(sp)
        start = tracker.compute()
        for _ in range(12):
            x = torch.randint(0, 128, (2, 32))
            opt.zero_grad()
            _, loss = tiny_gpt(x, x)
            loss.backward()
            opt.step()
            if sp.step():
                tracker.update()
        assert tracker.compute() > start

    def test_itop_never_exceeds_one(self, tiny_gpt):
        sp, _ = self._prepared(tiny_gpt)
        assert 0.0 <= ITOPTracker(sp).compute() <= 1.0
