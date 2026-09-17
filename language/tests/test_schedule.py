"""T2.2/T2.3: the chunk schedule and LR schedule match train.py's logic."""
import math

import pytest

from config_sparse import get_config
from train_sparse import build_run_name, chunk_configs, count_chunk_steps, get_lr

# Real sizes, so the numbers can be checked against the plan.
WIKITEXT_LEN = 119_541_000
OPENWEBTEXT_LEN = 9_035_582_489
TOKENS_PER_ITER = 8 * 60 * 1024  # grad_accum * batch * block at paper settings


@pytest.fixture
def cfg():
    return get_config([])


class TestChunkSteps:
    def test_paper_chunk0(self, cfg):
        """wikitext, ratio 1.0, replay 400 -> the 97,282 iterations in the plan."""
        chunk = {"dataset": "wikitext", "ratio": 1.0, "replay": 400}
        sched = count_chunk_steps(cfg, chunk, WIKITEXT_LEN, TOKENS_PER_ITER)
        assert sched["num_tokens"] == 47_816_400_000
        assert sched["num_iters"] == 97_282

    def test_paper_chunk1(self, cfg):
        chunk = {"dataset": "openwebtext", "ratio": 1.0, "replay": 2}
        sched = count_chunk_steps(cfg, chunk, OPENWEBTEXT_LEN, TOKENS_PER_ITER)
        assert sched["num_iters"] == 36_765

    def test_formula(self, cfg):
        chunk = {"dataset": "x", "ratio": 0.5, "replay": 3}
        sched = count_chunk_steps(cfg, chunk, 1_000_000, 1024)
        assert sched["num_tokens"] == int(3 * 1_000_000 * 0.5)
        assert sched["num_iters"] == int(3 * 1_000_000 * 0.5) // 1024

    @pytest.mark.parametrize("ratio,replay", [(0.0, 400), (1.0, 0)])
    def test_empty_chunk_is_skipped(self, cfg, ratio, replay):
        chunk = {"dataset": "x", "ratio": ratio, "replay": replay}
        assert count_chunk_steps(cfg, chunk, 1_000_000, 1024) is None

    def test_warmup_capped(self, cfg):
        chunk = {"dataset": "x", "ratio": 1.0, "replay": 400}
        sched = count_chunk_steps(cfg, chunk, WIKITEXT_LEN, TOKENS_PER_ITER)
        assert sched["warmup_iters"] == cfg.max_warmup_iters

    def test_warmup_is_ten_percent_when_short(self, cfg):
        chunk = {"dataset": "x", "ratio": 1.0, "replay": 1}
        sched = count_chunk_steps(cfg, chunk, 1024 * 1000, 1024)
        assert sched["warmup_iters"] == int(0.1 * sched["num_iters"])

    def test_at_least_twenty_evals(self, cfg):
        chunk = {"dataset": "x", "ratio": 1.0, "replay": 1}
        sched = count_chunk_steps(cfg, chunk, 1024 * 1000, 1024)
        assert sched["num_iters"] // sched["eval_interval"] >= 20

    def test_eval_interval_never_zero(self, cfg):
        """train.py divides by this; a very short chunk made it zero."""
        chunk = {"dataset": "x", "ratio": 1.0, "replay": 1}
        sched = count_chunk_steps(cfg, chunk, 5 * 1024, 1024)
        assert sched["eval_interval"] >= 1


class TestLearningRate:
    def test_warmup_rises_to_peak(self, cfg):
        lrs = [get_lr(cfg, i, 100, 1000) for i in range(100)]
        assert lrs == sorted(lrs)
        assert lrs[0] < cfg.learning_rate
        assert get_lr(cfg, 99, 100, 1000) == pytest.approx(cfg.learning_rate, rel=0.02)

    def test_decays_to_min(self, cfg):
        assert get_lr(cfg, 1000, 100, 1000) == pytest.approx(cfg.min_lr)
        assert get_lr(cfg, 5000, 100, 1000) == cfg.min_lr

    def test_cosine_midpoint(self, cfg):
        mid = get_lr(cfg, 550, 100, 1000)
        expected = cfg.min_lr + 0.5 * (cfg.learning_rate - cfg.min_lr)
        assert mid == pytest.approx(expected, rel=1e-3)

    def test_constant_when_disabled(self):
        cfg = get_config(["--decay_lr=False"])
        assert get_lr(cfg, 0, 100, 1000) == cfg.learning_rate
        assert get_lr(cfg, 999, 100, 1000) == cfg.learning_rate

    def test_never_exceeds_peak(self, cfg):
        for i in range(0, 1200, 7):
            assert 0 < get_lr(cfg, i, 100, 1000) <= cfg.learning_rate + 1e-12

    def test_min_lr_is_the_floor_after_warmup(self, cfg):
        # Warmup deliberately starts near zero, below min_lr, as in train.py.
        for i in range(100, 1200, 7):
            assert get_lr(cfg, i, 100, 1000) >= cfg.min_lr


class TestRunNames:
    def _name(self, argv):
        return build_run_name(get_config(argv))

    def test_arms_have_distinct_names(self):
        names = {
            self._name([]),
            self._name(["--method=fire"]),
            self._name(["--sparsifier=rigl"]),
            self._name(["--sparsifier=set"]),
            self._name(["--sparsifier=gmp"]),
            self._name(["--sparsifier=static"]),
        }
        assert len(names) == 6, "arms would share an output directory"

    def test_sparsity_in_name(self):
        assert "s0.9" in self._name(["--sparsifier=rigl"])
        assert self._name(["--sparsifier=rigl", "--sparsity=0.5"]) != self._name(["--sparsifier=rigl"])

    def test_seed_in_name(self):
        assert self._name(["--seed=1"]) != self._name(["--seed=2"])

    def test_chunk_config_in_name(self):
        assert "wikitext_1.0_400" in self._name([])


class TestChunkConfigs:
    def test_two_chunks_from_flags(self):
        cfg = get_config(["--c0_dataset=wikitext", "--c1_dataset=openwebtext"])
        chunks = chunk_configs(cfg)
        assert [c["dataset"] for c in chunks] == ["wikitext", "openwebtext"]
        assert chunks[0]["replay"] == cfg.c0_data_replay_ratio
