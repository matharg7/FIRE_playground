"""T2.2/T2.3: train_sparse.py runs end to end on CPU for each arm."""
import math

import pytest

from helpers import make_tiny_data, run_train


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    return make_tiny_data(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="module")
def dense_run(tmp_path_factory, tiny_data):
    evals, stdout, _ = run_train(tmp_path_factory.mktemp("dense"), tiny_data)
    return evals, stdout


class TestDenseRun:
    def test_both_chunks_run(self, dense_run):
        evals, _ = dense_run
        assert {e["chunk"] for e in evals} == {0, 1}

    def test_losses_finite(self, dense_run):
        evals, _ = dense_run
        assert all(math.isfinite(e["train"]) and math.isfinite(e["val"]) for e in evals)

    def test_model_learns(self, dense_run):
        evals, _ = dense_run
        c0 = [e for e in evals if e["chunk"] == 0]
        assert c0[-1]["train"] < c0[0]["train"]

    def test_global_iter_continues_across_chunks(self, dense_run):
        evals, _ = dense_run
        c0_last = max(e["global_iter"] for e in evals if e["chunk"] == 0)
        c1_first = min(e["global_iter"] for e in evals if e["chunk"] == 1)
        assert c1_first > c0_last

    def test_local_iter_restarts_each_chunk(self, dense_run):
        evals, _ = dense_run
        assert min(e["local_iter"] for e in evals if e["chunk"] == 1) == 0


class TestFireRun:
    def test_fire_at_boundary(self, tmp_path, tiny_data):
        """FIRE runs chunk 0, reinitializes, then runs chunk 1 in one process."""
        evals, stdout, _ = run_train(tmp_path, tiny_data, extra_args=["--method=fire"])
        assert "Applying FIRE" in stdout
        assert {e["chunk"] for e in evals} == {0, 1}
        assert all(math.isfinite(e["train"]) for e in evals)

    def test_fire_only_once(self, tmp_path, tiny_data):
        _, stdout, _ = run_train(tmp_path, tiny_data, extra_args=["--method=fire"])
        assert stdout.count("Applying FIRE") == 1


class TestFullReset:
    def test_skips_first_chunk(self, tmp_path, tiny_data):
        evals, stdout, _ = run_train(tmp_path, tiny_data, extra_args=["--method=full_reset"])
        assert "[SKIPPED]" in stdout
        assert {e["chunk"] for e in evals} == {1}

    def test_counters_still_advance_through_skipped_chunk(self, tmp_path, tiny_data):
        """Skipped chunks keep the global step aligned with runs that trained them."""
        evals, _, _ = run_train(tmp_path, tiny_data, extra_args=["--method=full_reset"])
        assert min(e["global_iter"] for e in evals) > 0


class TestCheckpoints:
    def test_writes_expected_files(self, tmp_path, tiny_data):
        from helpers import stage_run
        evals, _, _ = run_train(
            tmp_path, tiny_data,
            extra_args=["--save_checkpoint=True", "--out_root=output"],
        )
        out = stage_run(tmp_path, tiny_data) / "output"
        runs = list(out.iterdir())
        assert len(runs) == 1, f"expected one run dir, got {runs}"
        names = {p.name for p in runs[0].iterdir()}
        assert "init_ckpt.pt" in names
        assert "best_chunk0_ckpt.pt" in names
        assert "chunk1_ckpt.pt" in names

    def test_checkpoint_keys_have_no_compile_prefix(self, tmp_path, tiny_data):
        import torch
        from helpers import stage_run
        run_train(tmp_path, tiny_data, extra_args=["--save_checkpoint=True"])
        out = stage_run(tmp_path, tiny_data) / "output"
        ckpt = torch.load(next(next(out.iterdir()).glob("init_ckpt.pt")),
                          map_location="cpu", weights_only=False)
        assert not any(k.startswith("_orig_mod.") for k in ckpt["model"])
