"""T2.4: every sparse method trains end to end through train_sparse.py."""
import math

import pytest

from helpers import make_tiny_data, parse_sparse, run_train

METHODS = ["static", "set", "rigl", "gmp"]


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    return make_tiny_data(tmp_path_factory.mktemp("sparse_data"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory, tiny_data):
    """One run per sparse method, shared across the tests below."""
    out = {}
    for name in METHODS:
        evals, stdout, _ = run_train(
            tmp_path_factory.mktemp(name), tiny_data,
            extra_args=[f"--sparsifier={name}", "--sparsity=0.9",
                        "--num_mask_updates=20"],
        )
        out[name] = (evals, parse_sparse(stdout), stdout)
    return out


@pytest.mark.parametrize("name", METHODS)
class TestSparseRuns:
    def test_both_chunks_run(self, runs, name):
        evals, _, _ = runs[name]
        assert {e["chunk"] for e in evals} == {0, 1}

    def test_losses_finite(self, runs, name):
        evals, _, _ = runs[name]
        assert all(math.isfinite(e["train"]) and math.isfinite(e["val"]) for e in evals)

    def test_sparse_metrics_logged(self, runs, name):
        _, sparse, _ = runs[name]
        assert sparse, "no SPARSE lines emitted"
        for key in ("mask_sparsity", "weight_sparsity", "itop_rate"):
            assert key in sparse[0]

    def test_reaches_target_sparsity_by_the_end(self, runs, name):
        _, sparse, _ = runs[name]
        assert sparse[-1]["mask_sparsity"] == pytest.approx(0.9, abs=0.02)

    def test_weights_match_the_mask(self, runs, name):
        _, sparse, _ = runs[name]
        assert sparse[-1]["weight_sparsity"] == pytest.approx(
            sparse[-1]["mask_sparsity"], abs=0.01)

    def test_sparsifier_announced(self, runs, name):
        _, _, stdout = runs[name]
        assert f"[sparsifier] {name}" in stdout

    def test_only_block_layers_masked(self, runs, name):
        _, sparse, _ = runs[name]
        assert sparse[0]["num_sparse_modules"] == 8  # 2 blocks x 4 Linear


class TestSparsityShape:
    def test_gmp_ramps_up(self, runs):
        """GMP starts dense and prunes gradually; the others start sparse."""
        _, sparse, _ = runs["gmp"]
        assert sparse[0]["mask_sparsity"] < sparse[-1]["mask_sparsity"]

    @pytest.mark.parametrize("name", ["static", "set", "rigl"])
    def test_others_start_sparse(self, runs, name):
        _, sparse, _ = runs[name]
        assert sparse[0]["mask_sparsity"] == pytest.approx(0.9, abs=0.02)

    def test_static_topology_frozen(self, runs):
        """Static never rewires, so ITOP stays at the initial density."""
        _, sparse, _ = runs["static"]
        assert sparse[-1]["itop_rate"] == pytest.approx(sparse[0]["itop_rate"], abs=1e-6)

    @pytest.mark.parametrize("name", ["set", "rigl"])
    def test_dynamic_methods_explore(self, runs, name):
        """SET and RigL rewire, so ITOP rises above the initial density."""
        _, sparse, _ = runs[name]
        assert sparse[-1]["itop_rate"] > sparse[0]["itop_rate"]


class TestDenseUnaffected:
    def test_dense_logs_no_sparse_metrics(self, tmp_path, tiny_data):
        _, stdout, _ = run_train(tmp_path, tiny_data)
        assert "SPARSE" not in stdout
        assert "[sparsifier]" not in stdout
