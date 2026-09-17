"""Phase 3: correctness on real GPUs — compile, fp16, DDP.

    pytest -m gpu      # inside a GPU allocation
"""
import math
import os
import subprocess
import sys

import pytest
import torch

from helpers import make_tiny_data, parse_sparse, run_ddp, run_train, stage_run

pytestmark = pytest.mark.gpu

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SPARSIFIERS = ["static", "gmp", "set", "rigl"]


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    return make_tiny_data(tmp_path_factory.mktemp("gpu_data"))


# ---------------------------------------------------------------------------
# T3.1 torch.compile
# ---------------------------------------------------------------------------

class TestCompile:
    @pytest.mark.parametrize("sparsifier", ["dense"] + SPARSIFIERS)
    def test_compiled_run_completes(self, tmp_path, tiny_data, sparsifier):
        """torch.compile wraps the model after the sparsifier reparametrizes it."""
        evals, stdout, _ = run_train(
            tmp_path, tiny_data,
            extra_args=["--device=cuda", "--dtype=float16", "--compile=True",
                        f"--sparsifier={sparsifier}", "--num_mask_updates=20"],
        )
        assert {e["chunk"] for e in evals} == {0, 1}
        assert all(math.isfinite(e["train"]) for e in evals)

    @pytest.mark.parametrize("sparsifier", ["dense", "rigl"])
    def test_compiled_matches_eager(self, tmp_path, tiny_data, sparsifier):
        """Compiling must not change what the model computes."""
        common = ["--device=cuda", "--dtype=float32", f"--sparsifier={sparsifier}",
                  "--num_mask_updates=20"]
        eager, _, _ = run_train(tmp_path / "eager", tiny_data,
                                extra_args=common + ["--compile=False"])
        comp, _, _ = run_train(tmp_path / "compiled", tiny_data,
                               extra_args=common + ["--compile=True"])
        assert len(eager) == len(comp)
        # Same seed and data, so the first eval (before any update) must agree closely.
        assert eager[0]["train"] == pytest.approx(comp[0]["train"], abs=1e-3)


# ---------------------------------------------------------------------------
# T1.6 fp16 + gradient accumulation
# ---------------------------------------------------------------------------

class TestMixedPrecision:
    @pytest.mark.parametrize("sparsifier", SPARSIFIERS)
    def test_fp16_with_micro_steps(self, tmp_path, tiny_data, sparsifier):
        """GradScaler skips steps on overflow; masks must survive that."""
        evals, stdout, _ = run_train(
            tmp_path, tiny_data,
            extra_args=["--device=cuda", "--dtype=float16", "--compile=False",
                        "--gradient_accumulation_steps=4", "--batch_size=8",
                        f"--sparsifier={sparsifier}", "--sparsity=0.9",
                        "--num_mask_updates=20"],
        )
        sparse = parse_sparse(stdout)
        assert all(math.isfinite(e["train"]) for e in evals)
        assert sparse[-1]["mask_sparsity"] == pytest.approx(0.9, abs=0.02)

    def test_bfloat16_runs(self, tmp_path, tiny_data):
        evals, _, _ = run_train(
            tmp_path, tiny_data,
            extra_args=["--device=cuda", "--dtype=bfloat16", "--compile=False",
                        "--sparsifier=rigl", "--num_mask_updates=20"],
        )
        assert all(math.isfinite(e["train"]) for e in evals)


# ---------------------------------------------------------------------------
# T3.2 / T3.4 DDP
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("SLURM_JOB_ID") and torch.cuda.device_count() < 2,
    reason="needs 2 GPUs or a SLURM allocation with 2 tasks",
)
class TestDDP:
    @pytest.mark.parametrize("sparsifier", SPARSIFIERS)
    def test_masks_identical_across_ranks(self, sparsifier):
        """Each rank sees different data; sparsimony broadcasts masks from rank 0."""
        from helpers import ddp_launcher
        proc = subprocess.run(
            [*ddp_launcher(2), sys.executable, "ddp_mask_check.py",
             f"--sparsifier={sparsifier}", "--sparsity=0.9", "--num_mask_updates=10"],
            cwd=TESTS_DIR, capture_output=True, text=True, timeout=900,
        )
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
        assert "MASKS_IDENTICAL" in proc.stdout

    def test_ddp_run_completes(self, tmp_path, tiny_data):
        evals, stdout, _run_dir = run_ddp(
            tmp_path, tiny_data, nproc=2,
            extra_args=["--sparsifier=rigl", "--sparsity=0.9", "--num_mask_updates=5",
                        "--c0_subset_ratio=0.3", "--c1_subset_ratio=0.1",
                        "--eval_iters=1"],
        )
        assert {e["chunk"] for e in evals} == {0, 1}

    def test_only_rank_zero_writes_checkpoints(self, tmp_path, tiny_data):
        """Every rank writing the same file races on a shared filesystem."""
        _, _, run_dir = run_ddp(
            tmp_path, tiny_data, nproc=2,
            extra_args=["--save_checkpoint=True", "--sparsifier=static",
                        "--c0_subset_ratio=0.3", "--c1_subset_ratio=0.1",
                        "--eval_iters=1"])
        run_dirs = list((run_dir / "output").iterdir())
        assert len(run_dirs) == 1
        assert len(list(run_dirs[0].glob("DONE"))) == 1
        ckpt = torch.load(run_dirs[0] / "init_ckpt.pt", map_location="cpu",
                          weights_only=False)
        assert "model" in ckpt  # loads cleanly, so it was not written twice
