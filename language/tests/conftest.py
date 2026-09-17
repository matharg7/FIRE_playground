"""Shared fixtures for the language/ test-suite.

Run from language/:
    pytest -m "not gpu"      # everything that works on a login node
    pytest -m gpu            # inside a GPU allocation
"""
import os
import sys

import numpy as np
import pytest
import torch

LANGUAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(LANGUAGE_DIR)
# sparsimony is vendored under vision/, not language/.
SPARSIMONY_REPO = os.path.join(REPO_ROOT, "vision", "sparsimony")

for path in (LANGUAGE_DIR, SPARSIMONY_REPO):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

# Real data uses 50256 as <|endoftext|>; the tiny fixtures use a separator
# inside the tiny vocab so the small model can embed every id.
TINY_VOCAB = 128
TINY_EOT = TINY_VOCAB - 1


def pytest_collection_modifyitems(config, items):
    """Skip gpu-marked tests when there is no GPU to reach.

    A SLURM allocation counts: the DDP tests launch their ranks with srun, so
    they work from a login node as long as a job is allocated. Those srun calls
    must come from outside a job step, since a step cannot create a step.
    """
    if torch.cuda.is_available() or os.environ.get("SLURM_JOB_ID"):
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tiny_gpt_config():
    """A GPT small enough to build and train on a CPU in milliseconds."""
    from model import GPTConfig

    return GPTConfig(
        block_size=32,
        vocab_size=128,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=False,
    )


@pytest.fixture
def tiny_gpt(tiny_gpt_config):
    from model import GPT

    torch.manual_seed(0)
    return GPT(tiny_gpt_config)


@pytest.fixture
def tiny_data_dir(tmp_path):
    """A data/ tree with two datasets shaped like the real .bin files.

    Layout mirrors what train.py expects:
        <tmp>/data/wikitext/{train,val}.bin
        <tmp>/data/openwebtext/{train,val}.bin
    Token ids stay below the tiny vocab size so a tiny GPT can consume them.
    """
    rng = np.random.default_rng(0)
    data_dir = tmp_path / "data"
    for name, n_train in (("wikitext", 20_000), ("openwebtext", 50_000)):
        d = data_dir / name
        d.mkdir(parents=True)
        for split, n in (("train", n_train), ("val", 2_000)):
            arr = rng.integers(0, TINY_EOT, size=n, dtype=np.uint16)
            # Sprinkle document separators so doc-boundary logic is exercised.
            arr[rng.integers(0, n, size=max(1, n // 200))] = TINY_EOT
            arr.tofile(d / f"{split}.bin")
    return data_dir

