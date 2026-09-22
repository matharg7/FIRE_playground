"""Shared pytest setup for cl_ft_language.

TRACE's modules import each other as top-level packages (`from metrics import`,
`from utils.data import`), so trace/ goes on sys.path; src/ holds our code.
Models are read from the local HF cache only: compute nodes have no internet.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("trace", "src"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODELS = ["HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B"]
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def pytest_collection_modifyitems(config, items):
    import torch
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device (run inside a GPU job)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def trace_data_dir():
    """The 5000-sample TRACE benchmark, or skip if it has not been downloaded."""
    base = os.environ.get("TRACE_DATA_DIR", os.path.join(
        os.environ.get("SCRATCH", ""), "fire", "data", "trace"))
    path = os.path.join(base, "TRACE-Benchmark", "LLM-CL-Benchmark_5000")
    if not os.path.isdir(path):
        pytest.skip(f"TRACE data not found at {path} (run scripts/build_env.sh download)")
    return path
