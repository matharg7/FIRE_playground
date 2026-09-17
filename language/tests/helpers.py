"""Run train_sparse.py end to end on CPU, in seconds.

It is a script, not a library, so testing the script itself means running it. stage_run() builds a throwaway directory that looks
like language/ (the scripts read data/ and output/ by relative path).
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np

LANGUAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The real separator is 50256, but the tiny test model has a 128-token vocab,
# so the fixtures use the last in-range id to play the same role.
TINY_VOCAB = 128
TINY_EOT = TINY_VOCAB - 1

# Tiny but structurally real:
#   tokens_per_iter = grad_accum * world * batch * block = 1 * 1 * 4 * 32 = 128
#   chunk 0 tokens  = replay * len * ratio = 1 * 20_000 * 1.0 -> 156 iters
# Comfortably above the 20-iteration floor where eval_interval would hit zero.
BASE_ARGS = [
    "--device=cpu",
    "--dtype=float32",
    "--compile=False",
    "--wandb_log=False",
    "--save_checkpoint=False",
    "--save_checkpoint_periodically=False",
    "--n_layer=2",
    "--n_head=2",
    "--n_embd=64",
    "--block_size=32",
    "--batch_size=4",
    "--gradient_accumulation_steps=1",
    "--eval_iters=2",
    "--c0_dataset=wikitext",
    "--c0_subset_ratio=1.0",
    "--c0_data_replay_ratio=1",
    "--c1_dataset=openwebtext",
    "--c1_subset_ratio=0.4",
    "--c1_data_replay_ratio=1",
]

# A small vocabulary is what makes the tiny model genuinely tiny.
VOCAB_ARG = "--vocab_size=128"

EVAL_RE = re.compile(
    r"EVAL chunk (\d+) global_iter (\d+) local_iter (\d+) "
    r"train ([-\d.naif]+) val ([-\d.naif]+)"
)

SPARSE_RE = re.compile(r"^SPARSE (.+)$", re.MULTILINE)


def parse_sparse(stdout):
    """Pull the SPARSE lines train_sparse.py prints at each eval into dicts."""
    out = []
    for line in SPARSE_RE.findall(stdout):
        parts = line.split()
        out.append({parts[i]: float(parts[i + 1]) for i in range(0, len(parts) - 1, 2)})
    return out


_LINKED = ("train_sparse.py", "config_sparse.py", "sparse_utils.py",
           "model.py", "interventions")


def make_tiny_data(tmp_path, sizes=(("wikitext", 20_000), ("openwebtext", 50_000)), seed=0):
    """Write .bin files shaped like the real ones: flat uint16 token ids."""
    rng = np.random.default_rng(seed)
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    for name, n_train in sizes:
        d = data_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for split, n in (("train", n_train), ("val", 2_000)):
            arr = rng.integers(0, TINY_EOT, size=n, dtype=np.uint16)
            # Sprinkle document separators so doc-boundary behaviour is exercised.
            arr[rng.integers(0, n, size=max(1, n // 200))] = TINY_EOT
            arr.tofile(d / f"{split}.bin")
    return data_dir


def stage_run(tmp_path, data_dir):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in _LINKED:
        src = os.path.join(LANGUAGE_DIR, name)
        link = run_dir / name
        if os.path.exists(src) and not link.exists():
            os.symlink(src, link)
    data_link = run_dir / "data"
    if not data_link.exists():
        # Absolute, or a relative target would resolve inside run_dir and loop.
        os.symlink(os.path.abspath(data_dir), data_link)
    return run_dir


def ddp_launcher(nproc):
    """How to start nproc ranks here.

    On DRAC, SLURM binds one GPU per task, so torchrun cannot see them all and
    srun is the launcher. --gpus-per-task gives each rank its own GPU and
    --gpu-bind=none keeps both visible, without which NCCL fails with
    "invalid device ordinal". train_sparse.adopt_slurm_env() maps SLURM's
    variables onto the ones torch.distributed expects.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return ["srun", f"--jobid={job_id}", "--overlap", f"--ntasks={nproc}",
                "--gpus-per-task=1", "--gpu-bind=none", "--cpus-per-task=4"]
    return ["torchrun", "--standalone", f"--nproc_per_node={nproc}"]


def shared_tmp(name):
    """A directory every node can see.

    pytest's tmp_path lives in /tmp, which is node-local: an srun step on a
    compute node cannot read files staged on the login node. Multi-rank tests
    therefore stage on $SCRATCH.
    """
    root = os.environ.get("SCRATCH")
    if not root:
        return None
    path = os.path.join(root, "fire_pytest", name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return pathlib.Path(path)


def run_ddp(tmp_path, data_dir, nproc=2, script="train_sparse.py",
            extra_args=(), timeout=1800):
    """Run a script on nproc ranks.

    Returns (evals, stdout, run_dir). run_dir is where the script actually ran,
    which is on $SCRATCH rather than the caller's tmp_path (see shared_tmp).
    """
    shared = shared_tmp(f"ddp_{os.path.basename(str(tmp_path))}")
    if shared is not None:
        data_dir = make_tiny_data(shared)
        tmp_path = shared
    run_dir = stage_run(tmp_path, data_dir)
    args = [a for a in BASE_ARGS if not a.startswith("--device=")]
    args += [VOCAB_ARG, "--device=cuda", "--gradient_accumulation_steps=2",
             "--data_dir=data", "--out_root=output"]
    cmd = [*ddp_launcher(nproc), sys.executable, script, *args, *extra_args]
    env = dict(os.environ, PYTHONUNBUFFERED="1", WANDB_MODE="offline",
               OMP_NUM_THREADS="4")
    proc = subprocess.run(cmd, cwd=run_dir, env=env, timeout=timeout,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"ddp {script} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr ---\n{proc.stderr[-3000:]}"
        )
    evals = [
        {"chunk": int(m[0]), "global_iter": int(m[1]), "local_iter": int(m[2]),
         "train": float(m[3]), "val": float(m[4])}
        for m in EVAL_RE.findall(proc.stdout)
    ]
    return evals, proc.stdout, run_dir


def run_train(tmp_path, data_dir, script="train_sparse.py", extra_args=(), timeout=900):
    """Run a training script with the tiny config. Returns (evals, stdout, stderr)."""
    run_dir = stage_run(tmp_path, data_dir)
    args = list(BASE_ARGS) + [VOCAB_ARG, "--data_dir=data", "--out_root=output"]
    cmd = [sys.executable, script, *args, *extra_args]
    # Pin threads: torch otherwise spawns one per core (192 on rorqual's login
    # node) for a 0.1M-parameter model, and the contention dominates the runtime.
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        WANDB_MODE="offline",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TORCH_NUM_THREADS="1",
    )
    proc = subprocess.run(cmd, cwd=run_dir, env=env, timeout=timeout,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"{script} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr ---\n{proc.stderr[-3000:]}"
        )
    evals = [
        {"chunk": int(m[0]), "global_iter": int(m[1]), "local_iter": int(m[2]),
         "train": float(m[3]), "val": float(m[4])}
        for m in EVAL_RE.findall(proc.stdout)
    ]
    return evals, proc.stdout, proc.stderr
