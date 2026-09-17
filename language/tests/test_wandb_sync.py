"""T2.5/T2.6: offline W&B runs, DONE markers, and the login-node sync script."""
import os
import stat
import subprocess

import pytest

from helpers import make_tiny_data, run_train

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYNC_SCRIPT = os.path.join(REPO_ROOT, "bash_scripts", "sync_wandb.sh")


# ---------------------------------------------------------------------------
# The DONE marker written by train_sparse.py
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def offline_run(tmp_path_factory):
    """A real (tiny) run with offline W&B enabled."""
    tmp_path = tmp_path_factory.mktemp("offline")
    data = make_tiny_data(tmp_path)
    wandb_dir = tmp_path / "wandb"
    _, stdout, _ = run_train(
        tmp_path, data,
        extra_args=["--wandb_log=True", "--wandb_mode=offline",
                    f"--wandb_dir={wandb_dir}", "--save_checkpoint=False"],
    )
    from helpers import stage_run
    return stage_run(tmp_path, data) / "output", wandb_dir, stdout


class TestOfflineRun:
    def test_offline_directory_created(self, offline_run):
        _, wandb_dir, _ = offline_run
        assert wandb_dir.is_dir()
        # wandb.init(dir=X) puts runs in X/wandb/offline-run-*
        assert any(p.name.startswith("offline-run-") for p in wandb_dir.rglob("offline-run-*"))

    def test_done_marker_written(self, offline_run):
        out, _, _ = offline_run
        run_dir = next(out.iterdir())
        assert (run_dir / "DONE").is_file()

    def test_done_marker_points_at_the_offline_run(self, offline_run):
        out, _, _ = offline_run
        fields = dict(
            line.split("=", 1)
            for line in (next(out.iterdir()) / "DONE").read_text().splitlines() if line
        )
        assert fields["wandb_mode"] == "offline"
        assert os.path.isdir(fields["wandb_run_dir"])
        assert fields["run_name"]

    def test_nothing_uploaded(self, offline_run):
        """Offline mode must not contact wandb.ai; compute nodes have no internet."""
        _, _, stdout = offline_run
        assert "offline" in stdout.lower()


class TestNoMarkerOnFailure:
    def test_failed_run_has_no_done_marker(self, tmp_path):
        """A crashed run must not be synced, so it must not be marked DONE."""
        data = make_tiny_data(tmp_path)
        with pytest.raises(AssertionError):
            run_train(tmp_path, data, extra_args=["--c0_dataset=does_not_exist"])
        from helpers import stage_run
        out = stage_run(tmp_path, data) / "output"
        assert not any(p.name == "DONE" for p in out.rglob("*"))


# ---------------------------------------------------------------------------
# The sync script
# ---------------------------------------------------------------------------

def _make_run(root, name, wandb_root, with_done=True, synced=False, valid_wandb=True):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    wandb_run = wandb_root / f"offline-run-{name}"
    if valid_wandb:
        wandb_run.mkdir(parents=True)
    if with_done:
        (run_dir / "DONE").write_text(
            f"run_name={name}\nout_dir={run_dir}\nwandb_run_dir={wandb_run}\n"
            "wandb_mode=offline\nfinished_at=now\n"
        )
    if synced:
        (run_dir / "SYNCED").write_text("synced_at=earlier\n")
    return run_dir


@pytest.fixture
def stub_wandb(tmp_path):
    """A fake `wandb` that records the directories it was asked to sync."""
    log = tmp_path / "wandb_calls.log"
    stub = tmp_path / "wandb_stub"
    stub.write_text(f'#!/bin/bash\necho "$@" >> "{log}"\nexit 0\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub, log


def _sync_calls(log):
    """How many times the stub was invoked (paths may contain the word 'sync')."""
    if not log.exists():
        return 0
    return len([ln for ln in log.read_text().splitlines() if ln.startswith("sync ")])


def _run_sync(root, stub, extra=()):
    return subprocess.run(
        ["bash", SYNC_SCRIPT, "--root", str(root), *extra],
        env=dict(os.environ, FIRE_SYNC_NO_ENV="1", WANDB_BIN=str(stub)),
        capture_output=True, text=True,
    )


class TestSyncScript:
    def test_syncs_finished_runs(self, tmp_path, stub_wandb):
        stub, log = stub_wandb
        root, wandb_root = tmp_path / "runs", tmp_path / "wandb"
        _make_run(root, "run_a", wandb_root)
        _make_run(root, "run_b", wandb_root)
        proc = _run_sync(root, stub)
        assert proc.returncode == 0, proc.stderr
        assert _sync_calls(log) == 2

    def test_marks_runs_as_synced(self, tmp_path, stub_wandb):
        stub, _ = stub_wandb
        root, wandb_root = tmp_path / "runs", tmp_path / "wandb"
        run = _make_run(root, "run_a", wandb_root)
        _run_sync(root, stub)
        assert (run / "SYNCED").is_file()

    def test_is_idempotent(self, tmp_path, stub_wandb):
        """Running twice must not push the same run twice."""
        stub, log = stub_wandb
        root, wandb_root = tmp_path / "runs", tmp_path / "wandb"
        _make_run(root, "run_a", wandb_root)
        _run_sync(root, stub)
        _run_sync(root, stub)
        assert _sync_calls(log) == 1

    def test_skips_already_synced(self, tmp_path, stub_wandb):
        stub, log = stub_wandb
        root, wandb_root = tmp_path / "runs", tmp_path / "wandb"
        _make_run(root, "run_a", wandb_root, synced=True)
        _run_sync(root, stub)
        assert _sync_calls(log) == 0

    def test_skips_unfinished_runs(self, tmp_path, stub_wandb):
        """No DONE marker means the run crashed or is still going."""
        stub, log = stub_wandb
        root, wandb_root = tmp_path / "runs", tmp_path / "wandb"
        _make_run(root, "run_a", wandb_root, with_done=False)
        _run_sync(root, stub)
        assert _sync_calls(log) == 0

    def test_skips_missing_offline_directory(self, tmp_path, stub_wandb):
        stub, log = stub_wandb
        root, wandb_root = tmp_path / "runs", tmp_path / "wandb"
        _make_run(root, "run_a", wandb_root, valid_wandb=False)
        proc = _run_sync(root, stub)
        assert proc.returncode == 0
        assert "SKIP" in proc.stdout
        assert _sync_calls(log) == 0

    def test_dry_run_changes_nothing(self, tmp_path, stub_wandb):
        stub, log = stub_wandb
        root, wandb_root = tmp_path / "runs", tmp_path / "wandb"
        run = _make_run(root, "run_a", wandb_root)
        proc = _run_sync(root, stub, ["--dry-run"])
        assert "WOULD SYNC" in proc.stdout
        assert not (run / "SYNCED").exists()
        assert _sync_calls(log) == 0

    def test_missing_root_is_not_an_error(self, tmp_path, stub_wandb):
        stub, _ = stub_wandb
        proc = _run_sync(tmp_path / "nope", stub)
        assert proc.returncode == 0
        assert "Nothing to do" in proc.stdout

    def test_reports_failures(self, tmp_path):
        """A failing sync must be visible, not silently counted as success."""
        failing = tmp_path / "failing_wandb"
        failing.write_text("#!/bin/bash\nexit 1\n")
        failing.chmod(failing.stat().st_mode | stat.S_IEXEC)
        root, wandb_root = tmp_path / "runs", tmp_path / "wandb"
        run = _make_run(root, "run_a", wandb_root)
        proc = _run_sync(root, failing)
        assert proc.returncode != 0
        assert not (run / "SYNCED").exists()
