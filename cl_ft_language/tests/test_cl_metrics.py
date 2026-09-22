"""T1.4: src/cl_metrics.py -- OP, BWT, FWT on toy score matrices."""

import math

import pytest

from cl_metrics import ScoreMatrix

TASKS = ["a", "b", "c"]


@pytest.fixture
def m():
    # rows: after training task t; lower triangle filled, plus R[0][1] and R[1][2]
    m = ScoreMatrix(TASKS)
    m.record(0, {"a": 0.8, "b": 0.3})
    m.record(1, {"a": 0.6, "b": 0.9, "c": 0.2})
    m.record(2, {"a": 0.5, "b": 0.7, "c": 1.0})
    return m


def test_op_is_mean_of_the_row_up_to_t(m):
    assert m.op(0) == pytest.approx(0.8)
    assert m.op(1) == pytest.approx((0.6 + 0.9) / 2)
    assert m.op() == pytest.approx((0.5 + 0.7 + 1.0) / 3)


def test_bwt_is_mean_drop_from_diagonal(m):
    assert m.bwt(0) == 0.0
    assert m.bwt(1) == pytest.approx(0.6 - 0.8)
    assert m.bwt() == pytest.approx(((0.5 - 0.8) + (0.7 - 0.9)) / 2)


def test_no_forgetting_gives_zero_bwt():
    m = ScoreMatrix(TASKS)
    for t in range(3):
        m.record(t, {task: 0.5 for task in TASKS[: t + 1]})
    assert m.bwt() == 0.0 and m.op() == 0.5


def test_fwt_uses_pre_training_scores_and_baselines(m):
    base = {"a": 0.1, "b": 0.1, "c": 0.1}
    assert m.fwt(base) == pytest.approx(((0.3 - 0.1) + (0.2 - 0.1)) / 2)


def test_recorded_baselines_are_used_by_default(m):
    with pytest.raises(ValueError):
        m.fwt()
    m.record_baselines({"a": 0.1, "b": 0.1, "c": 0.1})
    assert m.fwt() == pytest.approx(((0.3 - 0.1) + (0.2 - 0.1)) / 2)


def test_losses_are_recorded_per_step(m, tmp_path):
    m.record_baselines({"a": 0.1, "b": 0.2, "c": 0.3})
    m.record_losses(-1, {"a": 2.0, "b": 3.0})
    m.record_losses(0, {"a": 1.0})
    path = tmp_path / "R.json"
    m.save(path)
    back = ScoreMatrix.load(path)
    assert back.baselines == {"a": 0.1, "b": 0.2, "c": 0.3}
    assert back.losses == {"-1": {"a": 2.0, "b": 3.0}, "0": {"a": 1.0}}


def test_missing_scores_raise_instead_of_silently_averaging():
    m = ScoreMatrix(TASKS)
    m.record(0, {"a": 0.5})
    m.record(1, {"b": 0.5})
    with pytest.raises(ValueError):
        m.op(1)
    with pytest.raises(ValueError):
        m.fwt({"a": 0, "b": 0, "c": 0})


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        ScoreMatrix(TASKS).record(0, {"zzz": 1.0})


def test_save_load_roundtrip_keeps_nans(m, tmp_path):
    path = tmp_path / "R.json"
    m.save(path)
    back = ScoreMatrix.load(path)
    assert back.tasks == TASKS
    assert math.isnan(back.R[0, 2])
    assert back.op() == m.op() and back.bwt() == m.bwt()


def test_summary(m):
    assert m.summary(1) == {"op": m.op(1), "bwt": m.bwt(1)}
