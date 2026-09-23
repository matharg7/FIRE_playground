"""T3.2-T3.4: RigL on a pretrained HF model inside the continual trainer."""

import os

import pytest
import torch

import config
import sparse_utils
import train
from sparsimony.utils import get_mask

SMOL = "HuggingFaceTB/SmolLM2-135M"
SMOL_LAYERS = 30


def cfg_with(**kw):
    return config.get_config([], **kw)


def expected_sparse_modules(target_set=None, n_layers=SMOL_LAYERS):
    """How many tensors a target set masks, derived rather than hardcoded:
    the count changes with --sparse_targets (7, 3 or 2 Linears per block)."""
    target_set = target_set or sparse_utils.DEFAULT_TARGETS
    return len(sparse_utils.TARGET_SETS[target_set]) * n_layers


def test_rigl_config_validation():
    assert cfg_with(sparsifier="rigl", sparsity=0.05).sparsity == 0.05
    for bad in ({"sparsity": 0.0}, {"sparsity": 1.0}, {"sparse_distribution": "x"},
                {"compile": True}):
        with pytest.raises(ValueError):
            cfg_with(sparsifier="rigl", **bad)


def test_total_run_steps_sums_cumulative_tasks():
    tasks = {name: ([0] * n, [], []) for name, n in (("a", 10), ("b", 6), ("c", 4))}
    cfg = cfg_with(batch_size=4, gradient_accumulation_steps=1, epochs_per_task=1)
    # cumulative sizes 10, 16, 20 -> batches 3, 4, 5
    assert train.total_run_steps(cfg, tasks) == 3 + 4 + 5


# --- T3.2: initial magnitude prune of the pretrained weights -----------------------

def _prepared(sparsity, distribution):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(SMOL, dtype=torch.float32).cuda()
    cfg = cfg_with(sparsifier="rigl", sparsity=sparsity, sparse_distribution=distribution)
    opt = train.build_optimizer(cfg, model)
    return model, sparse_utils.build_sparsifier(cfg, model, opt, total_steps=100)


@pytest.mark.gpu
@pytest.mark.parametrize("sparsity", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("distribution", ["erk", "uniform"])
def test_prepare_prunes_to_target_sparsity(sparsity, distribution):
    model, sp = _prepared(sparsity, distribution)
    stats = sparse_utils.get_sparsity_stats(model)
    assert stats["num_sparse_modules"] == expected_sparse_modules()
    assert stats["mask_sparsity"] == pytest.approx(sparsity, abs=1e-3)
    assert stats["weight_sparsity"] >= stats["mask_sparsity"] - 1e-6


@pytest.mark.gpu
def test_prepare_prunes_smallest_magnitudes_and_leaves_other_params_alone():
    from transformers import AutoModelForCausalLM
    model, sp = _prepared(0.1, "uniform")
    for cfg in sp.groups:
        mask = get_mask(cfg["module"], cfg["tensor_name"])
        w = cfg["module"].parametrizations.weight.original.detach().abs()
        assert 1 - mask.float().mean().item() == pytest.approx(0.1, abs=1e-3)
        assert w[~mask].max() <= w[mask].min() + 1e-12, cfg["tensor_fqn"]
    ref = AutoModelForCausalLM.from_pretrained(SMOL, dtype=torch.float32)
    assert torch.equal(model.model.embed_tokens.weight.cpu(), ref.model.embed_tokens.weight)
    assert torch.equal(model.model.norm.weight.cpu(), ref.model.norm.weight)


# --- T3.3/T3.4: a RigL run across task boundaries ----------------------------------

@pytest.fixture(scope="module")
def rigl_run(tmp_path_factory, trace_data_dir):
    root = trace_data_dir.split("/TRACE-Benchmark/")[0]
    out_root = tmp_path_factory.mktemp("rigl")
    cfg = cfg_with(model=SMOL, data_dir=root, subset=500, tasks="C-STANCE,FOMC",
                   max_train_per_task=16, max_eval_per_task=8, epochs_per_task=4,
                   batch_size=4, gradient_accumulation_steps=1, learning_rate=1e-4,
                   warmup_ratio=0.0, max_prompt_len=256, max_ans_len=32, num_workers=0,
                   eval_batch_size=8, log_interval=4, wandb_mode="disabled",
                   out_root=str(out_root), run_name="rigl", sparsifier="rigl",
                   sparsity=0.1, num_mask_updates=8, t_end_ratio=0.8)
    return cfg, train.main(cfg)


@pytest.mark.gpu
def test_schedule_spans_all_tasks(rigl_run):
    cfg, out = rigl_run
    assert out["total_steps_planned"] == out["global_steps"] == 16 + 32
    sp = out["sparsifier"]
    assert sp._step_count == 48
    # t_end = 38, delta_t = 4: updates at 4, 8, ..., 36 -> both tasks see updates
    assert sp.scheduler.t_end == 38 and sp.scheduler.delta_t == 4


@pytest.mark.gpu
def test_sparsity_exact_at_every_task_boundary(rigl_run):
    _, out = rigl_run
    for t in out["timings"]:
        assert t["dst/mask_sparsity"] == pytest.approx(0.1, abs=1e-3)
        assert t["dst/weight_sparsity"] >= t["dst/mask_sparsity"] - 1e-6


@pytest.mark.gpu
def test_topology_actually_changes(rigl_run):
    _, out = rigl_run
    itop = [t["dst/itop_rate"] for t in out["timings"]]
    assert itop[0] > 0.9 + 1e-4, "RigL should have regrown some pruned weights"
    assert itop[1] >= itop[0]


@pytest.mark.gpu
def test_adamw_moments_are_zero_for_masked_weights(rigl_run):
    _, out = rigl_run
    sp = out["sparsifier"]
    checked = 0
    for cfg in sp.groups:
        mask = get_mask(cfg["module"], cfg["tensor_name"])
        state = sp.optimizer.state[cfg["module"].parametrizations.weight.original]
        assert (state["exp_avg"][~mask] == 0).all()
        assert (state["exp_avg_sq"][~mask] == 0).all()
        checked += 1
    assert checked == expected_sparse_modules()


@pytest.mark.gpu
def test_loss_falls_and_scores_recorded(rigl_run):
    _, out = rigl_run
    for h in out["histories"]:
        assert sum(h[-4:]) / 4 < sum(h[:4]) / 4
    assert out["scores"].R[1, 0] == out["scores"].R[1, 0]  # not NaN


@pytest.mark.gpu
def test_checkpoint_is_a_plain_hf_model_matching_the_sparse_one(rigl_run):
    from torch.nn.utils import parametrize
    from transformers import AutoModelForCausalLM
    _, out = rigl_run
    path = os.path.join(out["out_dir"], "checkpoint_task1")
    plain = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).cuda().eval()
    assert not any("parametrizations" in k for k in plain.state_dict())
    masks = torch.load(os.path.join(path, "masks.pt"))
    assert len(masks) == expected_sparse_modules()
    params = dict(plain.named_parameters())
    for name, mask in masks.items():
        assert (params[name][~mask.cuda()] == 0).all(), name

    sparse = out["model"].eval()
    ids = torch.randint(0, 1000, (2, 32), device="cuda")
    with torch.no_grad(), parametrize.cached():
        a = sparse(input_ids=ids).logits
    with torch.no_grad():
        b = plain(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-4)


# --- the delta_t / per-task-window mismatch (2026-09-23) --------------------

def test_cadence_check_rejects_a_delta_t_that_starves_short_tasks():
    """delta_t comes from the whole run, but per_task restarts the cosine over
    each task's own steps. A coarse delta_t then leaves the short early tasks
    with ~1 topology update, which showed up only as a rising pruning ratio.
    """
    # TRACE at effective batch 128: C-STANCE 195 steps, 20Minuten 2187.
    per_task = [195, 234, 820, 781, 585, 1171, 1367, 2187]

    starved = cfg_with(sparsifier="rigl", drop_fraction_schedule="per_task",
                       num_mask_updates=60, t_end_ratio=0.8, min_updates_per_task=10)
    sched = sparse_utils.sparsifier_schedule(starved, sum(per_task))
    assert sched["delta_t"] == 97
    counts = sparse_utils.updates_per_task(starved, per_task, sched["delta_t"])
    assert counts[0] == 1 and counts[1] == 1, counts
    with pytest.raises(ValueError, match="too coarse"):
        sparse_utils.check_update_cadence(starved, per_task, sched)

    ok = cfg_with(sparsifier="rigl", drop_fraction_schedule="per_task",
                  num_mask_updates=600, t_end_ratio=0.8, min_updates_per_task=10)
    sched = sparse_utils.sparsifier_schedule(ok, sum(per_task))
    assert sched["delta_t"] == 9
    counts = sparse_utils.updates_per_task(ok, per_task, sched["delta_t"])
    assert min(counts) >= 10, counts
    sparse_utils.check_update_cadence(ok, per_task, sched)   # must not raise


def test_cadence_check_is_skipped_for_dense():
    cfg = cfg_with(sparsifier="dense", min_updates_per_task=10)
    sched = sparse_utils.sparsifier_schedule(cfg, 7340)
    sparse_utils.check_update_cadence(cfg, [195, 2187], sched)   # no raise


def test_sweep_default_gives_every_task_enough_updates():
    """The shipped default must pass its own check."""
    per_task = [195, 234, 820, 781, 585, 1171, 1367, 2187]
    cfg = cfg_with(sparsifier="rigl", drop_fraction_schedule="per_task")
    sched = sparse_utils.sparsifier_schedule(cfg, sum(per_task))
    sparse_utils.check_update_cadence(cfg, per_task, sched)
