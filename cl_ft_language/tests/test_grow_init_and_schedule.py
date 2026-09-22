"""Migration of the dst-fire-full-reset branch: grow_init and the per-task
drop-fraction schedule (vision/train_st.py + the sparsimony grow_mask patch).

FIRE and full_reset from that branch are deliberately not migrated: the first
TRACE comparison is dense vs RigL only.
"""

import pytest
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

import config
import sparse_utils
import train
from sparsimony.utils import get_mask


def cfg_with(**kw):
    return config.get_config([], **kw)


class TinyDecoder(nn.Module):
    """Two 'decoder blocks' with the HF naming sparse_utils looks for."""

    def __init__(self, dim=16):
        super().__init__()
        layers = []
        for _ in range(2):
            block = nn.Module()
            block.self_attn = nn.Module()
            block.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
            block.self_attn.o_proj = nn.Linear(dim, dim, bias=False)
            layers.append(block)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)
        self.head = nn.Linear(dim, 4)

    def forward(self, x):
        for block in self.model.layers:
            x = torch.relu(block.self_attn.o_proj(block.self_attn.q_proj(x)))
        return self.head(x)


def _sparse_model(**overrides):
    torch.manual_seed(0)
    model = TinyDecoder()
    cfg = cfg_with(sparsifier="rigl", sparsity=0.5, sparse_distribution="uniform",
                   num_mask_updates=8, pruning_ratio=0.3, **overrides)
    # weight_decay=0 (train.py's default): AdamW's decoupled decay would shrink
    # masked weights too, so 'previous' would regrow a decayed value.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=cfg.weight_decay)
    sp = sparse_utils.build_sparsifier(cfg, model, opt, total_steps=40)
    return cfg, model, opt, sp


def _layers(model):
    out = []
    for name, m in model.named_modules():
        if parametrize.is_parametrized(m, "weight"):
            out.append((name, m.parametrizations.weight.original, get_mask(m, "weight")))
    return out


def _train_step(model, opt):
    opt.zero_grad()
    nn.CrossEntropyLoss()(model(torch.randn(8, 16)), torch.randint(0, 4, (8,))).backward()
    opt.step()


def _run_until_mask_update(model, opt, sp, limit=60):
    for _ in range(limit):
        _train_step(model, opt)
        before = {n: (w.detach().clone(), m.clone()) for n, w, m in _layers(model)}
        if sp.step():
            return before
    raise AssertionError("no mask update happened")


# --- grow_init ----------------------------------------------------------------

@pytest.mark.parametrize("grow_init", ["zero", "previous"])
def test_grow_init(grow_init):
    _, model, opt, sp = _sparse_model(grow_init=grow_init)
    assert sp.grow_init == grow_init
    before = _run_until_mask_update(model, opt, sp)
    n_grown = 0
    for name, w, mask in _layers(model):
        w_before, mask_before = before[name]
        grown = mask & ~mask_before
        n_grown += int(grown.sum())
        if grow_init == "zero":
            assert (w[grown] == 0).all()
        else:
            # a mask update leaves the stored weights untouched
            assert torch.equal(w, w_before)
    assert n_grown > 0


def test_previous_regrows_nonzero_values():
    _, model, opt, sp = _sparse_model(grow_init="previous")
    before = _run_until_mask_update(model, opt, sp)
    grown = torch.cat([w[m & ~before[n][1]] for n, w, m in _layers(model)])
    assert grown.numel() > 0 and grown.abs().sum() > 0


def test_previous_keeps_the_pretrained_value_of_never_active_weights():
    # The point of 'previous' for a pretrained model: a weight that was pruned
    # at prepare() and later regrown comes back with its pretrained value.
    torch.manual_seed(0)
    model = TinyDecoder()
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    cfg = cfg_with(sparsifier="rigl", sparsity=0.5, sparse_distribution="uniform",
                   num_mask_updates=8, grow_init="previous")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    sp = sparse_utils.build_sparsifier(cfg, model, opt, total_steps=40)
    name, w, mask = _layers(model)[0]
    pruned_at_start = ~mask.clone()
    before = _run_until_mask_update(model, opt, sp)
    w, mask = [(x, m) for n, x, m in _layers(model) if n == name][0]
    regrown = mask & pruned_at_start
    assert regrown.any()
    init_w = initial[f"{name}.weight"]
    assert torch.equal(w[regrown], init_w[regrown])


def test_weight_decay_shrinks_masked_weights_so_previous_is_not_exact():
    """Why train.py keeps weight_decay=0 with grow_init='previous': AdamW's
    decoupled decay shrinks masked weights too, which have no gradient."""
    torch.manual_seed(0)
    model = TinyDecoder()
    cfg = cfg_with(sparsifier="rigl", sparsity=0.5, sparse_distribution="uniform",
                   num_mask_updates=8, grow_init="previous", weight_decay=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=cfg.weight_decay)
    sp = sparse_utils.build_sparsifier(cfg, model, opt, total_steps=40)
    name, w, mask = _layers(model)[0]
    masked = ~mask.clone()
    before = w[masked].clone()
    for _ in range(5):
        _train_step(model, opt)
    after = _layers(model)[0][1][masked]
    assert not torch.equal(before, after)
    assert (after.abs() < before.abs() + 1e-12).all()   # shrunk, never grown


def test_grow_init_is_validated():
    with pytest.raises(ValueError):
        cfg_with(sparsifier="rigl", grow_init="something_else")
    with pytest.raises(ValueError):
        cfg_with(sparsifier="dense", grow_init="previous")  # dense never regrows
    _, model, opt, sp = _sparse_model()
    sp.grow_init = "something_else"
    with pytest.raises(ValueError):
        _run_until_mask_update(model, opt, sp)


# --- drop-fraction schedule ---------------------------------------------------

def test_schedules_are_validated_and_built():
    from sparsimony.schedulers.base import ConstantScheduler, CosineDecayScheduler
    with pytest.raises(ValueError):
        cfg_with(sparsifier="rigl", drop_fraction_schedule="nope")
    _, _, _, cosine = _sparse_model(drop_fraction_schedule="per_task")
    _, _, _, const = _sparse_model(drop_fraction_schedule="constant")
    assert isinstance(cosine.scheduler, CosineDecayScheduler)
    assert isinstance(const.scheduler, ConstantScheduler)


def test_per_task_restart_retargets_t_end_and_resets_the_step_count():
    cfg, model, opt, sp = _sparse_model(drop_fraction_schedule="per_task", t_end_ratio=0.8)
    global_t_end = sp.scheduler.t_end
    for _ in range(5):
        _train_step(model, opt)
        sp.step()
    assert sp._step_count == 5
    t_end = sparse_utils.restart_drop_fraction_schedule(cfg, sp, task_steps=100)
    assert t_end == 80 != global_t_end          # 80% within the task
    assert sp._step_count == 0                  # cosine warmed back up
    assert sp.scheduler.delta_t == global_t_end // cfg.num_mask_updates  # cadence unchanged


def test_t_end_ratio_one_reproduces_the_vision_branch():
    # vision/train_st.py set t_end to the full task length.
    cfg, _, _, sp = _sparse_model(drop_fraction_schedule="per_task", t_end_ratio=1.0)
    assert sparse_utils.restart_drop_fraction_schedule(cfg, sp, task_steps=100) == 100


def test_global_schedule_is_not_restarted():
    cfg, model, opt, sp = _sparse_model(drop_fraction_schedule="global")
    t_end = sp.scheduler.t_end
    for _ in range(3):
        _train_step(model, opt)
        sp.step()
    assert sparse_utils.restart_drop_fraction_schedule(cfg, sp, task_steps=100) is None
    assert sp.scheduler.t_end == t_end and sp._step_count == 3


def test_per_task_keeps_updating_in_the_final_task_where_global_stops():
    """The reason for per_task: with the global schedule t_end falls inside the
    last task, so its topology is frozen for most of it."""
    tasks = {name: ([0] * n, [], []) for name, n in (("a", 100), ("b", 100), ("c", 100))}
    cfg = cfg_with(sparsifier="rigl", batch_size=1, gradient_accumulation_steps=1,
                   epochs_per_task=1, num_mask_updates=10, t_end_ratio=0.8,
                   drop_fraction_schedule="per_task")
    per_task_steps = [len(train.data.cumulative_train(tasks, t)) for t in range(3)]
    total = train.total_run_steps(cfg, tasks)          # 100 + 200 + 300 = 600
    assert per_task_steps == [100, 200, 300] and total == 600

    global_t_end = sparse_utils.sparsifier_schedule(cfg, total)["t_end"]   # 480
    steps_before_last_task = sum(per_task_steps[:2])                      # 300
    assert global_t_end - steps_before_last_task < per_task_steps[2]      # freezes mid-task
    # per_task instead gives the last task its own horizon
    _, _, _, sp = _sparse_model(drop_fraction_schedule="per_task", t_end_ratio=0.8)
    assert sparse_utils.restart_drop_fraction_schedule(cfg, sp, per_task_steps[2]) == 240
