"""T2.1/T2.2: src/train.py -- schedule, loss, and a 2-task GPU smoke run."""

import json
import math
import os

import pytest
import torch

import config
import train


def cfg_with(**kw):
    return config.get_config([], **kw)


# --- config -----------------------------------------------------------------------

def test_defaults_validate_and_list_trace_tasks():
    cfg = cfg_with()
    assert cfg.task_list()[0] == "C-STANCE" and len(cfg.task_list()) == 8


@pytest.mark.parametrize("bad", [{"tasks": "Nope"}, {"subset": 123}, {"eval_split": "train"},
                                 {"wandb_mode": "x"}, {"save_checkpoint": "sometimes"}])
def test_bad_config_is_rejected(bad):
    with pytest.raises(ValueError):
        cfg_with(**bad)


def test_cli_flags_accept_both_spellings():
    cfg = config.get_config(["--max-prompt-len", "256", "--reset_optimizer", "false"])
    assert cfg.max_prompt_len == 256 and cfg.reset_optimizer is False


# --- schedule ---------------------------------------------------------------------

def test_lr_warms_up_then_decays_to_floor():
    cfg = cfg_with(learning_rate=1e-4, warmup_ratio=0.1, min_lr_ratio=0.1,
                   lr_schedule='cosine')
    lrs = [train.get_lr(cfg, s, 100) for s in range(100)]
    assert lrs[0] == pytest.approx(1e-5) and lrs[9] == pytest.approx(1e-4)
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))
    assert lrs[-1] == pytest.approx(1e-5, rel=0.05)


def test_constant_schedule_is_flat_like_trace():
    """TRACE uses get_constant_schedule_with_warmup with 0 warmup steps, so the
    learning rate never moves. That is the default here."""
    cfg = cfg_with(learning_rate=1e-5, warmup_ratio=0.0, lr_schedule='constant')
    lrs = [train.get_lr(cfg, s, 100) for s in range(100)]
    assert all(lr == pytest.approx(1e-5) for lr in lrs)
    assert cfg_with().lr_schedule == 'constant'

    # With warmup it ramps, then holds flat rather than decaying.
    warm = cfg_with(learning_rate=1e-5, warmup_ratio=0.1, lr_schedule='constant')
    lrs = [train.get_lr(warm, s, 100) for s in range(100)]
    assert lrs[0] == pytest.approx(1e-6) and lrs[9] == pytest.approx(1e-5)
    assert all(lr == pytest.approx(1e-5) for lr in lrs[9:])


def test_effective_batch_matches_trace():
    """TRACE: per_device 2 x accum 8 x 8 GPUs = 128. One GPU, same product."""
    cfg = cfg_with()
    assert cfg.batch_size * cfg.gradient_accumulation_steps == 128


def test_steps_for_task_counts_accumulation_epochs_and_cap():
    cfg = cfg_with(gradient_accumulation_steps=4, epochs_per_task=2)
    assert train.steps_for_task(cfg, 10) == 5
    cfg = cfg_with(gradient_accumulation_steps=4, epochs_per_task=2, max_steps_per_task=3)
    assert train.steps_for_task(cfg, 10) == 3


def test_position_ids_start_at_first_real_token():
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert train.position_ids_for(mask).tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]


# --- GPU ----------------------------------------------------------------------------

SMOL = "HuggingFaceTB/SmolLM2-135M"


@pytest.mark.gpu
def test_loss_matches_hf_and_per_task_sums_add_up(trace_data_dir):
    import data
    from transformers import AutoModelForCausalLM
    root = trace_data_dir.split("/TRACE-Benchmark/")[0]
    tasks = data.load_tasks(root, tasks=["C-STANCE", "FOMC"], subset=500, max_train=8, seed=0)
    tok = data.load_tokenizer(SMOL)
    loader = data.train_loader(data.cumulative_train(tasks, 1), tok, 16, 256, 32, seed=0,
                               num_workers=0)
    batch = next(iter(loader))
    model = AutoModelForCausalLM.from_pretrained(SMOL, dtype=torch.float32).cuda()
    loss, sums, counts, _ = train.loss_and_task_stats(model, batch, "cuda",
                                                      torch.autocast("cuda", enabled=False), 2)
    ref = model(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(),
                position_ids=train.position_ids_for(batch["attention_mask"].cuda()),
                labels=batch["labels"].cuda()).loss
    assert loss.item() == pytest.approx(ref.item(), rel=1e-4)
    assert (sums.sum() / counts.sum()).item() == pytest.approx(loss.item(), rel=1e-4)
    assert counts.sum().item() == (batch["labels"][:, 1:] != -100).sum().item()


@pytest.mark.gpu
def test_two_task_smoke_run(tmp_path, trace_data_dir):
    root = trace_data_dir.split("/TRACE-Benchmark/")[0]
    cfg = cfg_with(model=SMOL, data_dir=root, subset=500, tasks="C-STANCE,FOMC",
                   max_train_per_task=16, max_eval_per_task=8, epochs_per_task=4,
                   batch_size=4, gradient_accumulation_steps=1, learning_rate=1e-4,
                   warmup_ratio=0.0, max_prompt_len=256, max_ans_len=32, num_workers=0,
                   eval_batch_size=8, log_interval=4, wandb_mode="disabled",
                   eval_lookahead=True, out_root=str(tmp_path), run_name="smoke")
    out = train.main(cfg)
    run = tmp_path / "smoke"

    # zero-shot row: every task scored before training, plus eval losses
    scores = json.loads((run / "scores.json").read_text())
    assert set(scores["baselines"]) == {"C-STANCE", "FOMC"}
    assert set(scores["losses"]) == {"-1", "0", "1"}
    assert scores["losses"]["-1"].keys() == {"C-STANCE", "FOMC"}
    assert (run / "eval" / "zero_shot_1_FOMC.json").exists()
    assert math.isfinite(out["fwt"])

    # task 0 trains on 16 examples, task 1 on the cumulative 32
    assert [len(h) for h in out["histories"]] == [16, 32]
    for h in out["histories"]:
        assert sum(h[-4:]) / 4 < sum(h[:4]) / 4, "loss should fall within each task"

    R = json.loads((run / "scores.json").read_text())["R"]
    assert R[0][0] is not None and R[0][1] is not None       # lookahead scores task 1 too
    assert R[1][0] is not None and R[1][1] is not None
    assert all(0 <= x <= 1 for row in R for x in row if x is not None)
    assert math.isfinite(out["op"]) and math.isfinite(out["bwt"])

    summary = json.loads((run / "summary.json").read_text())
    assert summary["op"] == pytest.approx((R[1][0] + R[1][1]) / 2)
    assert summary["bwt"] == pytest.approx(R[1][0] - R[0][0])
    assert sorted(os.listdir(run / "eval")) == [
        "step0_0_C-STANCE.json", "step0_1_FOMC.json", "step1_0_C-STANCE.json",
        "step1_1_FOMC.json", "zero_shot_0_C-STANCE.json", "zero_shot_1_FOMC.json"]
    assert (run / "checkpoint_task1" / "config.json").exists()
    assert not (run / "checkpoint_task0").exists()
    assert (run / "DONE").exists() and "run_name=smoke" in (run / "DONE").read_text()
