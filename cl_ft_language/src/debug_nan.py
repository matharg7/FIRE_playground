"""Find the first non-finite tensor in a Qwen2.5-0.5B run on TRACE task 0.

All four sweep runs went NaN at the same step regardless of sparsity, so this
walks the dense path step by step and reports which of logits / loss / grads /
params goes non-finite first, and under which dtype.

  python debug_nan.py --dtype bfloat16 --steps 120
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data  # noqa: E402
import train  # noqa: E402
from config import get_config  # noqa: E402


def finite(t):
    return bool(torch.isfinite(t).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--fused", default="true")
    args = ap.parse_args()

    cfg = get_config([
        f"--model=Qwen/Qwen2.5-0.5B", "--subset=5000", "--tasks=C-STANCE",
        "--batch_size=8", "--gradient_accumulation_steps=1",
        "--max_prompt_len=1024", "--max_ans_len=512",
        f"--learning_rate={args.lr}", "--weight_decay=0.0",
        "--length_grouped=True", f"--dtype={args.dtype}",
        "--wandb_mode=disabled", "--sparsifier=dense",
        f"--data_dir={os.environ['TRACE_DATA_DIR']}",
    ])
    device = "cuda"
    ptdtype = getattr(torch, cfg.dtype)
    ctx = (torch.autocast(device_type="cuda", dtype=ptdtype)
           if cfg.dtype != "float32" else torch.enable_grad())

    tok = data.load_tokenizer(cfg.model)
    td = data.load_tasks(cfg.data_dir, tasks=["C-STANCE"], subset=5000)
    ds = data.cumulative_train(td, 0)
    dl = data.train_loader(ds, tok, cfg.batch_size, cfg.max_prompt_len,
                           cfg.max_ans_len, seed=0, num_workers=0, length_grouped=True)

    model = train.build_model(cfg, device)
    opt = train.build_optimizer(cfg, model)
    if args.fused.lower() != "true":
        opt = torch.optim.AdamW(opt.param_groups, lr=cfg.learning_rate, fused=False)

    print(f"dtype={cfg.dtype} fused={args.fused} lr={cfg.learning_rate}")
    print(f"{'step':>5} {'seqlen':>7} {'labels':>7} {'loss':>10} {'gnorm':>11} "
          f"{'logit_absmax':>13}  status")

    warmup = max(1, int(cfg.warmup_ratio * 3125))
    it = iter(dl)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            break
        lr = cfg.learning_rate * (step + 1) / warmup if step < warmup else cfg.learning_rate
        for g in opt.param_groups:
            g["lr"] = lr

        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with ctx:
            logits = model(input_ids=ids, attention_mask=am,
                           position_ids=train.position_ids_for(am)).logits
        targets = labels[:, 1:]
        tl = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), targets.flatten(),
                             ignore_index=-100, reduction="none").view_as(targets)
        mask = (targets != -100).float()
        loss = (tl * mask).sum() / mask.sum().clamp(min=1)

        problems = []
        if not finite(logits):
            problems.append("LOGITS non-finite")
        if not finite(loss):
            problems.append("LOSS non-finite")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        bad_grads = [n for n, p in model.named_parameters()
                     if p.grad is not None and not finite(p.grad)]
        if bad_grads:
            problems.append(f"{len(bad_grads)} non-finite GRADS, first={bad_grads[0]}")

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not finite(gnorm):
            problems.append(f"GRAD_NORM={gnorm}")
        opt.step()

        bad_params = [n for n, p in model.named_parameters() if not finite(p)]
        if bad_params:
            problems.append(f"{len(bad_params)} non-finite PARAMS, first={bad_params[0]}")

        n_lab = int(mask.sum().item())
        status = "; ".join(problems) if problems else "ok"
        if problems or step % 10 == 0 or step > 60:
            print(f"{step+1:>5} {ids.shape[1]:>7} {n_lab:>7} {loss.item():>10.4f} "
                  f"{float(gnorm):>11.4f} {logits.abs().max().item():>13.1f}  {status}")
        if problems:
            print("\n>>> FIRST FAILURE AT STEP", step + 1)
            print("    min ctx len :", int(am.sum(1).min().item()))
            print("    min row labs:", int((targets != -100).sum(1).min().item()))
            print("    logits absmax:", logits.abs().max().item())
            print("    loss:", loss.item(), " gnorm:", float(gnorm))
            break
    else:
        print(f"\nno failure in {args.steps} steps")


if __name__ == "__main__":
    main()
