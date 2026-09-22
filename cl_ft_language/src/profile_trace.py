"""Profiling for the TRACE continual runs (T4.1): what does a full run cost?

Three parts, each writing JSON to <out>/<part>_<model>.json and printing a table:

  data   token-length statistics per task (CPU only): prompt/answer tokens,
         share of prompts truncated at each cap, training tokens per task
  train  training throughput per arm (dense | rigl) and prompt cap, on batches
         drawn from all 8 tasks (the cumulative mix of the last task): real
         and padded tokens/s, step time, peak memory, RigL mask-update cost
  eval   generation seconds per test example per task (worst case: an
         untrained model often runs to the per-task token cap)

    python src/profile_trace.py --part data  --model Qwen/Qwen2.5-0.5B
    python src/profile_trace.py --part train --model Qwen/Qwen2.5-0.5B
    python src/profile_trace.py --part eval  --model Qwen/Qwen2.5-0.5B
"""

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.utils import parametrize

import data
import evaluate
import sparse_utils
import train
from config import get_config


def pct(values, q):
    return float(np.percentile(values, q)) if values else 0.0


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def profile_data(args, task_data, tokenizer):
    caps = [512, 1024, 2048]
    rows = {}
    for task, (train_set, _, test_set) in task_data.items():
        enc = lambda s: len(tokenizer(s, add_special_tokens=False)["input_ids"])
        prompts = [enc(train_set[i]["prompt"]) for i in range(len(train_set))]
        answers = [enc(train_set[i]["answer"]) + 1 for i in range(len(train_set))]  # + EOS
        row = {"n_train": len(train_set), "n_test": len(test_set),
               "prompt_p50": pct(prompts, 50), "prompt_p95": pct(prompts, 95),
               "prompt_max": max(prompts), "answer_p50": pct(answers, 50),
               "answer_p95": pct(answers, 95), "answer_max": max(answers),
               "answer_over_512": float(np.mean([a > args.max_ans_len for a in answers]))}
        for cap in caps:
            row[f"truncated_at_{cap}"] = float(np.mean([p > cap for p in prompts]))
            # tokens actually trained on per epoch with this prompt cap
            row[f"train_tokens_cap{cap}"] = int(sum(min(p, cap) + min(a, args.max_ans_len)
                                                    for p, a in zip(prompts, answers)))
        rows[task] = row
        print(f"{task:<12} n={row['n_train']:>5} prompt p50/p95/max {row['prompt_p50']:>6.0f}"
              f"/{row['prompt_p95']:>6.0f}/{row['prompt_max']:>7} answer p50/p95/max "
              f"{row['answer_p50']:>4.0f}/{row['answer_p95']:>5.0f}/{row['answer_max']:>5} "
              f"trunc@1024 {row['truncated_at_1024']:.1%} trunc@2048 {row['truncated_at_2048']:.1%}",
              flush=True)
    for cap in caps:
        per_task = [rows[t][f"train_tokens_cap{cap}"] for t in task_data]
        cumulative = sum(sum(per_task[: t + 1]) for t in range(len(per_task)))
        rows[f"_run_tokens_cap{cap}"] = {"sequential": sum(per_task), "cumulative": cumulative}
        print(f"prompt cap {cap}: tokens/epoch sequential {sum(per_task):,}  "
              f"full cumulative run {cumulative:,} ({cumulative / sum(per_task):.2f}x)")
    return rows


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def profile_train_config(args, task_data, tokenizer, arm, prompt_cap):
    cfg = get_config([], model=args.model, sparsifier=arm, sparsity=0.1,
                     batch_size=args.batch_size, gradient_accumulation_steps=1,
                     max_prompt_len=prompt_cap, max_ans_len=args.max_ans_len,
                     learning_rate=1e-5, gradient_checkpointing=args.gradient_checkpointing,
                     num_mask_updates=max(1, args.steps // 10), t_end_ratio=1.0,
                     wandb_mode="disabled")
    device = "cuda"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = train.build_model(cfg, device)
    optimizer = train.build_optimizer(cfg, model)
    t0 = time.time()
    sparsifier = sparse_utils.build_sparsifier(cfg, model, optimizer, total_steps=args.steps)
    prepare_s = time.time() - t0
    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    mix = data.cumulative_train(task_data, len(task_data) - 1)
    loader = data.train_loader(mix, tokenizer, cfg.batch_size, prompt_cap, args.max_ans_len,
                               seed=0, num_workers=2, length_grouped=not args.no_length_grouping)
    it = iter(loader)
    step_times, update_times, real, padded = [], [], [], []
    for step in range(args.warmup + args.steps):
        batch = next(it)
        torch.cuda.synchronize()
        start = time.time()
        loss, _, _, n_tok = train.loss_and_task_stats(model, batch, device, ctx, len(task_data))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        mid = time.time()
        updated = bool(sparsifier.step()) if sparsifier is not None else False
        torch.cuda.synchronize()
        end = time.time()
        if updated:
            update_times.append(end - mid)
        if step >= args.warmup:
            step_times.append(end - start)
            real.append(n_tok)
            padded.append(batch["input_ids"].numel())
    total_s = sum(step_times)
    row = {"arm": arm, "prompt_cap": prompt_cap, "batch_size": cfg.batch_size,
           "length_grouped": not args.no_length_grouping,
           "gradient_checkpointing": cfg.gradient_checkpointing,
           "steps": len(step_times), "step_s_mean": statistics.mean(step_times),
           "step_s_p50": pct(step_times, 50), "step_s_max": max(step_times),
           "real_tokens_per_s": sum(real) / total_s, "padded_tokens_per_s": sum(padded) / total_s,
           "padding_fraction": 1 - sum(real) / sum(padded),
           "examples_per_s": len(step_times) * cfg.batch_size / total_s,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
           "prepare_s": prepare_s, "mask_updates": len(update_times),
           "mask_update_s": update_times}
    print(f"{arm:<5} cap {prompt_cap:>4} bs {cfg.batch_size} | step {row['step_s_mean']:.3f}s "
          f"(max {row['step_s_max']:.2f}) | {row['real_tokens_per_s']:,.0f} real tok/s, "
          f"{row['padding_fraction']:.0%} padding | {row['examples_per_s']:.1f} ex/s | "
          f"peak {row['peak_mem_gb']:.1f}GB | prepare {prepare_s:.1f}s | updates "
          f"{[round(u, 2) for u in update_times]}", flush=True)
    del model, optimizer, sparsifier
    return row


def profile_train(args, task_data, tokenizer):
    rows = []
    for arm in args.arms.split(","):
        for cap in [int(c) for c in args.prompt_caps.split(",")]:
            try:
                rows.append(profile_train_config(args, task_data, tokenizer, arm, cap))
            except torch.cuda.OutOfMemoryError:
                print(f"{arm} cap {cap}: OOM at batch size {args.batch_size}", flush=True)
                rows.append({"arm": arm, "prompt_cap": cap, "oom": True})
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

def profile_eval(args, task_data, tokenizer):
    cfg = get_config([], model=args.model)
    model = train.build_model(cfg, "cuda")
    rows = {}
    for task, (_, _, test_set) in task_data.items():
        n = min(args.eval_examples, len(test_set))
        subset = data.TaskDataset(test_set.data, test_set.task_id, test_set.indices[:n])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16), parametrize.cached():
            r = evaluate.evaluate_task(model, tokenizer, subset, task, max_prompt_len=args.eval_prompt_cap,
                                       batch_size=args.eval_batch_size)
        new_tokens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in r["predictions"]]
        rows[task] = {"n": n, "seconds": r["seconds"], "s_per_example": r["seconds"] / n,
                      "cap": evaluate.MAX_NEW_TOKENS[task],
                      "mean_new_tokens": float(np.mean(new_tokens)),
                      "hit_cap": float(np.mean([k >= evaluate.MAX_NEW_TOKENS[task] - 1 for k in new_tokens])),
                      "full_test_minutes": r["seconds"] / n * len(test_set) / 60,
                      "n_test": len(test_set)}
        print(f"{task:<12} {rows[task]['s_per_example']:.3f}s/example ({n} ex, bs {args.eval_batch_size}) "
              f"new tokens {rows[task]['mean_new_tokens']:.0f}/{rows[task]['cap']} "
              f"hit cap {rows[task]['hit_cap']:.0%} -> full test set "
              f"{rows[task]['full_test_minutes']:.1f} min", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=["data", "train", "eval"], required=True)
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--data_dir", default=os.environ.get("TRACE_DATA_DIR", ""))
    p.add_argument("--out", default=os.path.join(os.environ.get("TRACE_OUTPUT_DIR", "out"), "profile"))
    p.add_argument("--max_ans_len", type=int, default=512)
    # train
    p.add_argument("--arms", default="dense,rigl")
    p.add_argument("--prompt_caps", default="1024,2048")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--no_length_grouping", action="store_true", help="plain shuffling instead")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=40)
    # eval
    p.add_argument("--eval_examples", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--eval_prompt_cap", type=int, default=1024)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)
    tokenizer = data.load_tokenizer(args.model)
    task_data = data.load_tasks(args.data_dir, subset=5000, seed=0)
    print(f"== {args.part} | {args.model}", flush=True)
    fn = {"data": profile_data, "train": profile_train, "eval": profile_eval}[args.part]
    result = {"model": args.model, "args": vars(args), "results": fn(args, task_data, tokenizer)}
    if args.part != "data":
        result["gpu"] = torch.cuda.get_device_name(0)
    path = os.path.join(args.out, f"{args.part}_{args.model.split('/')[-1]}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
