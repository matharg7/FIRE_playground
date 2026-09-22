"""Generation-based evaluation on TRACE tasks, one scalar score per task.

Mirrors TRACE's inference/infer_single.py: decode only the new tokens with
special tokens skipped and no stripping, then score with TRACE's
evaluations/eval_*.py. Differences: greedy decoding (TRACE sampled at
temperature 0.1) and a per-task cap on new tokens (TRACE always allowed
max_ans_len; generation still stops at EOS).

The per-task scalar follows the TRACE paper: accuracy for the classification
and math tasks and ScienceQA, ROUGE-L for MeetingBank, edit similarity for
Py150, SARI for 20Minuten. All scalars are in [0, 1].
"""

import time

import torch

import data  # noqa: F401  (puts trace/ on sys.path)
from evaluations import (eval_20Minuten, eval_CStance, eval_FOMC, eval_MeetingBank,  # noqa: E402
                         eval_NumGLUE_cm, eval_NumGLUE_ds, eval_Py150, eval_ScienceQA)

# task -> (metric function, name of the reported scalar, scale to [0, 1])
_SCORERS = {
    "C-STANCE": (lambda s, p, g: eval_CStance.eval(p, g), "accuracy", 1.0),
    "FOMC": (lambda s, p, g: eval_FOMC.eval(p, g), "accuracy", 1.0),
    "MeetingBank": (lambda s, p, g: eval_MeetingBank.eval(p, g), "rouge-L", 1.0),
    "Py150": (lambda s, p, g: eval_Py150.eval(p, g), "similarity", 0.01),
    "ScienceQA": (lambda s, p, g: eval_ScienceQA.eval(p, g), "accuracy", 1.0),
    "NumGLUE-cm": (lambda s, p, g: eval_NumGLUE_cm.eval(p, g), "accuracy", 1.0),
    "NumGLUE-ds": (lambda s, p, g: eval_NumGLUE_ds.eval(p, g), "accuracy", 1.0),
    "20Minuten": (lambda s, p, g: eval_20Minuten.eval(s, p, g), "sari", 0.01),
}

# Cap on generated tokens per task, from the longest training answers
# (C-STANCE/FOMC answer one letter; ScienceQA explanations reach ~3.7k chars).
MAX_NEW_TOKENS = {
    "C-STANCE": 8, "FOMC": 8, "NumGLUE-cm": 16, "NumGLUE-ds": 16,
    "Py150": 128, "MeetingBank": 384, "20Minuten": 384, "ScienceQA": 512,
}


def score_task(task, sources, predictions, ground_truths):
    """(scalar in [0, 1], TRACE's full metric dict) for one task."""
    fn, key, scale = _SCORERS[task]
    metrics = fn(sources, predictions, ground_truths)
    value = metrics[key]
    if isinstance(value, dict):  # SARI reports {"sari": x}
        value = value["sari"]
    return float(value) * scale, metrics


@torch.no_grad()
def generate(model, loader, tokenizer, max_new_tokens, device="cuda"):
    """Greedy continuations for every batch of an eval_loader, in order."""
    was_training = model.training
    model.eval()
    sources, predictions, gts = [], [], []
    for batch in loader:
        sources += batch["sources"]
        gts += batch["gts"]
        input_ids = batch["input_ids"].to(device)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=batch["attention_mask"].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None, top_k=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        predictions += tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
    model.train(was_training)
    return sources, predictions, gts


@torch.no_grad()
def eval_loss(model, tokenizer, dataset, max_prompt_len, max_ans_len, batch_size=16,
              device="cuda"):
    """Teacher-forced token loss on the answer tokens: one forward pass per batch.

    Cheap next to generation (a few seconds per task), and it moves smoothly
    where exact-match accuracy is jumpy, so gradual forgetting stays visible.
    """
    was_training = model.training
    model.eval()
    total, n_tokens = 0.0, 0
    for batch in data.loss_loader(dataset, tokenizer, batch_size, max_prompt_len, max_ans_len):
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=batch["input_ids"].to(device), attention_mask=attention_mask,
                       position_ids=(attention_mask.cumsum(-1) - 1).clamp(min=0)).logits
        targets = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().flatten(0, 1), targets.flatten(),
            ignore_index=-100, reduction="sum")
        total += loss.item()
        n_tokens += int((targets != -100).sum())
    model.train(was_training)
    return total / max(n_tokens, 1)


def evaluate_task(model, tokenizer, dataset, task, max_prompt_len, batch_size=16,
                  max_new_tokens=None, device="cuda", max_ans_len=512, with_loss=True):
    """Generate on one task's (test) split and score it; optionally also its loss."""
    loader = data.eval_loader(dataset, tokenizer, batch_size, max_prompt_len)
    n_new = max_new_tokens or MAX_NEW_TOKENS[task]
    start = time.time()
    sources, preds, gts = generate(model, loader, tokenizer, n_new, device)
    seconds = time.time() - start
    score, metrics = score_task(task, sources, preds, gts)
    result = {"task": task, "score": score, "metrics": metrics, "n": len(preds),
              "seconds": seconds, "predictions": preds, "ground_truths": gts}
    if with_loss:
        start = time.time()
        result["loss"] = eval_loss(model, tokenizer, dataset, max_prompt_len, max_ans_len,
                                   batch_size, device)
        result["loss_seconds"] = time.time() - start
    return result


def evaluate_tasks(model, tokenizer, task_data, upto, max_prompt_len, split="test", **kw):
    """Evaluate tasks 0..upto; returns {task: result}. upto=-1 evaluates none,
    len(task_data)-1 all of them (the zero-shot row uses the latter)."""
    idx = {"train": 0, "eval": 1, "test": 2}[split]
    results = {}
    for i, (task, splits) in enumerate(task_data.items()):
        if i > upto:
            break
        results[task] = evaluate_task(model, tokenizer, splits[idx], task, max_prompt_len, **kw)
    return results
