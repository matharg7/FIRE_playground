"""TRACE data for continual fine-tuning: per-task splits, cumulative replay
training sets, the tokenizer, and data loaders.

Reuses TRACE's json loader (utils/data/data_utils.create_dataset) and its
left-padding collator (utils/data/data_collator.DataCollator). Replay is full
cumulative: task t trains on the union of the train splits of tasks 0..t.
"""

import os
import random
import sys

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_TRACE_DIR = os.path.join(os.path.dirname(_SRC_DIR), "trace")
# trace/ must be importable as top-level packages (utils, metrics, evaluations),
# but src/ has to stay ahead of it: trace/train.py (TRACE's RLHF launcher)
# would otherwise shadow our train.py.
for _path in (_TRACE_DIR, _SRC_DIR):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

from utils.data.data_collator import DataCollator  # noqa: E402
from utils.data.data_utils import create_dataset  # noqa: E402

# TRACE's default task order.
TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
         "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]
# Benchmark sizes shipped in the archive (train examples per task).
SUBSETS = {5000: "LLM-CL-Benchmark_5000", 1000: "LLM-CL-Benchmark_1000",
           500: "LLM-CL-Benchmark_500"}


def load_tokenizer(model_name_or_path):
    """Tokenizer set up for TRACE's collator: left padding, left truncation.

    SmolLM2 ships without a pad token, so pad = eos (padding is masked out by
    attention_mask and never supervised). A missing BOS (Qwen2.5) is left
    missing rather than invented; the collator skips it.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    return tok


def task_dir(data_root, task, subset=5000):
    return os.path.join(data_root, "TRACE-Benchmark", SUBSETS[subset], task)


class TaskDataset(Dataset):
    """A split of one task; items are {prompt, answer, task_id}."""

    def __init__(self, prompt_dataset, task_id, indices=None):
        self.data = prompt_dataset
        self.task_id = task_id
        self.indices = list(range(len(prompt_dataset))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.data[self.indices[i]]
        return {"prompt": item["prompt"], "answer": item["answer"], "task_id": self.task_id}


def _subsample(n, k, seed):
    if k is None or k >= n:
        return None
    return sorted(random.Random(seed).sample(range(n), k))


def load_task(data_root, task, task_id, subset=5000, max_train=None, max_eval=None,
              max_test=None, seed=0):
    """(train, eval, test) TaskDatasets for one task.

    max_* draw a seeded random subset (TRACE's own sample_ratio took the first
    N examples in file order instead).
    """
    train, dev, test = create_dataset(-1, task_dir(data_root, task, subset), None, seed)
    return tuple(
        TaskDataset(split, task_id, _subsample(len(split), k, seed + j))
        for j, (split, k) in enumerate(((train, max_train), (dev, max_eval), (test, max_test))))


def load_tasks(data_root, tasks=TASKS, **kw):
    """{task: (train, eval, test)} in task order; task_id is the position."""
    return {task: load_task(data_root, task, i, **kw) for i, task in enumerate(tasks)}


def cumulative_train(task_data, t):
    """Full cumulative replay: the train splits of tasks 0..t, concatenated."""
    splits = list(task_data.values())[: t + 1]
    return ConcatDataset([train for train, _, _ in splits])


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Order examples so each batch holds similar lengths, without losing shuffling.

    68% of every batch was padding when short tasks (FOMC, NumGLUE) shared
    batches with long ones (MeetingBank). Per epoch: shuffle, cut into
    megabatches of `group_factor` batches, sort each megabatch by length, then
    shuffle the resulting batch order. Every example still appears exactly once
    per epoch, and batches still mix tasks -- they just mix tasks of similar
    length.

    Lengths are estimated from character counts (cheap, and monotone in tokens).
    """

    def __init__(self, dataset, batch_size, seed=0, group_factor=50):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.group_factor = group_factor
        self.epoch = 0
        self.lengths = [len(ex["prompt"]) + len(ex["answer"]) for ex in _iter_examples(dataset)]

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        megabatch = self.batch_size * self.group_factor
        batches = []
        for start in range(0, len(order), megabatch):
            chunk = sorted(order[start:start + megabatch], key=lambda i: self.lengths[i])
            batches += [chunk[i:i + self.batch_size] for i in range(0, len(chunk), self.batch_size)]
        rng.shuffle(batches)
        return iter([i for batch in batches for i in batch])


def _iter_examples(dataset):
    for i in range(len(dataset)):
        yield dataset[i]


class TaskCollator:
    """TRACE's DataCollator plus a task_ids tensor (for per-task loss logging)."""

    def __init__(self, tokenizer, max_prompt_len, max_ans_len, inference=False):
        self.collate = DataCollator(tokenizer, padding="longest", max_prompt_len=max_prompt_len,
                                    max_ans_len=max_ans_len, pad_to_multiple_of=8,
                                    inference=inference)

    def __call__(self, batch):
        out = self.collate(batch)
        out["task_ids"] = torch.tensor([ex["task_id"] for ex in batch])
        return out


def train_loader(dataset, tokenizer, batch_size, max_prompt_len, max_ans_len, seed,
                 num_workers=2, length_grouped=True):
    """Shuffled loader; with a cumulative dataset, batches mix all seen tasks.

    length_grouped batches examples of similar length together (far less
    padding); set it False for plain uniform shuffling.
    """
    collate = TaskCollator(tokenizer, max_prompt_len, max_ans_len)
    if length_grouped:
        sampler = LengthGroupedSampler(dataset, batch_size, seed=seed)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate,
                          num_workers=num_workers, drop_last=False)
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen,
                      collate_fn=collate, num_workers=num_workers, drop_last=False)


def loss_loader(dataset, tokenizer, batch_size, max_prompt_len, max_ans_len, num_workers=2):
    """In-order loader with labels, for teacher-forced eval loss (no generation)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=TaskCollator(tokenizer, max_prompt_len, max_ans_len),
                      num_workers=num_workers)


def eval_loader(dataset, tokenizer, batch_size, max_prompt_len, num_workers=2):
    """In-order loader for generation (TRACE's metrics assume dataset order)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      collate_fn=TaskCollator(tokenizer, max_prompt_len, 0, inference=True),
                      num_workers=num_workers)
