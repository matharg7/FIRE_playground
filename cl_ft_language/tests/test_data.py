"""T1.2: src/data.py -- tokenizer, per-task splits, cumulative replay, loaders."""

import collections
import os

import pytest
from conftest import MODELS

import data


@pytest.fixture(scope="module", params=MODELS)
def tokenizer(request):
    return data.load_tokenizer(request.param)


@pytest.fixture(scope="module")
def three_tasks(trace_data_dir):
    root = trace_data_dir.split("/TRACE-Benchmark/")[0]
    return data.load_tasks(root, tasks=data.TASKS[:3], subset=500, seed=0)


def test_tokenizer_is_left_padded_with_a_pad_token(tokenizer):
    assert tokenizer.pad_token_id is not None
    assert tokenizer.padding_side == "left" and tokenizer.truncation_side == "left"


def test_qwen_bos_is_not_invented():
    assert data.load_tokenizer("Qwen/Qwen2.5-0.5B").bos_token_id is None


def test_src_modules_are_not_shadowed_by_trace(tmp_path):
    # trace/train.py must never win over src/train.py, whatever the entry point.
    import subprocess
    import sys
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    code = "import data, train; print(hasattr(train, 'build_model'))"
    out = subprocess.run([sys.executable, "-c", code], cwd=src, capture_output=True, text=True)
    assert out.stdout.strip().splitlines()[-1] == "True", out.stderr[-2000:]  # deepspeed also prints


def test_task_order_is_trace_default():
    assert data.TASKS == ["C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
                          "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]


def test_full_task_sizes(trace_data_dir):
    root = trace_data_dir.split("/TRACE-Benchmark/")[0]
    train, dev, test = data.load_task(root, "FOMC", 1)
    assert (len(train), len(dev), len(test)) == (5000, 496, 496)
    assert train[0]["task_id"] == 1 and set(train[0]) == {"prompt", "answer", "task_id"}


def test_subsampling_is_seeded_and_random(trace_data_dir):
    root = trace_data_dir.split("/TRACE-Benchmark/")[0]
    a = data.load_task(root, "FOMC", 0, max_train=50, max_test=20, seed=1)
    b = data.load_task(root, "FOMC", 0, max_train=50, max_test=20, seed=1)
    c = data.load_task(root, "FOMC", 0, max_train=50, seed=2)
    assert len(a[0]) == 50 and len(a[2]) == 20 and len(a[1]) == 496
    assert a[0].indices == b[0].indices != c[0].indices
    assert a[0].indices != list(range(50))  # not TRACE's first-N


def test_cumulative_sizes_and_task_ids(three_tasks):
    sizes = [len(train) for train, _, _ in three_tasks.values()]
    for t in range(3):
        cum = data.cumulative_train(three_tasks, t)
        assert len(cum) == sum(sizes[: t + 1])
        counts = collections.Counter(cum[i]["task_id"] for i in range(len(cum)))
        assert counts == {k: sizes[k] for k in range(t + 1)}


def test_train_loader_mixes_tasks_and_masks_prompts(three_tasks, tokenizer):
    cum = data.cumulative_train(three_tasks, 2)
    # plain shuffling: every task shows up within the first few batches. With
    # length grouping that holds over an epoch, not batch by batch (see
    # test_length_grouping_cuts_padding_but_keeps_tasks_mixed).
    loader = data.train_loader(cum, tokenizer, batch_size=16, max_prompt_len=256,
                               max_ans_len=64, seed=0, num_workers=0, length_grouped=False)
    seen = set()
    for step, batch in enumerate(loader):
        assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
        assert batch["input_ids"].shape[1] <= 256 + 64
        assert batch["task_ids"].shape == (16,)
        # every example supervises at least EOS, and never a padding position
        assert ((batch["labels"] != -100).sum(1) >= 1).all()
        assert ((batch["labels"] != -100) <= (batch["attention_mask"] == 1)).all()
        seen |= set(batch["task_ids"].tolist())
        if step == 4:
            break
    assert seen == {0, 1, 2}


def test_train_loader_order_is_seeded(three_tasks, tokenizer):
    cum = data.cumulative_train(three_tasks, 1)
    first = lambda seed: next(iter(data.train_loader(cum, tokenizer, 8, 128, 16, seed, 0)))["sources"]
    assert first(3) == first(3) != first(4)


def test_length_grouping_covers_every_example_and_varies_by_epoch(three_tasks):
    cum = data.cumulative_train(three_tasks, 2)
    sampler = data.LengthGroupedSampler(cum, batch_size=8, seed=0)
    first, second = list(sampler), list(sampler)
    assert sorted(first) == sorted(second) == list(range(len(cum)))
    assert first != second  # reshuffled each epoch
    assert list(data.LengthGroupedSampler(cum, 8, seed=0)) == first  # seeded


def test_length_grouping_cuts_padding_but_keeps_tasks_mixed(three_tasks, tokenizer):
    cum = data.cumulative_train(three_tasks, 2)
    waste, tasks_per_batch = {}, {}
    for grouped in (False, True):
        loader = data.train_loader(cum, tokenizer, 8, 1024, 512, seed=0, num_workers=0,
                                   length_grouped=grouped)
        real = padded = 0
        seen = set()
        n_tasks = []
        for batch in loader:
            real += int(batch["attention_mask"].sum())
            padded += batch["input_ids"].numel()
            n_tasks.append(len(set(batch["task_ids"].tolist())))
            seen |= set(batch["task_ids"].tolist())
        waste[grouped] = 1 - real / padded
        tasks_per_batch[grouped] = sum(n_tasks) / len(n_tasks)
        assert seen == {0, 1, 2}
    # measured on 3 tasks x 500 examples: ~0.65 -> ~0.35 padding, i.e. ~1.9x
    # fewer tokens processed. The gain is larger over all 8 cumulative tasks.
    assert waste[True] < 0.6 * waste[False], waste
    assert tasks_per_batch[True] > 1.5, tasks_per_batch  # batches still mix tasks


def test_eval_loader_keeps_order_and_ground_truths(three_tasks, tokenizer):
    _, _, test = three_tasks["FOMC"]
    batch = next(iter(data.eval_loader(test, tokenizer, batch_size=4, max_prompt_len=256,
                                       num_workers=0)))
    assert "labels" not in batch
    assert batch["gts"] == [test[i]["answer"] for i in range(4)]
    assert (batch["task_ids"] == 1).all()
    assert (batch["attention_mask"][:, -1] == 1).all()  # prompts end at the right edge
