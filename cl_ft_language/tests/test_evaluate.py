"""T1.3: src/evaluate.py -- per-task scalar scores, and generation on a GPU."""

import pytest
from conftest import TASKS

import data
import evaluate


def test_every_task_has_a_scorer_and_token_cap():
    assert set(evaluate._SCORERS) == set(TASKS) == set(evaluate.MAX_NEW_TOKENS)


@pytest.mark.parametrize("task", ["C-STANCE", "FOMC", "NumGLUE-cm", "NumGLUE-ds"])
def test_accuracy_tasks(task):
    score, metrics = evaluate.score_task(task, ["p"] * 4, ["A", "B", "C", "A"], ["A", "B", "A", "B"])
    assert score == 0.5 and metrics == {"accuracy": 0.5}


def test_accuracy_is_exact_match_without_stripping():
    # Matches TRACE: a leading space counts as wrong.
    assert evaluate.score_task("FOMC", ["p"], [" A"], ["A"])[0] == 0.0


def test_scienceqa_scores_the_answer_letter():
    preds = ["A\nBecause it is.", "B\nBecause it is."]
    gts = ["A\nBecause it is not.", "C\nBecause it is."]
    assert evaluate.score_task("ScienceQA", ["p"] * 2, preds, gts)[0] == 0.5


def test_meetingbank_uses_rouge_l():
    text = "the council approved the budget for the new park"
    assert evaluate.score_task("MeetingBank", ["p"], [text], [text])[0] == pytest.approx(1.0, abs=1e-6)
    assert evaluate.score_task("MeetingBank", ["p"], ["something else"], [text])[0] < 0.2


def test_py150_similarity_is_scaled_to_unit_interval():
    assert evaluate.score_task("Py150", ["p"], ["return x"], ["return x"])[0] == 1.0
    assert 0 < evaluate.score_task("Py150", ["p"], ["return y"], ["return x"])[0] < 1


def test_20minuten_uses_sari_scaled_to_unit_interval():
    src = ["Provide a simplified version.\n\nDer sehr grosse Hund bellt am Abend laut."]
    good = evaluate.score_task("20Minuten", src, ["Der Hund bellt laut."], ["Der Hund bellt laut."])[0]
    bad = evaluate.score_task("20Minuten", src, ["Katze."], ["Der Hund bellt laut."])[0]
    assert 0 <= bad < good <= 1


def test_empty_predictions_score_zero_and_do_not_crash():
    # Upstream eval_ScienceQA indexed pred[0] and crashed on an empty generation.
    for task in TASKS:
        score = evaluate.score_task(task, ["src"], [""], ["A"])[0]
        if task != "20Minuten":
            assert score == 0.0
    # SARI is the exception: it rewards deleting source words and scores 0/0
    # components as 1, so an empty output on a tiny source scores high.
    assert evaluate.score_task("20Minuten", ["src"], [""], ["A"])[0] > 0.5


# --- generation (GPU) ---------------------------------------------------------------

SMOL = "HuggingFaceTB/SmolLM2-135M"


def _load(dtype_name):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(SMOL, dtype=getattr(torch, dtype_name),
                                                attn_implementation="sdpa").cuda()


@pytest.fixture(scope="module")
def smol():
    return _load("bfloat16"), data.load_tokenizer(SMOL)


@pytest.fixture(scope="module")
def fomc_test(trace_data_dir):
    root = trace_data_dir.split("/TRACE-Benchmark/")[0]
    return data.load_task(root, "FOMC", 1, max_test=12, seed=0)[2]


@pytest.mark.gpu
def test_generation_is_in_order_capped_and_deterministic(smol, fomc_test):
    model, tok = smol
    model.train()
    r1 = evaluate.evaluate_task(model, tok, fomc_test, "FOMC", max_prompt_len=512, batch_size=5)
    r2 = evaluate.evaluate_task(model, tok, fomc_test, "FOMC", max_prompt_len=512, batch_size=5)
    assert r1["n"] == 12 and r1["ground_truths"] == [fomc_test[i]["answer"] for i in range(12)]
    assert 0.0 <= r1["score"] <= 1.0
    assert r1["predictions"] == r2["predictions"]  # greedy is deterministic
    for p in r1["predictions"]:
        assert len(tok(p, add_special_tokens=False)["input_ids"]) <= evaluate.MAX_NEW_TOKENS["FOMC"]
    assert model.training  # train/eval mode restored


@pytest.mark.gpu
def test_left_padding_does_not_change_fp32_generations(fomc_test):
    # In bf16, different padding amounts change rounding and can flip a later
    # greedy token; in fp32, batched and unbatched generation must agree if the
    # left padding and attention mask are handled correctly.
    model, tok = _load("float32"), data.load_tokenizer(SMOL)
    one = evaluate.evaluate_task(model, tok, fomc_test, "FOMC", max_prompt_len=512, batch_size=1)
    many = evaluate.evaluate_task(model, tok, fomc_test, "FOMC", max_prompt_len=512, batch_size=6)
    assert one["predictions"] == many["predictions"]


@pytest.mark.gpu
def test_generation_continues_the_prompt_not_repeats_it(smol):
    import torch
    model, tok = smol
    ds = data.TaskDataset([{"prompt": "The capital of France is", "answer": " Paris"}], 0)
    loader = data.eval_loader(ds, tok, batch_size=1, max_prompt_len=64, num_workers=0)
    _, preds, _ = evaluate.generate(model, loader, tok, max_new_tokens=4)
    assert "capital" not in preds[0] and "paris" in preds[0].lower()


@pytest.mark.gpu
def test_eval_loss_is_finite_and_cheap(smol, fomc_test):
    model, tok = smol
    r = evaluate.evaluate_task(model, tok, fomc_test, "FOMC", max_prompt_len=512, batch_size=8,
                               with_loss=True)
    assert 0 < r["loss"] < 20
    # a forward pass per batch, next to generating up to 8 tokens per example
    assert r["loss_seconds"] < r["seconds"]
    manual = evaluate.eval_loss(model, tok, fomc_test, 512, 512, batch_size=4)
    assert manual == pytest.approx(r["loss"], rel=0.02)  # batch-size independent


@pytest.mark.gpu
def test_evaluate_tasks_covers_tasks_up_to_t(smol, trace_data_dir):
    model, tok = smol
    root = trace_data_dir.split("/TRACE-Benchmark/")[0]
    tasks = data.load_tasks(root, tasks=["C-STANCE", "FOMC", "NumGLUE-ds"], max_test=4, seed=0)
    res = evaluate.evaluate_tasks(model, tok, tasks, upto=1, max_prompt_len=512, batch_size=4)
    assert list(res) == ["C-STANCE", "FOMC"]
    assert all(r["n"] == 4 and r["seconds"] > 0 for r in res.values())
