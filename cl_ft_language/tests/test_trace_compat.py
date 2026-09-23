"""T1.1: the TRACE modules we reuse import and behave correctly in the new venv."""

import importlib
import json

import pytest
import torch
from conftest import MODELS

REUSED_MODULES = [
    "metrics", "sari", "inference.prompts",
    "utils.data.raw_datasets", "utils.data.data_utils", "utils.data.data_collator",
    "utils.model.model_utils",
    "evaluations.eval_CStance", "evaluations.eval_FOMC", "evaluations.eval_MeetingBank",
    "evaluations.eval_Py150", "evaluations.eval_ScienceQA", "evaluations.eval_NumGLUE_cm",
    "evaluations.eval_NumGLUE_ds", "evaluations.eval_20Minuten",
]


@pytest.mark.parametrize("name", REUSED_MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_collator_does_not_pull_in_training_stack():
    import sys
    importlib.import_module("utils.data.data_collator")
    assert "inference.ICL" not in sys.modules
    assert "training.params" not in sys.modules


# --- SARI (replaces datasets.load_metric) -------------------------------------

def test_sari_matches_hf_reference_example():
    # The example from the HuggingFace `sari` metric card.
    from sari import compute_sari
    out = compute_sari(
        sources=["About 95 species are currently accepted."],
        predictions=["About 95 you now get in."],
        references=[["About 95 species are currently known.",
                     "About 95 species are now accepted.",
                     "95 species are now accepted."]],
    )
    assert out["sari"] == pytest.approx(26.953601953601954, abs=1e-9)


def test_caculate_sari_returns_dict_not_tuple():
    from metrics import caculate_sari
    out = caculate_sari(["Der Hund bellt laut."], ["Der Hund bellt."], ["Der Hund bellt."])
    assert isinstance(out, dict) and 0 <= out["sari"] <= 100


def test_sari_copying_source_scores_below_matching_reference():
    from sari import compute_sari
    src = ["Der sehr grosse Hund bellt am Abend ausgesprochen laut."]
    ref = [["Der grosse Hund bellt laut."]]
    copy = compute_sari(src, src, ref)["sari"]
    match = compute_sari(src, ["Der grosse Hund bellt laut."], ref)["sari"]
    assert match > copy


# --- metric wrappers ------------------------------------------------------------

def test_accuracy_is_exact_match():
    from evaluations import eval_FOMC
    assert eval_FOMC.eval(["A", "B", "C", "A"], ["A", "B", "A", "C"])["accuracy"] == 0.5


def test_scienceqa_splits_answer_letter_and_reasoning():
    from evaluations import eval_ScienceQA
    out = eval_ScienceQA.eval(["A\nBecause x.", "B\nBecause y."], ["A\nBecause x.", "C\nBecause y."])
    assert out["accuracy"] == 0.5
    assert out["rouge-L"] > 0.9


def test_py150_postprocess_and_similarity():
    from evaluations import eval_Py150
    assert eval_Py150.postprocess("x = <NUM_LIT> + <STR_LIT:foo>") == "x = 0 + foo"
    assert eval_Py150.eval(["return x"], ["return x"])["similarity"] == 100


def test_20minuten_reports_scalar_sari():
    from evaluations import eval_20Minuten
    out = eval_20Minuten.eval(["Ein langer Satz."], ["Ein Satz."], ["Ein Satz."])
    assert isinstance(out["sari"], dict) and "sari" in out["sari"]


# --- tokenizer + collator on both models -----------------------------------------

@pytest.fixture(scope="module", params=MODELS)
def tokenizer(request):
    from utils.utils import load_hf_tokenizer
    return load_hf_tokenizer(request.param)


def test_trace_tokenizer_loader_sets_left_padding_and_pad(tokenizer):
    assert tokenizer.pad_token_id is not None
    assert tokenizer.padding_side == "left"
    assert tokenizer.truncation_side == "left"


def _collator(tokenizer, **kw):
    from utils.data.data_collator import DataCollator
    kw.setdefault("max_prompt_len", 64)
    kw.setdefault("max_ans_len", 16)
    return DataCollator(tokenizer, padding="longest", pad_to_multiple_of=8, **kw)


BATCH = [
    {"prompt": "What is the stance? Choose A, B or C.\nText: rates go up.\n", "answer": "B"},
    {"prompt": "Solve.\n2+2=", "answer": "4"},
]


def test_collator_left_pads_and_supervises_only_the_answer(tokenizer):
    out = _collator(tokenizer)(BATCH)
    ids, mask, labels = out["input_ids"], out["attention_mask"], out["labels"]
    assert ids.shape == mask.shape == labels.shape
    assert ids.shape[1] % 8 == 0
    for i, ex in enumerate(BATCH):
        # left padding: real tokens are right-aligned
        assert mask[i, -1] == 1
        first_real = int(mask[i].argmax())
        assert mask[i, first_real:].all() and not mask[i, :first_real].any()
        # labels: only the answer (+ EOS) is supervised, at the right edge
        sup = labels[i][labels[i] != -100]
        assert sup[-1].item() == tokenizer.eos_token_id
        assert tokenizer.decode(sup[:-1]).strip() == ex["answer"]
        assert torch.equal(sup, ids[i, -len(sup):])


def test_collator_truncates_prompt_from_the_left_keeping_answer(tokenizer):
    long = [{"prompt": "word " * 500 + "\nAnswer:", "answer": " yes"}]
    out = _collator(tokenizer, max_prompt_len=32, max_ans_len=8)(long)
    assert out["input_ids"].shape[1] <= 40
    assert out["attention_mask"][0].sum() == 40  # filled to the limit exactly
    sup = out["labels"][0][out["labels"][0] != -100]
    # Upstream bug: a truncated example lost its EOS and supervised ':' instead.
    assert sup[-1].item() == tokenizer.eos_token_id
    assert tokenizer.decode(sup[:-1]).strip() == "yes"
    assert tokenizer.decode(out["input_ids"][0][-len(sup) - 3:-len(sup)]).endswith("Answer:")


def test_collator_caps_long_answers_keeping_their_start(tokenizer):
    ex = [{"prompt": "Explain.\n", "answer": "B\n" + "because " * 100}]
    out = _collator(tokenizer, max_prompt_len=32, max_ans_len=8)(ex)
    sup = out["labels"][0][out["labels"][0] != -100]
    assert len(sup) == 8 and sup[-1].item() == tokenizer.eos_token_id
    assert tokenizer.decode(sup[:-1]).startswith("B")


def test_collator_inference_mode_has_no_labels(tokenizer):
    out = _collator(tokenizer, inference=True)(BATCH)
    assert "labels" not in out
    assert out["gts"] == ["B", "4"]
    assert (out["attention_mask"][:, -1] == 1).all()


def test_collator_skips_missing_bos():
    # Qwen2.5 has no BOS token; upstream prepended None and crashed.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    assert tok.bos_token_id is None
    tok.padding_side = tok.truncation_side = "left"
    out = _collator(tok)(BATCH)
    assert out["input_ids"].dtype == torch.long


# --- dataset loading ----------------------------------------------------------------

def test_create_prompt_dataset_cache_roundtrip(tmp_path):
    from utils.data.data_utils import create_prompt_dataset
    task = tmp_path / "Toy"
    task.mkdir()
    for split, n in (("train", 5), ("eval", 2), ("test", 3)):
        rows = [{"prompt": f"{split} q{i}\n", "answer": str(i)} for i in range(n)]
        (task / f"{split}.json").write_text(json.dumps(rows))
    for _ in range(2):  # the second call reloads the torch.save cache
        train, dev, test = create_prompt_dataset(
            -1, str(task), str(tmp_path / "cache"), seed=0, distributed=False)
    assert (len(train), len(dev), len(test)) == (5, 2, 3)
    assert train[3] == {"prompt": "train q3\n", "answer": "3"}


def test_real_trace_task_loads(trace_data_dir):
    import os
    from utils.data.raw_datasets import LocalJsonFileDataset
    ds = LocalJsonFileDataset(None, 0, -1, os.path.join(trace_data_dir, "FOMC"))
    assert len(ds.get_train_data()) == 5000
    assert len(ds.get_test_data()) == 496
    assert ds.get_answer(ds.get_train_data()[0]) in {"A", "B", "C"}


# --- degenerate generations must score 0, not crash (2026-09-23) ------------

def test_rouge_and_bleu_survive_degenerate_generations():
    """A heavily pruned untrained model emits punctuation-only text. rouge's
    rouge_l_summary_level raises "Collections must contain at least 1 sentence"
    on those, which killed the sparsity-0.5 run during zero-shot eval.
    """
    import metrics
    junk = ["", " ", "\n", ".", "...", "!?"]
    real = ["the cat sat on the mat"] * len(junk)
    # every degenerate prediction, against real targets
    assert metrics.caculate_rouge(junk, real) == 0.0
    assert metrics.caculate_bleu(junk, real, 1) == 0.0
    # degenerate targets too, and both sides at once
    assert metrics.caculate_rouge(real, junk) == 0.0
    assert metrics.caculate_rouge(junk, junk) == 0.0
    # empty input must not divide by zero
    assert metrics.caculate_rouge([], []) == 0.0
    assert metrics.caculate_bleu([], [], 4) == 0.0
    # a real pair still scores
    assert metrics.caculate_rouge(["the cat sat on the mat"], ["the cat sat"]) > 0.0


def test_scienceqa_eval_survives_punctuation_only_predictions():
    from evaluations import eval_ScienceQA
    preds = ["A.", "B", ".", "", "C Because it is warm"]
    gts = ["A Because heat rises"] * len(preds)
    out = eval_ScienceQA.eval(preds, gts)          # must not raise
    assert 0.0 <= out["accuracy"] <= 1.0 and 0.0 <= out["rouge-L"] <= 1.0
