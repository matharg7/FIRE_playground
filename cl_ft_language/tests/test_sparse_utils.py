"""T3.1: src/sparse_utils.py -- which tensors get masks on the HF models."""

import pytest
import torch
import torch.nn as nn
from conftest import MODELS

import sparse_utils

# Linear weights per decoder block: q, k, v, o, gate, up, down.
PER_BLOCK = 7


@pytest.fixture(scope="module", params=MODELS)
def hf_model(request):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(request.param, dtype=torch.float32)


def expected_block_params(config):
    """Weights in one decoder block's Linears, from the model config alone."""
    h, inter = config.hidden_size, config.intermediate_size
    head_dim = getattr(config, "head_dim", None) or h // config.num_attention_heads
    q = config.num_attention_heads * head_dim
    kv = config.num_key_value_heads * head_dim
    return h * q + 2 * h * kv + q * h + 3 * h * inter


def test_targets_are_exactly_the_decoder_block_linears(hf_model):
    targets = sparse_utils.get_sparse_targets(hf_model)
    n_layers = hf_model.config.num_hidden_layers
    assert len(targets) == PER_BLOCK * n_layers
    names = {t["tensor_fqn"] for t in targets}
    assert all(n.startswith("model.layers.") and n.endswith(".weight") for n in names)
    suffixes = {n.split(".")[-2] for n in names}
    assert suffixes == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def test_embeddings_head_and_norms_stay_dense(hf_model):
    names = {t["tensor_fqn"] for t in sparse_utils.get_sparse_targets(hf_model)}
    assert not any("embed" in n or "lm_head" in n or "norm" in n for n in names)
    # both models tie lm_head to the embeddings, so masking it would be wrong anyway
    assert hf_model.config.tie_word_embeddings


def test_sparse_param_count_matches_config(hf_model):
    n = sparse_utils.count_sparse_params(hf_model)
    assert n == expected_block_params(hf_model.config) * hf_model.config.num_hidden_layers
    total = sum(p.numel() for p in hf_model.parameters())
    print(f"{hf_model.config._name_or_path}: {n:,} of {total:,} weights sparsifiable "
          f"({n / total:.1%})")
    assert 0.5 < n / total < 0.9


def test_flatten_folds_masks_and_drops_parametrization_buffers():
    sd = {
        "model.layers.0.mlp.up_proj.parametrizations.weight.original": torch.ones(2, 2),
        "model.layers.0.mlp.up_proj.parametrizations.weight.0.mask": torch.tensor([[1, 0], [0, 1]]).bool(),
        "model.layers.0.mlp.up_proj.parametrizations.weight.0.dense_grad": torch.zeros(2, 2),
        "model.norm.weight": torch.ones(2),
    }
    flat, masks = sparse_utils.flatten_sparse_state_dict(sd)
    assert set(flat) == {"model.layers.0.mlp.up_proj.weight", "model.norm.weight"}
    assert flat["model.layers.0.mlp.up_proj.weight"].tolist() == [[1, 0], [0, 1]]
    assert list(masks) == ["model.layers.0.mlp.up_proj.weight"]


def test_schedule_spans_the_whole_run():
    class C:
        t_end_ratio, num_mask_updates = 0.8, 100
    s = sparse_utils.sparsifier_schedule(C, 10_000)
    assert s == {"t_end": 8000, "delta_t": 80}
    assert sparse_utils.sparsifier_schedule(C, 50)["delta_t"] == 1
