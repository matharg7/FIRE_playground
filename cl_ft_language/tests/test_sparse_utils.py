"""T3.1: src/sparse_utils.py -- which tensors get masks on the HF models."""

import pytest
import torch
import torch.nn as nn
from conftest import MODELS

import sparse_utils

# Linear weights per decoder block, by target set.
PER_BLOCK = {"all_linear": 7, "mlp": 3, "up_down": 2, "gate": 1}
SUFFIXES = {
    "all_linear": {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
    "mlp": {"gate_proj", "up_proj", "down_proj"},
    "up_down": {"up_proj", "down_proj"},
    "gate": {"gate_proj"},
}


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


@pytest.mark.parametrize("target_set", ["all_linear", "mlp", "up_down", "gate"])
def test_targets_are_exactly_the_named_block_linears(hf_model, target_set):
    targets = sparse_utils.get_sparse_targets(hf_model, target_set)
    n_layers = hf_model.config.num_hidden_layers
    assert len(targets) == PER_BLOCK[target_set] * n_layers
    names = {t["tensor_fqn"] for t in targets}
    assert all(n.startswith("model.layers.") and n.endswith(".weight") for n in names)
    assert {n.split(".")[-2] for n in names} == SUFFIXES[target_set]


def test_default_target_set_is_the_mlp_only(hf_model):
    """The experiment prunes the MLP only: attention, the embeddings and the
    tied lm_head all stay dense."""
    assert sparse_utils.DEFAULT_TARGETS == "mlp"
    names = {t["tensor_fqn"] for t in sparse_utils.get_sparse_targets(hf_model)}
    assert {n.split(".")[-2] for n in names} == {"gate_proj", "up_proj", "down_proj"}
    assert not any(k in n for n in names
                   for k in ("q_proj", "k_proj", "v_proj", "o_proj"))


@pytest.mark.parametrize("target_set", ["all_linear", "mlp", "up_down", "gate"])
def test_embeddings_head_and_norms_stay_dense(hf_model, target_set):
    names = {t["tensor_fqn"] for t in sparse_utils.get_sparse_targets(hf_model, target_set)}
    assert not any("embed" in n or "lm_head" in n or "norm" in n for n in names)
    # both models tie lm_head to the embeddings, so masking it would be wrong anyway
    assert hf_model.config.tie_word_embeddings


def test_config_and_sparse_utils_agree_on_target_names():
    from config import SPARSE_TARGETS
    assert set(SPARSE_TARGETS) == set(sparse_utils.TARGET_SETS)


def test_sparse_param_count_matches_config(hf_model):
    cfg, n_layers = hf_model.config, hf_model.config.num_hidden_layers
    total = sum(p.numel() for p in hf_model.parameters())

    n_all = sparse_utils.count_sparse_params(hf_model, None, "all_linear")
    assert n_all == expected_block_params(cfg) * n_layers

    # up_down is exactly the two hidden<->intermediate matrices per block.
    n_ud = sparse_utils.count_sparse_params(hf_model, None, "up_down")
    assert n_ud == 2 * cfg.hidden_size * cfg.intermediate_size * n_layers

    n_mlp = sparse_utils.count_sparse_params(hf_model, None, "mlp")
    assert n_mlp == 3 * cfg.hidden_size * cfg.intermediate_size * n_layers
    assert n_ud < n_mlp < n_all

    print(f"{cfg._name_or_path}: mlp {n_mlp:,} of {total:,} ({n_mlp / total:.1%}), "
          f"up_down {n_ud:,} ({n_ud / total:.1%})")
    assert 0.3 < n_ud / total < 0.5
    assert 0.55 < n_mlp / total < 0.7


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
