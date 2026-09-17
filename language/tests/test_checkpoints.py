"""T3.3: checkpoints from sparse runs load into a plain GPT."""
import pytest
import torch

from config_sparse import get_config
from helpers import make_tiny_data, run_train, stage_run
from sparse_utils import build_sparsifier, flatten_sparse_state_dict


class TestFlatten:
    def test_dense_state_dict_unchanged(self, tiny_gpt):
        flat, masks = flatten_sparse_state_dict(tiny_gpt.state_dict())
        assert masks == {}
        assert set(flat) == set(tiny_gpt.state_dict())

    def test_sparse_keys_are_renamed(self, tiny_gpt):
        cfg = get_config(["--sparsifier=static", "--sparsity=0.9"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        build_sparsifier(cfg, tiny_gpt, opt, 100)
        flat, masks = flatten_sparse_state_dict(tiny_gpt.state_dict())
        assert not any("parametrizations" in k for k in flat)
        assert "transformer.h.0.attn.c_attn.weight" in flat
        assert "transformer.h.0.attn.c_attn.weight" in masks

    def test_weights_are_actually_sparse(self, tiny_gpt):
        cfg = get_config(["--sparsifier=static", "--sparsity=0.9"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        build_sparsifier(cfg, tiny_gpt, opt, 100)
        flat, _ = flatten_sparse_state_dict(tiny_gpt.state_dict())
        w = flat["transformer.h.0.attn.c_attn.weight"]
        density = torch.count_nonzero(w).item() / w.numel()
        assert density == pytest.approx(0.1, abs=0.02)

    def test_masks_match_the_weights(self, tiny_gpt):
        cfg = get_config(["--sparsifier=static", "--sparsity=0.9"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        build_sparsifier(cfg, tiny_gpt, opt, 100)
        flat, masks = flatten_sparse_state_dict(tiny_gpt.state_dict())
        for key, mask in masks.items():
            assert torch.all(flat[key][~mask] == 0)

    def test_loads_into_a_plain_gpt(self, tiny_gpt, tiny_gpt_config):
        from model import GPT
        cfg = get_config(["--sparsifier=static", "--sparsity=0.9"])
        opt = torch.optim.AdamW(tiny_gpt.parameters(), lr=1e-3)
        build_sparsifier(cfg, tiny_gpt, opt, 100)
        flat, _ = flatten_sparse_state_dict(tiny_gpt.state_dict())
        fresh = GPT(tiny_gpt_config)
        fresh.load_state_dict(flat)  # raises if keys do not line up
        x = torch.randint(0, 128, (2, 32))
        assert torch.allclose(fresh(x)[0], tiny_gpt(x)[0], atol=1e-5)


@pytest.fixture(scope="module")
def sparse_ckpt(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("ckpt")
    data = make_tiny_data(tmp_path)
    run_train(tmp_path, data,
              extra_args=["--sparsifier=static", "--sparsity=0.9",
                          "--save_checkpoint=True"])
    out = stage_run(tmp_path, data) / "output"
    run_dir = next(out.iterdir())
    return torch.load(run_dir / "chunk1_ckpt.pt", map_location="cpu",
                      weights_only=False)


class TestSparseRunCheckpoints:
    def test_checkpoint_has_plain_keys(self, sparse_ckpt):
        assert not any("parametrizations" in k for k in sparse_ckpt["model"])

    def test_checkpoint_stores_masks(self, sparse_ckpt):
        assert sparse_ckpt["masks"]
        assert all(m.dtype == torch.bool for m in sparse_ckpt["masks"].values())

    def test_saved_weights_are_sparse(self, sparse_ckpt):
        w = sparse_ckpt["model"]["transformer.h.0.mlp.c_fc.weight"]
        assert torch.count_nonzero(w).item() / w.numel() == pytest.approx(0.1, abs=0.03)

    def test_reloadable_by_train_sparse(self, sparse_ckpt, tmp_path):
        """A sparse checkpoint can warm-start another run."""
        from model import GPT, GPTConfig
        model = GPT(GPTConfig(**sparse_ckpt["model_args"]))
        model.load_state_dict(sparse_ckpt["model"])


class TestCheckpointDeletion:
    """Checkpoints guard a crash; on success W&B is the record and they go."""

    def test_deleted_after_successful_run(self, tmp_path):
        data = make_tiny_data(tmp_path)
        run_train(tmp_path, data,
                  extra_args=["--save_checkpoint=True",
                              "--delete_checkpoints_on_success=True"])
        run_dir = next((stage_run(tmp_path, data) / "output").iterdir())
        assert list(run_dir.glob("*.pt")) == []

    def test_done_marker_survives(self, tmp_path):
        data = make_tiny_data(tmp_path)
        run_train(tmp_path, data,
                  extra_args=["--save_checkpoint=True",
                              "--delete_checkpoints_on_success=True"])
        run_dir = next((stage_run(tmp_path, data) / "output").iterdir())
        assert (run_dir / "DONE").is_file()

    def test_kept_when_flag_is_off(self, tmp_path):
        data = make_tiny_data(tmp_path)
        run_train(tmp_path, data, extra_args=["--save_checkpoint=True"])
        run_dir = next((stage_run(tmp_path, data) / "output").iterdir())
        assert list(run_dir.glob("*.pt")), "checkpoints should be kept by default"

    def test_failed_run_keeps_checkpoints(self, tmp_path):
        """A crash must leave them behind: DONE is never written, so nothing is deleted."""
        data = make_tiny_data(tmp_path)
        with pytest.raises(AssertionError):
            run_train(tmp_path, data,
                      extra_args=["--save_checkpoint=True",
                                  "--delete_checkpoints_on_success=True",
                                  "--c1_dataset=does_not_exist"])
        run_dir = next((stage_run(tmp_path, data) / "output").iterdir())
        assert list(run_dir.glob("*.pt")), "a failed run must keep its checkpoints"
        assert not (run_dir / "DONE").exists()
