"""T2.1: the config module."""
import pytest

from config_sparse import CONFIG, Config, get_config, str2bool


class TestDefaults:
    @pytest.mark.parametrize("key,expected", [
        ("eval_interval", 2000), ("eval_iters", 200), ("seed", 1337),
        ("gradient_accumulation_steps", 8), ("batch_size", 60), ("block_size", 1024),
        ("n_layer", 12), ("n_head", 12), ("n_embd", 768), ("bias", False),
        ("learning_rate", 6e-4), ("weight_decay", 1e-1), ("beta1", 0.9),
        ("beta2", 0.95), ("grad_clip", 1.0), ("min_lr", 6e-5),
        ("warmup_ratio", 0.1), ("max_warmup_iters", 2000), ("dtype", "float16"),
    ])
    def test_matches_train_py(self, key, expected):
        """Defaults are train.py's, so the two scripts describe the same run."""
        assert CONFIG[key] == expected

    def test_arms_available(self):
        cfg = get_config([])
        assert cfg.method == "vanilla"
        assert cfg.sparsifier == "dense"


class TestParsing:
    def test_underscore_and_dash_both_work(self):
        assert get_config(["--c0_dataset=wikitext"]).c0_dataset == "wikitext"
        assert get_config(["--c0-dataset=wikitext"]).c0_dataset == "wikitext"

    def test_space_separated(self):
        assert get_config(["--sparsity", "0.5"]).sparsity == 0.5

    @pytest.mark.parametrize("text,expected", [
        ("False", False), ("false", False), ("0", False), ("no", False),
        ("True", True), ("true", True), ("1", True), ("yes", True),
    ])
    def test_bool_spellings(self, text, expected):
        assert get_config([f"--compile={text}"]).compile is expected

    def test_float_flag_accepts_int_text(self):
        """train.py's configurator rejects this: its type assert requires a float."""
        assert get_config(["--c0_subset_ratio=1"]).c0_subset_ratio == 1.0

    def test_unknown_flag_rejected(self):
        with pytest.raises(SystemExit):
            get_config(["--not_a_real_flag=1"])

    def test_str2bool_rejects_nonsense(self):
        with pytest.raises(Exception):
            str2bool("maybe")


class TestValidation:
    def test_bad_method(self):
        with pytest.raises(ValueError, match="method must be"):
            get_config(["--method=nonsense"])

    def test_bad_sparsifier(self):
        with pytest.raises(ValueError, match="sparsifier must be"):
            get_config(["--sparsifier=nonsense"])

    def test_snp_needs_init_path(self):
        with pytest.raises(ValueError, match="snp_init_load_path"):
            get_config(["--method=snp"])

    def test_checkpoint_mode_needs_warm_start(self):
        with pytest.raises(ValueError, match="warm_start_load_path"):
            get_config(["--method=fire", "--intervene_at_boundary=False"])


class TestConfigObject:
    def test_as_dict_round_trip(self):
        cfg = Config({"a": 1, "b": "x"})
        assert cfg.as_dict() == {"a": 1, "b": "x"}

    def test_get_with_default(self):
        assert Config({"a": 1}).get("missing", "fallback") == "fallback"
