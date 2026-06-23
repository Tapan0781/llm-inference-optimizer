"""CPU-safe unit tests for src.utils.config."""

from __future__ import annotations

import pytest

from src.utils import config


def test_load_llama3_8b_by_name() -> None:
    cfg = config.load_model_config("llama3_8b")
    assert cfg["model_id"] == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert cfg["architecture"] == "llama"
    assert cfg["num_layers"] == 32


def test_load_model_config_accepts_filename() -> None:
    cfg = config.load_model_config("llama3_8b.yaml")
    assert cfg["hidden_dim"] == 4096


def test_load_llama3_70b_by_name() -> None:
    cfg = config.load_model_config("llama3_70b")
    assert cfg["num_parameters_b"] == 70
    assert cfg["tp_degree"] == 4


def test_all_required_keys_present_in_shipped_configs() -> None:
    for name in ("llama3_8b", "llama3_70b"):
        cfg = config.load_model_config(name)
        for key in config._REQUIRED_MODEL_KEYS:
            assert key in cfg, f"{name} missing {key}"


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        config.load_model_config("does_not_exist")


def test_load_default_benchmark_config() -> None:
    cfg = config.load_benchmark_config()
    assert cfg["batch_sizes"] == [1, 4, 8, 16, 32]
    assert "eager" in cfg["backends"]
