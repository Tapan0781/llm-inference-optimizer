"""CPU-safe unit tests for src.utils.env."""

from __future__ import annotations

from src.utils import env


def test_get_device_returns_valid_string() -> None:
    assert env.get_device() in ("cpu", "cuda")


def test_device_matches_cuda_availability() -> None:
    expected = "cuda" if env.is_cuda_available() else "cpu"
    assert env.get_device() == expected


def test_is_colab_returns_bool() -> None:
    assert isinstance(env.is_colab(), bool)


def test_get_gpu_name_cpu_only_when_no_cuda() -> None:
    if not env.is_cuda_available():
        assert env.get_gpu_name() == "CPU-only"
    else:
        assert isinstance(env.get_gpu_name(), str)


def test_get_env_info_has_expected_keys() -> None:
    info = env.get_env_info()
    assert set(info) == {"device", "gpu_name", "cuda_version", "torch_version", "is_colab"}


def test_get_env_info_cuda_version_none_on_cpu() -> None:
    info = env.get_env_info()
    if info["device"] == "cpu":
        assert info["cuda_version"] is None
