"""CPU-safe unit tests for src.optimization.quantization.

These verify the guard rails (method validation, GPU guard) without ever running
a real quantization — that lives in tests/integration and needs CUDA.
"""

from __future__ import annotations

import pytest

from src.optimization import quantization
from src.utils.env import is_cuda_available


def test_invalid_method_raises_before_gpu_check() -> None:
    # method is validated first, so this raises ValueError even on CPU.
    with pytest.raises(ValueError, match="Unsupported quantization method"):
        quantization.quantize_model("any/model", "int4", "out")


@pytest.mark.skipif(is_cuda_available(), reason="Asserts the CPU-only failure path.")
@pytest.mark.parametrize("method", ["awq", "gptq", "int8", "fp8"])
def test_gpu_guard_raises_on_cpu(method: str) -> None:
    with pytest.raises(RuntimeError, match="GPU required"):
        quantization.quantize_model("any/model", method, "out")


def test_supported_methods_constant() -> None:
    assert quantization._SUPPORTED_METHODS == ("int8", "fp8", "awq", "gptq")
