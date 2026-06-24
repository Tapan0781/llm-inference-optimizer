"""CPU-safe unit tests for src.optimization.trt_builder.

These verify the guard rails (precision validation, not-implemented precisions,
GPU guard) without ever building a real engine — that lives in tests/integration
and needs CUDA + TensorRT.
"""

from __future__ import annotations

import pytest

from src.optimization import trt_builder
from src.utils.env import is_cuda_available


def test_invalid_precision_raises_before_gpu_check() -> None:
    # precision is validated first, so this raises ValueError even on CPU.
    with pytest.raises(ValueError, match="Unsupported precision"):
        trt_builder.build_trt_engine("model.onnx", "model.engine", precision="bf16")


@pytest.mark.parametrize("precision", ["int8", "fp8"])
def test_unimplemented_precision_raises_before_gpu_check(precision: str) -> None:
    # int8/fp8 are valid but not implemented — raised before the GPU guard so
    # this is checkable on CPU.
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        trt_builder.build_trt_engine("model.onnx", "model.engine", precision=precision)


@pytest.mark.skipif(is_cuda_available(), reason="Asserts the CPU-only failure path.")
def test_gpu_guard_raises_on_cpu() -> None:
    with pytest.raises(RuntimeError, match="GPU required"):
        trt_builder.build_trt_engine("model.onnx", "model.engine", precision="fp16")


def test_supported_and_implemented_constants() -> None:
    assert trt_builder._SUPPORTED_PRECISIONS == ("fp32", "fp16", "int8", "fp8")
    assert trt_builder._IMPLEMENTED_PRECISIONS == ("fp32", "fp16")
