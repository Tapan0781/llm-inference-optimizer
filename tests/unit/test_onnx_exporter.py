"""CPU-safe unit tests for src.export.onnx_exporter.

These verify the guard rails (dtype validation, GPU guard) without ever running
a real export — that lives in tests/integration and needs CUDA.
"""

from __future__ import annotations

import pytest

from src.export import onnx_exporter
from src.utils.env import is_cuda_available


def test_invalid_dtype_raises_before_gpu_check() -> None:
    # dtype is validated first, so this raises ValueError even on CPU.
    with pytest.raises(ValueError, match="Unsupported dtype"):
        onnx_exporter.export_to_onnx("any/model", "out", dtype="int4")


@pytest.mark.skipif(is_cuda_available(), reason="Asserts the CPU-only failure path.")
def test_gpu_guard_raises_on_cpu() -> None:
    with pytest.raises(RuntimeError, match="GPU required"):
        onnx_exporter.export_to_onnx("any/model", "out", dtype="fp16")


def test_supported_dtypes_constant() -> None:
    assert onnx_exporter._SUPPORTED_DTYPES == ("fp32", "fp16")


def test_export_task_uses_kv_cache() -> None:
    # The export must produce a decoder *with* past_key_values.
    assert onnx_exporter._EXPORT_TASK == "text-generation-with-past"
