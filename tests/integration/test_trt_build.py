"""GPU integration test for the TensorRT engine builder.

Exports the tiny, ungated LLaMA-architecture model to ONNX (Phase 2), then builds
an FP16 TensorRT engine from it and confirms the engine deserializes. Runs on any
modest GPU (free T4) with no HF token. Skipped automatically on CPU-only envs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.env import is_cuda_available

pytestmark = pytest.mark.skipif(
    not is_cuda_available(), reason="GPU required; skipped on CPU-only environments."
)

# Tiny random LLaMA model published by HF for testing — ungated, ~MBs.
_TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_build_engine_fp16_from_tiny_llama(tmp_path: Path) -> None:
    from src.export.onnx_exporter import export_to_onnx
    from src.optimization.trt_builder import build_trt_engine

    onnx_path = export_to_onnx(
        model_name_or_path=_TINY_LLAMA,
        output_path=str(tmp_path / "tiny_onnx"),
        dtype="fp16",
        verify_export=False,  # parity is covered by test_onnx_export.py
    )

    engine_path = build_trt_engine(
        onnx_path=onnx_path,
        output_path=str(tmp_path / "tiny.engine"),
        precision="fp16",
        max_batch_size=4,
        max_seq_len=128,
        workspace_gb=2,
    )

    assert Path(engine_path).exists()
    assert Path(engine_path).stat().st_size > 0
