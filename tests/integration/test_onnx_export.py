"""GPU integration test for the ONNX export pipeline.

Uses a tiny, ungated LLaMA-architecture model so the full export + numerical
verification runs end-to-end on any modest GPU (free T4) with no HF token.
Skipped automatically on CPU-only environments.
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


def test_export_tiny_llama_fp32(tmp_path: Path) -> None:
    from src.export.onnx_exporter import export_to_onnx

    out_dir = tmp_path / "tiny_onnx"
    onnx_path = export_to_onnx(
        model_name_or_path=_TINY_LLAMA,
        output_path=str(out_dir),
        dtype="fp32",
        verify_export=True,  # runs the PyTorch-vs-ONNX parity check
    )

    assert Path(onnx_path).exists()
    assert Path(onnx_path).suffix == ".onnx"
