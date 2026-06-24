"""GPU integration test for the quantization wrappers.

AWQ-quantizes the tiny, ungated LLaMA-architecture model and checks artifacts are
written. AWQ is the most mature, dependency-light path (autoawq is already in
gpu.txt). Skipped automatically on CPU-only environments.
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


def test_awq_quantize_tiny_llama(tmp_path: Path) -> None:
    from src.optimization.quantization import quantize_model

    out_dir = quantize_model(
        model_name_or_path=_TINY_LLAMA,
        method="awq",
        output_path=str(tmp_path / "tiny_awq"),
    )

    # AWQ writes a HF-style directory with a config and at least one weight shard.
    assert Path(out_dir).is_dir()
    assert (Path(out_dir) / "config.json").exists()
    assert any(Path(out_dir).glob("*.safetensors")) or any(Path(out_dir).glob("*.bin"))
