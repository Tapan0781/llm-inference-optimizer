"""GPU integration test for the quantization wrappers.

AWQ-quantizes a small *real* LLaMA-architecture model and checks artifacts are
written. A real model is required (not the tiny-random one used elsewhere): AWQ
does group-wise 4-bit quantization with group_size=128, which needs weight dims
divisible by 128 — the tiny-random model (hidden=16) cannot satisfy that.

Requires the quantization environment (requirements/gpu-quant.txt, transformers
5.x + llmcompressor); skipped automatically on CPU-only environments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.env import is_cuda_available

pytestmark = pytest.mark.skipif(
    not is_cuda_available(), reason="GPU required; skipped on CPU-only environments."
)

# Small, ungated, *real* LLaMA-architecture model (hidden=2048=16*128, no HF token).
# ~2.2GB download — heavier than the export/TRT tests, but quantization needs real
# weight statistics and 128-divisible dims.
_TINY_LLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_awq_quantize_small_real_llama(tmp_path: Path) -> None:
    from src.optimization.quantization import quantize_model

    out_dir = quantize_model(
        model_name_or_path=_TINY_LLAMA,
        method="awq",
        output_path=str(tmp_path / "awq"),
    )

    # AWQ writes a HF-style directory with a config and at least one weight shard.
    assert Path(out_dir).is_dir()
    assert (Path(out_dir) / "config.json").exists()
    assert any(Path(out_dir).glob("*.safetensors")) or any(Path(out_dir).glob("*.bin"))
