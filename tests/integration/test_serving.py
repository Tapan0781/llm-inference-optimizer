"""GPU integration tests for the serving InferenceEngine.

Exercises the GPU-gated backends end-to-end on the tiny ungated model. The eager
backend is covered by CPU unit tests; this file covers onnx (and, where the env
provides vLLM, the vllm backend). Skipped automatically on CPU-only environments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.env import is_cuda_available

pytestmark = pytest.mark.skipif(
    not is_cuda_available(), reason="GPU required; skipped on CPU-only environments."
)

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_onnx_backend_generate(tmp_path: Path) -> None:
    from src.export.onnx_exporter import export_to_onnx
    from src.serving.inference_engine import InferenceEngine

    onnx_dir = tmp_path / "onnx"
    export_to_onnx(_TINY, str(onnx_dir), dtype="fp32", verify_export=False)

    engine = InferenceEngine(str(onnx_dir), backend="onnx", device="cuda")
    out = engine.generate(["Hello there", "The sky is"], max_new_tokens=8, temperature=0.0)

    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(s, str) for s in out)


# Small ungated real model — vLLM needs a real model (not the tiny-random toy).
# Requires the serving env (requirements/gpu-serve.txt); skipped if vLLM is absent.
_SMALL_REAL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def test_vllm_backend_generate() -> None:
    pytest.importorskip("vllm", reason="vLLM not installed (requirements/gpu-serve.txt).")
    from src.serving.inference_engine import InferenceEngine

    engine = InferenceEngine(_SMALL_REAL, backend="vllm", device="cuda")
    out = engine.generate(["Hello there", "The sky is"], max_new_tokens=8, temperature=0.0)

    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(s, str) for s in out)
