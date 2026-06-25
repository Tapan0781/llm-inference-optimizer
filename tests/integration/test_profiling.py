"""GPU integration test for the profiler.

Profiles an eager-backend generation on GPU and checks the GPU-only metrics
(peak CUDA memory, NVML power) are populated. Uses the tiny ungated model.
Skipped automatically on CPU-only environments.
"""

from __future__ import annotations

import pytest

from src.utils.env import is_cuda_available

pytestmark = pytest.mark.skipif(
    not is_cuda_available(), reason="GPU required; skipped on CPU-only environments."
)

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_profile_generation_gpu_metrics() -> None:
    from src.profiling.profiler import profile_generation
    from src.serving.inference_engine import InferenceEngine

    engine = InferenceEngine(_TINY, backend="eager", device="cuda")
    result = profile_generation(engine, ["Hello there", "The sky is"], max_new_tokens=8)

    assert result.ttft_ms > 0
    assert result.throughput_tps > 0
    # Model + activations are on the GPU, so peak memory must be non-zero.
    assert result.gpu_mem_gb > 0
    # Power is either a real NVML reading (>0) or the -1 sentinel if NVML is absent.
    assert result.power_watts > 0 or result.power_watts == -1.0
