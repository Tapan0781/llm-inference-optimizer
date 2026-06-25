"""CPU-safe unit tests for src.profiling.profiler.

Latency profiling is wall-clock timing, so it runs on CPU with the eager backend
— real profiling, not just guard rails. GPU-only metrics (power, CUDA memory)
degrade to sentinels off-GPU, which we assert. The tiny-model engine is skipped
if it can't be fetched (offline).
"""

from __future__ import annotations

import pytest

from src.profiling.profiler import ProfileResult, get_power_watts, profile_generation
from src.serving.inference_engine import InferenceEngine
from src.utils.env import is_cuda_available

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_get_power_watts_sentinel_on_cpu() -> None:
    if is_cuda_available():
        pytest.skip("Asserts the CPU sentinel path.")
    assert get_power_watts() == -1.0


@pytest.fixture(scope="module")
def eager_engine() -> InferenceEngine:
    try:
        return InferenceEngine(_TINY, backend="eager", device="cpu")
    except OSError as exc:  # offline — not a code bug
        pytest.skip(f"tiny model unavailable (offline?): {exc}")


def test_profile_generation_returns_metrics(eager_engine: InferenceEngine) -> None:
    result = profile_generation(eager_engine, ["Hello there"], max_new_tokens=4, warmup_iters=1)
    assert isinstance(result, ProfileResult)
    assert result.ttft_ms > 0
    assert result.tpot_ms >= 0
    assert result.throughput_tps > 0


@pytest.mark.skipif(is_cuda_available(), reason="Asserts the CPU sentinel paths.")
def test_profile_generation_cpu_sentinels(eager_engine: InferenceEngine) -> None:
    result = profile_generation(eager_engine, ["Hello there"], max_new_tokens=4, warmup_iters=1)
    assert result.gpu_mem_gb == 0.0
    assert result.power_watts == -1.0
    assert result.power_peak_watts == -1.0


def test_profile_generation_validates_args(eager_engine: InferenceEngine) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        profile_generation(eager_engine, [], max_new_tokens=4)
    with pytest.raises(ValueError, match="max_new_tokens"):
        profile_generation(eager_engine, ["hi"], max_new_tokens=1)
