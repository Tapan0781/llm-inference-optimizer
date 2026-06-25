"""CPU-safe unit tests for src.serving.inference_engine.

The eager backend runs on CPU, so unlike the other GPU phases we can exercise
*real* generation here (with the tiny-random model) — not just guard rails. Tests
that need the model are skipped if it can't be fetched (e.g. offline CI).
"""

from __future__ import annotations

import pytest

from src.serving.inference_engine import _SUPPORTED_BACKENDS, InferenceEngine
from src.utils.env import is_cuda_available

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# --------------------------------------------------------------- guard rails


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported backend"):
        InferenceEngine("any/model", backend="tensorflow")


@pytest.mark.skipif(is_cuda_available(), reason="Asserts the CPU-only GPU guard.")
@pytest.mark.parametrize("backend", ["onnx", "trt", "vllm"])
def test_gpu_backend_guard_on_cpu(backend: str) -> None:
    with pytest.raises(RuntimeError, match="requires a GPU"):
        InferenceEngine("any/model", backend=backend)


def test_supported_backends_constant() -> None:
    assert _SUPPORTED_BACKENDS == ("eager", "onnx", "trt", "vllm")


# ----------------------------------------------------- real eager generation


@pytest.fixture(scope="module")
def eager_engine() -> InferenceEngine:
    """Build a CPU eager engine on the tiny model; skip if it can't be fetched."""
    try:
        return InferenceEngine(_TINY, backend="eager", device="cpu")
    except OSError as exc:  # network/download failure (offline) — not a code bug
        pytest.skip(f"tiny model unavailable (offline?): {exc}")


def test_eager_generate_one_completion_per_prompt(eager_engine: InferenceEngine) -> None:
    out = eager_engine.generate(["Hello there", "The sky is"], max_new_tokens=8, temperature=0.0)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(s, str) for s in out)


def test_eager_greedy_is_deterministic(eager_engine: InferenceEngine) -> None:
    kw = {"max_new_tokens": 8, "temperature": 0.0}
    first = eager_engine.generate(["The capital of France is"], **kw)
    second = eager_engine.generate(["The capital of France is"], **kw)
    assert first == second


def test_eager_empty_prompts_returns_empty(eager_engine: InferenceEngine) -> None:
    assert eager_engine.generate([], max_new_tokens=8) == []


def test_eager_warmup_runs(eager_engine: InferenceEngine) -> None:
    # Should complete without raising.
    eager_engine.warmup(n_iters=1)
