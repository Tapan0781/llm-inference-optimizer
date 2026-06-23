"""Unified inference backend (eager / ONNX / TRT / vLLM).

Implemented in Phase 4; this module currently defines the public contract.
GPU-backed backends gate themselves on :func:`is_cuda_available`.
"""

from __future__ import annotations

from src.utils.env import is_cuda_available
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_BACKENDS = ("eager", "onnx", "trt", "vllm")


class InferenceEngine:
    """Backend-agnostic text generation engine.

    Args:
        model_path: HF model id or local path to a model/engine.
        backend: One of ``"eager"``, ``"onnx"``, ``"trt"``, ``"vllm"``.
        dtype: Compute precision, e.g. ``"fp16"``.
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "eager",
        dtype: str = "fp16",
        device: str = "auto",
    ) -> None:
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend {backend!r}. Expected one of {_SUPPORTED_BACKENDS}."
            )
        self.model_path = model_path
        self.backend = backend
        self.dtype = dtype
        self.device = device

        if backend in ("onnx", "trt", "vllm") and not is_cuda_available():
            raise RuntimeError(
                f"Backend {backend!r} requires a GPU. "
                "Run on Google Colab (runtime > change runtime type > GPU) "
                "or a CUDA-enabled machine."
            )

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> list[str]:
        """Generate completions for a batch of prompts.

        Args:
            prompts: Input prompts.
            max_new_tokens: Maximum number of tokens to generate per prompt.
            temperature: Sampling temperature.

        Returns:
            One generated string per input prompt.

        Raises:
            NotImplementedError: Pending Phase 4 implementation.
        """
        raise NotImplementedError("InferenceEngine.generate is implemented in Phase 4.")

    def warmup(self, n_iters: int = 5) -> None:
        """Run warmup iterations to stabilize timing measurements.

        Args:
            n_iters: Number of warmup iterations.

        Raises:
            NotImplementedError: Pending Phase 4 implementation.
        """
        raise NotImplementedError("InferenceEngine.warmup is implemented in Phase 4.")
