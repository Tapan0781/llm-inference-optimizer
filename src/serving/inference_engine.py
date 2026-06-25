"""Unified inference backend (eager / ONNX / TRT / vLLM).

A single ``generate()`` contract over multiple execution backends:

* ``eager`` — HuggingFace ``AutoModelForCausalLM.generate``. Runs on CPU **or**
  GPU; this is the reference implementation and the only CPU-testable backend.
* ``onnx``  — optimum ``ORTModelForCausalLM.generate`` over a Phase 2 export.
* ``vllm``  — vLLM with continuous batching + chunked prefill (GPU, own env).
* ``trt``   — raw TensorRT engine execution (deferred; hand-rolled decode loop).

Heavy backend libraries are imported lazily inside each ``_init_*`` so this module
imports cleanly on CPU/Mac. GPU-backed backends gate on :func:`is_cuda_available`.
"""

from __future__ import annotations

from typing import Any

from src.utils.env import get_device, is_cuda_available
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_BACKENDS = ("eager", "onnx", "trt", "vllm")
# Backends that cannot run on CPU.
_GPU_BACKENDS = ("onnx", "trt", "vllm")


class InferenceEngine:
    """Backend-agnostic text generation engine.

    Args:
        model_path: HF model id or local path to a model/engine/ONNX dir.
        backend: One of ``"eager"``, ``"onnx"``, ``"trt"``, ``"vllm"``.
        dtype: Compute precision, ``"fp16"`` or ``"fp32"``. ``fp16`` is honoured
            only on CUDA; CPU always runs ``fp32``.
        device: ``"auto"`` (cuda if available else cpu), ``"cuda"`` or ``"cpu"``.
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
        if backend in _GPU_BACKENDS and not is_cuda_available():
            raise RuntimeError(
                f"Backend {backend!r} requires a GPU. "
                "Run on Google Colab (runtime > change runtime type > GPU) "
                "or a CUDA-enabled machine."
            )

        self.model_path = model_path
        self.backend = backend
        self.dtype = dtype
        self.device = self._resolve_device(device)

        # Backend handles, populated by the matching loader.
        self._model: Any = None
        self._tokenizer: Any = None

        loaders = {
            "eager": self._init_eager,
            "onnx": self._init_onnx,
            "trt": self._init_trt,
            "vllm": self._init_vllm,
        }
        logger.info(
            "Initializing InferenceEngine(backend=%s, dtype=%s, device=%s) for %s",
            backend,
            dtype,
            self.device,
            model_path,
        )
        loaders[backend]()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``"auto"`` to a concrete device string.

        Args:
            device: ``"auto"``, ``"cuda"`` or ``"cpu"``.

        Returns:
            ``"cuda"`` or ``"cpu"``.
        """
        return get_device() if device == "auto" else device

    def _torch_dtype(self) -> Any:
        """Map the configured dtype to a torch dtype, honouring CPU limits.

        Returns:
            ``torch.float16`` only on CUDA when ``dtype == "fp16"``; otherwise
            ``torch.float32`` (CPU has no fast/standard half support for this path).
        """
        import torch

        if self.dtype == "fp16" and self.device == "cuda":
            return torch.float16
        return torch.float32

    # ------------------------------------------------------------------ loaders

    def _init_eager(self) -> None:
        """Load the HuggingFace model + tokenizer for the eager backend."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Left-pad so batched decoder-only generation aligns and the prompt prefix
        # can be sliced off uniformly.
        tokenizer.padding_side = "left"
        self._tokenizer = tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=self._torch_dtype()
        )
        model.to(self.device)
        model.eval()
        self._model = model
        logger.info("Eager backend ready on %s.", self.device)

    def _init_onnx(self) -> None:
        """Load the ONNX backend (implemented in a later Phase 4 step)."""
        raise NotImplementedError("The onnx backend lands in a later Phase 4 step.")

    def _init_vllm(self) -> None:
        """Load the vLLM backend (implemented in a later Phase 4 step)."""
        raise NotImplementedError("The vllm backend lands in a later Phase 4 step.")

    def _init_trt(self) -> None:
        """Load the TensorRT backend (deferred)."""
        raise NotImplementedError(
            "The trt serving backend is deferred — it needs a hand-rolled "
            "autoregressive decode loop over the engine's KV-cache bindings."
        )

    # --------------------------------------------------------------- generation

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
            temperature: Sampling temperature; ``0`` (or negative) means greedy.

        Returns:
            One generated completion (new tokens only, prompt stripped) per prompt.
        """
        if not prompts:
            return []
        if self.backend == "eager":
            return self._generate_eager(prompts, max_new_tokens, temperature)
        raise NotImplementedError(f"generate for backend {self.backend!r} is not implemented yet.")

    def _generate_eager(
        self, prompts: list[str], max_new_tokens: int, temperature: float
    ) -> list[str]:
        """Eager-backend generation via ``model.generate``.

        Args:
            prompts: Input prompts.
            max_new_tokens: Max new tokens per prompt.
            temperature: Sampling temperature; ``0`` means greedy.

        Returns:
            The decoded completions (prompt tokens stripped).
        """
        import torch

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        gen_kwargs = self._sampling_kwargs(max_new_tokens, temperature)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Left padding makes the prompt length uniform, so slice it off all rows.
        prompt_len = inputs["input_ids"].shape[1]
        completions = output_ids[:, prompt_len:]
        decoded: list[str] = list(
            self._tokenizer.batch_decode(completions, skip_special_tokens=True)
        )
        return decoded

    def _sampling_kwargs(self, max_new_tokens: int, temperature: float) -> dict[str, Any]:
        """Build ``generate`` sampling kwargs from the temperature policy.

        Args:
            max_new_tokens: Max new tokens per prompt.
            temperature: ``0`` (or negative) → greedy; otherwise sampled.

        Returns:
            Keyword arguments for ``model.generate``.
        """
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature)
        else:
            kwargs.update(do_sample=False)
        return kwargs

    def warmup(self, n_iters: int = 5) -> None:
        """Run warmup iterations to stabilize timing measurements.

        Args:
            n_iters: Number of warmup iterations.
        """
        logger.info("Warming up %s backend for %d iters...", self.backend, n_iters)
        for _ in range(n_iters):
            self.generate(["warmup"], max_new_tokens=8, temperature=0.0)
