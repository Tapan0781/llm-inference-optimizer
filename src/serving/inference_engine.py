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
        """Load the optimum ORT model + tokenizer for the ONNX backend.

        ``model_path`` is the directory produced by Phase 2's ``export_to_onnx``
        (containing ``model.onnx``). Picks the CUDA ORT provider when available,
        else CPU — mirroring the Phase 2 verification path.
        """
        import onnxruntime as ort
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self._tokenizer = tokenizer

        available = ort.get_available_providers()
        provider = (
            "CUDAExecutionProvider"
            if "CUDAExecutionProvider" in available
            else "CPUExecutionProvider"
        )
        self._model = ORTModelForCausalLM.from_pretrained(self.model_path, provider=provider)
        logger.info("ONNX backend ready (provider=%s).", provider)

    def _init_vllm(self) -> None:
        """Load the vLLM engine.

        Continuous batching is inherent to vLLM's scheduler; chunked prefill is
        enabled explicitly. vLLM tokenizes internally, so no separate tokenizer is
        kept. Runs in its own environment (``requirements/gpu-serve.txt``).
        """
        self._prepare_vllm_runtime()
        from vllm import LLM

        vllm_dtype = "float16" if self.dtype == "fp16" else "float32"
        self._model = LLM(model=self.model_path, dtype=vllm_dtype, enable_chunked_prefill=True)
        logger.info(
            "vLLM backend ready (continuous batching + chunked prefill, dtype=%s).", vllm_dtype
        )

    @staticmethod
    def _prepare_vllm_runtime() -> None:
        """Make vLLM importable + runnable on Colab-style CUDA-mismatched runtimes.

        Three workarounds, all harmless on a correctly-configured box:

        1. Force ``spawn`` for vLLM workers — CUDA cannot be re-initialized in a
           forked child, and notebooks default to ``fork``.
        2. Append the bundled NVIDIA runtime lib dirs (e.g. the cu13 wheels vLLM
           pulls) to ``LD_LIBRARY_PATH`` so *spawned* child processes can load
           ``libcudart`` etc.
        3. Preload those libs into *this* process (``LD_LIBRARY_PATH`` can't help an
           already-started process, so the parent's ``import vllm._C`` needs them
           resident).
        """
        import ctypes
        import glob
        import os
        import sysconfig

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        site = sysconfig.get_paths()["purelib"]
        nvidia_libs = glob.glob(os.path.join(site, "nvidia", "**", "*.so*"), recursive=True)
        if not nvidia_libs:
            return

        lib_dirs = sorted({os.path.dirname(p) for p in nvidia_libs})
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (f":{existing}" if existing else "")

        # A few passes so inter-dependent libs resolve regardless of load order.
        for _ in range(3):
            for so in nvidia_libs:
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

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
        if self.backend in ("eager", "onnx"):
            # Both wrap a HF GenerationMixin (.generate). Eager tensors live on
            # self.device; the ORT model manages its own device, so leave inputs on CPU.
            return self._generate_hf(prompts, max_new_tokens, temperature, self.backend == "eager")
        if self.backend == "vllm":
            return self._generate_vllm(prompts, max_new_tokens, temperature)
        raise NotImplementedError(f"generate for backend {self.backend!r} is not implemented yet.")

    def _generate_vllm(
        self, prompts: list[str], max_new_tokens: int, temperature: float
    ) -> list[str]:
        """vLLM-backend generation. Returns completions in input order.

        Args:
            prompts: Input prompts.
            max_new_tokens: Max new tokens per prompt.
            temperature: Sampling temperature; ``0`` means greedy in vLLM.

        Returns:
            The generated completion text per prompt.
        """
        from vllm import SamplingParams

        params = SamplingParams(temperature=max(temperature, 0.0), max_tokens=max_new_tokens)
        request_outputs = self._model.generate(prompts, params)
        return [ro.outputs[0].text for ro in request_outputs]

    def _generate_hf(
        self, prompts: list[str], max_new_tokens: int, temperature: float, to_device: bool
    ) -> list[str]:
        """Generation via a HF ``.generate`` (shared by eager and onnx backends).

        Args:
            prompts: Input prompts.
            max_new_tokens: Max new tokens per prompt.
            temperature: Sampling temperature; ``0`` means greedy.
            to_device: Move tokenized inputs to ``self.device`` (eager); the ORT
                model handles its own placement, so onnx passes ``False``.

        Returns:
            The decoded completions (prompt tokens stripped).
        """
        import torch

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True)
        if to_device:
            inputs = inputs.to(self.device)
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
