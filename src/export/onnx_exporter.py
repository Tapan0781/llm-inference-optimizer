"""HuggingFace -> ONNX export via HF Optimum.

Exports a HuggingFace causal-LM (LLaMA 3) to ONNX as a decoder *with* KV-cache
(the ``text-generation-with-past`` task), which is the only form useful for fast
autoregressive inference and for the downstream TensorRT build (Phase 3).

GPU-only. The heavy imports (``optimum``, ``transformers``, ``onnxruntime``) are
deferred until after the GPU guard so this module imports cleanly on CPU/Mac.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.env import is_cuda_available
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_DTYPES = ("fp32", "fp16")

# Optimum task that exports the decoder together with past_key_values.
_EXPORT_TASK = "text-generation-with-past"

# Verification tolerances on last-token logits (PyTorch reference vs ONNXRuntime).
_TOLERANCES: dict[str, dict[str, float]] = {
    "fp32": {"atol": 1e-4, "rtol": 1e-3},
    "fp16": {"atol": 1e-2, "rtol": 1e-2},
}

_GPU_REQUIRED_MSG = (
    "GPU required for this operation. "
    "Run on Google Colab (runtime > change runtime type > GPU) "
    "or a CUDA-enabled machine."
)


def export_to_onnx(
    model_name_or_path: str,
    output_path: str,
    dtype: str = "fp16",
    opset_version: int = 17,
    verify_export: bool = True,
) -> str:
    """Export a HuggingFace causal-LM to ONNX (decoder with KV-cache).

    Args:
        model_name_or_path: HF model id or local path. Gated models (e.g.
            ``meta-llama/Meta-Llama-3-8B-Instruct``) require ``HF_TOKEN``.
        output_path: Destination directory for the ONNX artifacts. The primary
            graph is written as ``model.onnx`` inside this directory.
        dtype: Export precision, ``"fp32"`` or ``"fp16"``.
        opset_version: ONNX opset version.
        verify_export: If True, run a numerical parity check (PyTorch reference
            vs ONNXRuntime) on last-token logits after export.

    Returns:
        The path to the exported ``model.onnx`` file.

    Raises:
        ValueError: If ``dtype`` is unsupported.
        RuntimeError: If no CUDA GPU is available, or verification fails.
    """
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}. Expected one of {_SUPPORTED_DTYPES}.")
    if not is_cuda_available():
        raise RuntimeError(_GPU_REQUIRED_MSG)

    # Heavy, GPU-only imports — only after the guard passes.
    from optimum.exporters.onnx import main_export

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting %s -> ONNX (task=%s, dtype=%s, opset=%d) at %s",
        model_name_or_path,
        _EXPORT_TASK,
        dtype,
        opset_version,
        out_dir,
    )

    main_export(
        model_name_or_path=model_name_or_path,
        output=str(out_dir),
        task=_EXPORT_TASK,
        opset=opset_version,
        dtype=dtype,
        device="cuda",
        no_post_process=False,
    )

    onnx_path = out_dir / "model.onnx"
    if not onnx_path.exists():
        # Optimum may name the merged graph differently across versions; fall back.
        candidates = sorted(out_dir.glob("*.onnx"))
        if not candidates:
            raise RuntimeError(f"Export produced no .onnx file in {out_dir}.")
        onnx_path = candidates[0]

    logger.info("Export complete: %s", onnx_path)

    if verify_export:
        _verify_export(model_name_or_path, out_dir, dtype)

    return str(onnx_path)


def _verify_export(model_name_or_path: str, onnx_dir: Path, dtype: str) -> None:
    """Verify the ONNX export against the PyTorch reference on last-token logits.

    Runs the original HF model and the exported ONNX model on the same prompt and
    asserts their final-position logits agree within the dtype's tolerance.

    Args:
        model_name_or_path: The source HF model id/path (reference).
        onnx_dir: Directory containing the exported ONNX model.
        dtype: Export precision, used to pick comparison tolerances.

    Raises:
        RuntimeError: If logits diverge beyond tolerance.
    """
    import numpy as np
    import torch
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Verifying ONNX export against PyTorch reference (dtype=%s)...", dtype)
    tol = _TOLERANCES[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    prompt = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(prompt, return_tensors="pt")

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype
    ).to("cuda")
    reference.eval()
    with torch.no_grad():
        ref_logits = reference(**{k: v.to("cuda") for k, v in inputs.items()}).logits
    ref_last = ref_logits[:, -1, :].float().cpu().numpy()

    onnx_model = ORTModelForCausalLM.from_pretrained(onnx_dir, provider="CUDAExecutionProvider")
    onnx_out = onnx_model(**inputs)
    onnx_last = onnx_out.logits[:, -1, :].float().cpu().numpy()

    max_abs_diff = float(np.max(np.abs(ref_last - onnx_last)))
    if not np.allclose(ref_last, onnx_last, atol=tol["atol"], rtol=tol["rtol"]):
        raise RuntimeError(
            f"ONNX verification FAILED: max abs diff {max_abs_diff:.4g} exceeds "
            f"atol={tol['atol']}, rtol={tol['rtol']}."
        )

    # Argmax (next-token) agreement is the practically important check.
    ref_argmax = int(ref_last.argmax())
    onnx_argmax = int(onnx_last.argmax())
    logger.info(
        "ONNX verification PASSED: max abs diff %.4g, next-token match=%s",
        max_abs_diff,
        ref_argmax == onnx_argmax,
    )
