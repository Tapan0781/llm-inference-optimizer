"""Model quantization wrappers (INT8 / FP8 / AWQ / GPTQ).

Each method is a thin wrapper around its mature ecosystem library, dispatched by
name. The heavy, GPU-only libraries (``awq``, ``auto_gptq``, ``bitsandbytes``,
``transformer_engine``) are imported lazily inside each helper, so this module
imports cleanly on CPU/Mac and method/argument validation stays unit-testable.

GPU-only. ``fp8`` additionally requires an H100-class (Hopper, SM 9.0+) GPU via
Transformer Engine and is rejected elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.env import is_cuda_available
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_METHODS = ("int8", "fp8", "awq", "gptq")

_GPU_REQUIRED_MSG = (
    "GPU required for this operation. "
    "Run on Google Colab (runtime > change runtime type > GPU) "
    "or a CUDA-enabled machine."
)


def quantize_model(
    model_name_or_path: str,
    method: str,
    output_path: str,
    calibration_dataset: str = "pileval",
) -> str:
    """Quantize a model using the specified method.

    Args:
        model_name_or_path: HF model id or local path.
        method: One of ``"int8"``, ``"fp8"``, ``"awq"``, ``"gptq"``.
        output_path: Destination directory for the quantized model.
        calibration_dataset: Calibration dataset name. Used by ``awq``/``gptq``;
            ignored by ``int8`` (bitsandbytes LLM.int8() is calibration-free).

    Returns:
        The path to the quantized model directory.

    Raises:
        ValueError: If ``method`` is not supported.
        RuntimeError: If no CUDA GPU is available (or, for ``fp8``, no Hopper GPU).
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported quantization method {method!r}. Expected one of {_SUPPORTED_METHODS}."
        )
    if not is_cuda_available():
        raise RuntimeError(_GPU_REQUIRED_MSG)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Quantizing %s with method=%s -> %s", model_name_or_path, method, out_dir)
    dispatch = {
        "awq": _quantize_awq,
        "gptq": _quantize_gptq,
        "int8": _quantize_int8,
        "fp8": _quantize_fp8,
    }
    dispatch[method](model_name_or_path, out_dir, calibration_dataset)

    logger.info("Quantization complete: %s", out_dir)
    return str(out_dir)


def _quantize_awq(model_name_or_path: str, out_dir: Path, calibration_dataset: str) -> None:
    """Quantize to 4-bit AWQ via ``autoawq`` and save to ``out_dir``.

    Args:
        model_name_or_path: HF model id or local path.
        out_dir: Destination directory.
        calibration_dataset: Calibration dataset passed to ``model.quantize``.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    model = AutoAWQForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_dataset)
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


def _quantize_gptq(model_name_or_path: str, out_dir: Path, calibration_dataset: str) -> None:
    """Quantize to 4-bit GPTQ via transformers ``GPTQConfig`` (auto-gptq backend).

    Args:
        model_name_or_path: HF model id or local path.
        out_dir: Destination directory.
        calibration_dataset: Calibration dataset name or sample list for GPTQ.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    gptq_config = GPTQConfig(bits=4, dataset=calibration_dataset, tokenizer=tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", quantization_config=gptq_config
    )

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


def _quantize_int8(model_name_or_path: str, out_dir: Path, calibration_dataset: str) -> None:
    """Quantize to INT8 via bitsandbytes LLM.int8() (calibration-free).

    Args:
        model_name_or_path: HF model id or local path.
        out_dir: Destination directory.
        calibration_dataset: Unused — LLM.int8() needs no calibration pass.
    """
    del calibration_dataset  # not used by bitsandbytes LLM.int8()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", quantization_config=bnb_config
    )

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


def _quantize_fp8(model_name_or_path: str, out_dir: Path, calibration_dataset: str) -> None:
    """Reject FP8 unless running on a Hopper (SM 9.0+) GPU.

    FP8 weight quantization is provided by NVIDIA Transformer Engine and is only
    meaningful on H100-class hardware — not the T4/A100 GPUs available on Colab.

    Args:
        model_name_or_path: HF model id or local path.
        out_dir: Destination directory.
        calibration_dataset: Unused.

    Raises:
        RuntimeError: Always on non-Hopper GPUs; FP8 is not implemented for the
            Colab-available T4/A100 tiers.
    """
    del out_dir, calibration_dataset
    import torch

    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(
            f"fp8 quantization requires an H100-class (Hopper, SM 9.0+) GPU, but the "
            f"current device reports SM {major}.x. Use awq/gptq/int8 on T4/A100 instead."
        )
    raise NotImplementedError(
        "fp8 quantization on Hopper via Transformer Engine is not wired up yet "
        f"(model={model_name_or_path!r})."
    )
