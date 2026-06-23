"""Model quantization wrappers (INT8 / FP8 / AWQ / GPTQ).

GPU-only. Implemented in Phase 3; this module currently defines the public
contract and the GPU guard.
"""

from __future__ import annotations

from src.utils.env import is_cuda_available
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_METHODS = ("int8", "fp8", "awq", "gptq")


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
        calibration_dataset: Calibration dataset name.

    Returns:
        The path to the quantized model.

    Raises:
        ValueError: If ``method`` is not supported.
        RuntimeError: If no CUDA GPU is available.
        NotImplementedError: Pending Phase 3 implementation.
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported quantization method {method!r}. "
            f"Expected one of {_SUPPORTED_METHODS}."
        )
    if not is_cuda_available():
        raise RuntimeError(
            "GPU required for this operation. "
            "Run on Google Colab (runtime > change runtime type > GPU) "
            "or a CUDA-enabled machine."
        )
    raise NotImplementedError("quantize_model is implemented in Phase 3.")
