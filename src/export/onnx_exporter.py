"""HuggingFace -> ONNX export.

GPU-only. Implemented in Phase 2; this module currently defines the public
contract and the GPU guard.
"""

from __future__ import annotations

from src.utils.env import is_cuda_available
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    model_name_or_path: str,
    output_path: str,
    dtype: str = "fp16",
    opset_version: int = 17,
    verify_export: bool = True,
) -> str:
    """Export a HuggingFace LLM to ONNX.

    Args:
        model_name_or_path: HF model id or local path.
        output_path: Destination ``.onnx`` path.
        dtype: Export precision, ``"fp32"`` or ``"fp16"``.
        opset_version: ONNX opset version.
        verify_export: Run a numerical parity check after export.

    Returns:
        The path to the written ONNX file.

    Raises:
        RuntimeError: If no CUDA GPU is available.
        NotImplementedError: Pending Phase 2 implementation.
    """
    if not is_cuda_available():
        raise RuntimeError(
            "GPU required for this operation. "
            "Run on Google Colab (runtime > change runtime type > GPU) "
            "or a CUDA-enabled machine."
        )
    raise NotImplementedError("export_to_onnx is implemented in Phase 2.")
