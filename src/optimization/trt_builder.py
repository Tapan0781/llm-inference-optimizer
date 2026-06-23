"""TensorRT engine builder.

GPU-only. Implemented in Phase 3; this module currently defines the public
contract and the GPU guard. Note that ``tensorrt``/``pycuda`` must only be
imported *after* the GPU guard passes (they cannot be installed on Mac).
"""

from __future__ import annotations

from src.utils.env import is_cuda_available
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_trt_engine(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 32,
    max_seq_len: int = 2048,
    workspace_gb: int = 8,
) -> str:
    """Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Source ``.onnx`` path.
        output_path: Destination ``.engine`` path.
        precision: One of ``"fp32"``, ``"fp16"``, ``"int8"``, ``"fp8"``.
        max_batch_size: Maximum batch size for the optimization profile.
        max_seq_len: Maximum sequence length for the optimization profile.
        workspace_gb: Builder workspace size in gigabytes.

    Returns:
        The path to the written ``.engine`` file.

    Raises:
        RuntimeError: If no CUDA GPU is available.
        NotImplementedError: Pending Phase 3 implementation.
    """
    if not is_cuda_available():
        raise RuntimeError(
            "GPU required for this operation. "
            "Run on Google Colab (runtime > change runtime type > GPU) "
            "or a CUDA-enabled machine."
        )
    # import tensorrt, pycuda  # noqa: ERA001 -- only after the guard passes (Phase 3)
    raise NotImplementedError("build_trt_engine is implemented in Phase 3.")
