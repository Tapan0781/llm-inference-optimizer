"""Profiling wrapper: PyTorch profiler + NVML power tracking.

GPU-only. Implemented in Phase 5; this module currently defines the public
contract and the GPU guard. ``pynvml`` is imported only after the guard passes.
"""

from __future__ import annotations

from src.utils.env import is_cuda_available
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_power_watts() -> float:
    """Return current GPU power draw in watts via NVML.

    Returns:
        Instantaneous board power in watts, or ``-1.0`` if NVML is unavailable.
    """
    if not is_cuda_available():
        return -1.0
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
        pynvml.nvmlShutdown()
        return milliwatts / 1000.0
    except Exception as exc:  # noqa: BLE001 -- NVML/driver errors vary; degrade gracefully
        logger.warning("NVML power read failed: %s", exc)
        return -1.0
