"""Environment detection utilities.

These helpers let every module decide, at runtime, whether a CUDA GPU is
available and where the code is executing (local Mac vs. Google Colab). All
GPU code paths in this project gate themselves on :func:`is_cuda_available`.

The module is CPU-safe: importing it never touches CUDA, and ``torch`` is
imported lazily so the package can be inspected even in minimal environments.
"""

from __future__ import annotations

import importlib.util
import sys


def _torch_available() -> bool:
    """Return True if the ``torch`` package can be imported."""
    return importlib.util.find_spec("torch") is not None


def is_cuda_available() -> bool:
    """Return True if a CUDA GPU is accessible.

    Returns:
        True when ``torch`` is installed and reports at least one usable CUDA
        device, False otherwise (including when ``torch`` is absent).
    """
    if not _torch_available():
        return False
    import torch

    return bool(torch.cuda.is_available())


def is_colab() -> bool:
    """Return True if running inside Google Colab.

    Returns:
        True when the ``google.colab`` module is importable, which is only the
        case inside a Colab runtime.
    """
    if "google.colab" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        # The parent ``google`` namespace package is absent entirely.
        return False


def get_device() -> str:
    """Return the preferred torch device string.

    Returns:
        ``"cuda"`` if a CUDA GPU is available, otherwise ``"cpu"``.
    """
    return "cuda" if is_cuda_available() else "cpu"


def get_gpu_name() -> str:
    """Return the GPU model name.

    Returns:
        The CUDA device name (e.g. ``"NVIDIA A100-SXM4-40GB"``), or
        ``"CPU-only"`` when no GPU is available.
    """
    if not is_cuda_available():
        return "CPU-only"
    import torch

    return str(torch.cuda.get_device_name(0))


def get_env_info() -> dict[str, object]:
    """Return a summary of the current execution environment.

    Returns:
        A dict with keys ``device``, ``gpu_name``, ``cuda_version``,
        ``torch_version`` and ``is_colab``. Version fields are ``None`` when
        ``torch`` is not installed, and ``cuda_version`` is ``None`` on CPU.
    """
    torch_version: str | None = None
    cuda_version: str | None = None

    if _torch_available():
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda if is_cuda_available() else None

    return {
        "device": get_device(),
        "gpu_name": get_gpu_name(),
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "is_colab": is_colab(),
    }
