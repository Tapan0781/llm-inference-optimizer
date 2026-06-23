"""GPU integration smoke tests.

These are skipped automatically when no CUDA device is present, so the suite
stays green on CPU-only machines (Mac, CI ubuntu-latest).
"""

from __future__ import annotations

import pytest

from src.utils.env import is_cuda_available

pytestmark = pytest.mark.skipif(
    not is_cuda_available(), reason="GPU required; skipped on CPU-only environments."
)


def test_cuda_device_reports_name() -> None:
    from src.utils.env import get_gpu_name

    assert get_gpu_name() != "CPU-only"
