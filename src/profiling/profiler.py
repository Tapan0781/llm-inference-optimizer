"""Profiling wrapper: latency metrics + NVML power tracking.

Wraps an :class:`~src.serving.inference_engine.InferenceEngine` as a black box and
measures the latency/power/memory metrics that feed Phase 6's ``BenchmarkResult``:
TTFT, TPOT, throughput, peak GPU memory, and (mean/peak) board power.

Degrade-don't-fail: wall-clock timing always works, so this runs on CPU (with the
eager backend) for testing; the GPU-only parts (CUDA memory, NVML power) guard
themselves and return sentinels (``0.0`` / ``-1.0``) off-GPU.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

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
        return float(milliwatts) / 1000.0
    except Exception as exc:  # noqa: BLE001 -- NVML/driver errors vary; degrade gracefully
        logger.warning("NVML power read failed: %s", exc)
        return -1.0


@dataclass
class ProfileResult:
    """Latency/power/memory metrics for one profiled generation run.

    Attributes:
        ttft_ms: Time to first token, milliseconds (prefill + first token).
        tpot_ms: Time per output token, milliseconds (steady-state decode).
        throughput_tps: Generated tokens per second across the batch.
        gpu_mem_gb: Peak GPU memory used during the run, gigabytes (``0.0`` on CPU).
        power_watts: Mean board power over the run (``-1.0`` if NVML unavailable).
        power_peak_watts: Peak board power over the run (``-1.0`` if unavailable).
    """

    ttft_ms: float
    tpot_ms: float
    throughput_tps: float
    gpu_mem_gb: float
    power_watts: float
    power_peak_watts: float


class _PowerSampler:
    """Background NVML power sampler (mean + peak) over a code block.

    Initializes NVML once, samples board power on a daemon thread at a fixed
    interval, and shuts NVML down on exit. A no-op (sentinels) off-GPU or if NVML
    is unavailable.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: object = None
        self._handle: object = None

    def __enter__(self) -> _PowerSampler:
        if not is_cuda_available():
            return self
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception as exc:  # noqa: BLE001 -- NVML errors vary; degrade gracefully
            logger.warning("NVML power sampling unavailable: %s", exc)
            self._handle = None
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)  # type: ignore[attr-defined]
                self._samples.append(float(mw) / 1000.0)
            except Exception:  # noqa: BLE001 -- skip a bad sample, keep going
                pass

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._handle is not None:
            try:
                self._pynvml.nvmlShutdown()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    @property
    def mean_watts(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else -1.0

    @property
    def peak_watts(self) -> float:
        return max(self._samples) if self._samples else -1.0


def profile_generation(
    engine: object,
    prompts: list[str],
    max_new_tokens: int = 128,
    warmup_iters: int = 3,
) -> ProfileResult:
    """Profile a generation run of an ``InferenceEngine`` (any backend).

    Black-box approach (no change to ``generate``): TTFT is the latency of a
    single-token run; TPOT is the marginal per-token cost from a full
    ``max_new_tokens`` run. Token counts use ``max_new_tokens`` (run greedy so
    output length is the requested length), so results are comparable across
    backends. CUDA work is synchronized around each timed region.

    Args:
        engine: An ``InferenceEngine`` (or anything with ``generate``/``warmup``).
        prompts: Batch of prompts to profile.
        max_new_tokens: Output length for the throughput/TPOT run (``>= 2``).
        warmup_iters: Warmup iterations before timing.

    Returns:
        A :class:`ProfileResult`.

    Raises:
        ValueError: If ``prompts`` is empty or ``max_new_tokens < 2``.
    """
    if not prompts:
        raise ValueError("prompts must be non-empty.")
    if max_new_tokens < 2:
        raise ValueError("max_new_tokens must be >= 2 to measure TPOT.")

    engine.warmup(warmup_iters)  # type: ignore[attr-defined]

    # TTFT: single-token run (prefill + first token).
    t_first = _timed(lambda: engine.generate(prompts, max_new_tokens=1, temperature=0.0))  # type: ignore[attr-defined]

    # Full run, capturing peak memory + power.
    _reset_peak_mem()
    with _PowerSampler() as sampler:
        t_full = _timed(
            lambda: engine.generate(prompts, max_new_tokens=max_new_tokens, temperature=0.0)  # type: ignore[attr-defined]
        )

    batch = len(prompts)
    decode_tokens = max_new_tokens - 1
    ttft_ms = t_first * 1000.0
    tpot_ms = ((t_full - t_first) / decode_tokens) * 1000.0 if decode_tokens > 0 else 0.0
    throughput_tps = (batch * max_new_tokens) / t_full if t_full > 0 else 0.0

    result = ProfileResult(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        throughput_tps=throughput_tps,
        gpu_mem_gb=_peak_mem_gb(),
        power_watts=sampler.mean_watts,
        power_peak_watts=sampler.peak_watts,
    )
    logger.info(
        "Profiled: TTFT=%.2fms TPOT=%.2fms throughput=%.1ftok/s mem=%.2fGB power=%.1fW",
        result.ttft_ms,
        result.tpot_ms,
        result.throughput_tps,
        result.gpu_mem_gb,
        result.power_watts,
    )
    return result


def _timed(fn: Callable[[], object]) -> float:
    """Run ``fn`` and return its wall-clock duration in seconds (CUDA-synced)."""
    _sync()
    start = time.perf_counter()
    fn()
    _sync()
    return time.perf_counter() - start


def _sync() -> None:
    """Synchronize CUDA so async kernels are accounted for in timing (no-op on CPU)."""
    if is_cuda_available():
        import torch

        torch.cuda.synchronize()


def _reset_peak_mem() -> None:
    """Reset the CUDA peak-memory counter (no-op on CPU)."""
    if is_cuda_available():
        import torch

        torch.cuda.reset_peak_memory_stats()


def _peak_mem_gb() -> float:
    """Return peak CUDA memory since the last reset, in GB (``0.0`` on CPU)."""
    if not is_cuda_available():
        return 0.0
    import torch

    return float(torch.cuda.max_memory_allocated()) / 1e9
