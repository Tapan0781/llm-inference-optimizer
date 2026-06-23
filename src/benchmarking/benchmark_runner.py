"""Benchmark sweep runner and result schema.

The :class:`BenchmarkResult` schema and :func:`calculate_mfu` are CPU-safe and
fully implemented here. The :func:`run_sweep` driver is implemented in Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.serving.inference_engine import InferenceEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """A single benchmark measurement.

    Attributes:
        model: Model identifier.
        backend: Inference backend used.
        dtype: Compute precision.
        batch_size: Batch size for this run.
        seq_len: Sequence length for this run.
        ttft_ms: Time to first token, milliseconds.
        tpot_ms: Time per output token, milliseconds.
        throughput_tps: Throughput in tokens per second.
        mfu_percent: Model FLOP utilization, as a percentage.
        gpu_mem_gb: Peak GPU memory used, gigabytes.
        power_watts: Mean board power from NVML, or -1 if unavailable.
    """

    model: str
    backend: str
    dtype: str
    batch_size: int
    seq_len: int
    ttft_ms: float
    tpot_ms: float
    throughput_tps: float
    mfu_percent: float
    gpu_mem_gb: float
    power_watts: float


def calculate_mfu(
    model_params: int,
    tokens_per_second: float,
    gpu_peak_tflops: float,
    num_layers: int,
    hidden_dim: int,
) -> float:
    """Compute Model FLOP Utilization (MFU) as a percentage.

    MFU = actual_tflops / theoretical_peak_tflops. For inference (forward pass
    only) the per-token cost is approximated as ``2 * model_params`` FLOPs.

    Args:
        model_params: Total number of model parameters.
        tokens_per_second: Measured generation throughput.
        gpu_peak_tflops: GPU peak TFLOPs (A100=312, H100=989, T4=65).
        num_layers: Number of transformer layers (reserved for refinement).
        hidden_dim: Hidden dimension (reserved for refinement).

    Returns:
        MFU as a percentage of theoretical peak.
    """
    actual_tflops = 2 * model_params * tokens_per_second / 1e12
    return (actual_tflops / gpu_peak_tflops) * 100


def run_sweep(
    config_path: str,
    engine: InferenceEngine,
    output_dir: str = "results/",
) -> list[BenchmarkResult]:
    """Run a full benchmark sweep defined by a config file.

    Args:
        config_path: Path to a benchmark sweep YAML config.
        engine: The inference engine to benchmark.
        output_dir: Directory where CSV and JSON results are written.

    Returns:
        A list of :class:`BenchmarkResult` for every sweep point.

    Raises:
        NotImplementedError: Pending Phase 6 implementation.
    """
    raise NotImplementedError("run_sweep is implemented in Phase 6.")
