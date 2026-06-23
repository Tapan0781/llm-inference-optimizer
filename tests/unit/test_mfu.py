"""CPU-safe unit tests for the MFU calculation."""

from __future__ import annotations

import math

from src.benchmarking.benchmark_runner import BenchmarkResult, calculate_mfu


def test_calculate_mfu_basic() -> None:
    # 8B params, 1000 tok/s, A100 peak 312 TFLOPs.
    # actual = 2 * 8e9 * 1000 / 1e12 = 16 TFLOPs -> 16/312 * 100
    mfu = calculate_mfu(
        model_params=8_000_000_000,
        tokens_per_second=1000.0,
        gpu_peak_tflops=312.0,
        num_layers=32,
        hidden_dim=4096,
    )
    assert math.isclose(mfu, 16.0 / 312.0 * 100, rel_tol=1e-9)


def test_calculate_mfu_zero_throughput() -> None:
    assert calculate_mfu(8_000_000_000, 0.0, 312.0, 32, 4096) == 0.0


def test_benchmark_result_fields() -> None:
    result = BenchmarkResult(
        model="llama3_8b",
        backend="eager",
        dtype="fp16",
        batch_size=1,
        seq_len=128,
        ttft_ms=12.3,
        tpot_ms=4.5,
        throughput_tps=900.0,
        mfu_percent=5.1,
        gpu_mem_gb=16.0,
        power_watts=-1.0,
    )
    assert result.model == "llama3_8b"
    assert result.power_watts == -1.0
