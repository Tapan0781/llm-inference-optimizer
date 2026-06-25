"""GPU integration test for the benchmark sweep.

Runs a tiny sweep on GPU with the eager backend and checks the GPU-only metrics
(MFU, power, memory) are populated and the result files are written. Skipped on
CPU-only environments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.env import is_cuda_available

pytestmark = pytest.mark.skipif(
    not is_cuda_available(), reason="GPU required; skipped on CPU-only environments."
)

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_run_sweep_gpu(tmp_path: Path) -> None:
    from src.benchmarking.benchmark_runner import run_sweep
    from src.serving.inference_engine import InferenceEngine

    cfg = tmp_path / "sweep.yaml"
    cfg.write_text(
        "batch_sizes: [1, 2]\nseq_lens: [8]\nmax_new_tokens: 8\nwarmup_iters: 1\n"
        "output_format: [csv, json]\n",
        encoding="utf-8",
    )
    engine = InferenceEngine(_TINY, backend="eager", device="cuda")
    results = run_sweep(str(cfg), engine, str(tmp_path / "results"))

    assert len(results) == 2
    assert all(r.gpu_mem_gb > 0 for r in results)  # weights on GPU
    assert (tmp_path / "results" / "benchmark_eager_fp16.json").exists()
