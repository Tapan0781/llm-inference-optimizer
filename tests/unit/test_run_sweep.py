"""CPU-safe unit tests for src.benchmarking.run_sweep.

The sweep mechanics (grid expansion, prompt synthesis, result schema, CSV/JSON
writing) run on CPU with the eager backend — real output, not mocks. GPU-only
metrics (MFU, power, memory) degrade to sentinels off-GPU. Skips if the tiny model
can't be fetched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.benchmarking.benchmark_runner import (
    BenchmarkResult,
    _max_context,
    _model_metadata,
    run_sweep,
)
from src.serving.inference_engine import InferenceEngine
from src.utils.env import is_cuda_available

_TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def eager_engine() -> InferenceEngine:
    try:
        return InferenceEngine(_TINY, backend="eager", device="cpu")
    except OSError as exc:  # offline — not a code bug
        pytest.skip(f"tiny model unavailable (offline?): {exc}")


def test_model_metadata_resolves_by_model_id() -> None:
    # An HF model id (the engine's model_path) should resolve to its project
    # config via model_id matching — no loaded model needed (works for vLLM).
    stub = SimpleNamespace(model_path="meta-llama/Meta-Llama-3-8B-Instruct", _model=None)
    meta = _model_metadata(stub)  # type: ignore[arg-type]
    assert meta == (8_000_000_000, 32, 4096)  # from configs/model_configs/llama3_8b.yaml


def test_tinyllama_config_resolves_metadata_and_context() -> None:
    # The benchmark model has a project config, so MFU + context resolve for any
    # backend (incl. vLLM, which isn't introspectable).
    stub = SimpleNamespace(model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", _model=None)
    assert _model_metadata(stub) == (1_100_000_000, 22, 2048)  # type: ignore[arg-type]
    assert _max_context(stub) == 2048  # type: ignore[arg-type]


def test_run_sweep_skips_oversized_seq(tmp_path: Path, eager_engine: InferenceEngine) -> None:
    # seq_len + max_new_tokens beyond the model context must be skipped, not crash.
    cfg = tmp_path / "big.yaml"
    cfg.write_text(
        "batch_sizes: [1]\nseq_lens: [8]\nmax_new_tokens: 100000000\nwarmup_iters: 1\n"
        "output_format: [json]\n",
        encoding="utf-8",
    )
    results = run_sweep(str(cfg), eager_engine, str(tmp_path / "r"))
    assert results == []  # the single grid point is skipped (exceeds context)


def _write_cfg(tmp_path: Path) -> Path:
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text(
        "batch_sizes: [1, 2]\n"
        "seq_lens: [4]\n"
        "max_new_tokens: 4\n"
        "warmup_iters: 1\n"
        "output_format: [csv, json]\n",
        encoding="utf-8",
    )
    return cfg


def test_run_sweep_grid_and_schema(tmp_path: Path, eager_engine: InferenceEngine) -> None:
    out = tmp_path / "results"
    results = run_sweep(str(_write_cfg(tmp_path)), eager_engine, str(out))

    # 2 batch sizes × 1 seq len = 2 grid points.
    assert len(results) == 2
    assert all(isinstance(r, BenchmarkResult) for r in results)
    assert {r.batch_size for r in results} == {1, 2}
    assert all(r.backend == "eager" and r.seq_len == 4 for r in results)
    assert all(r.throughput_tps > 0 for r in results)


def test_run_sweep_writes_csv_and_json(tmp_path: Path, eager_engine: InferenceEngine) -> None:
    out = tmp_path / "results"
    run_sweep(str(_write_cfg(tmp_path)), eager_engine, str(out))

    csv_path = out / "benchmark_eager_fp16.csv"
    json_path = out / "benchmark_eager_fp16.json"
    assert csv_path.exists()
    assert json_path.exists()

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    assert {"throughput_tps", "mfu_percent", "power_watts", "ttft_ms"} <= rows[0].keys()
    # CSV has a header + one line per result.
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) == 3


@pytest.mark.skipif(is_cuda_available(), reason="Asserts the CPU sentinel paths.")
def test_run_sweep_cpu_sentinels(tmp_path: Path, eager_engine: InferenceEngine) -> None:
    results = run_sweep(str(_write_cfg(tmp_path)), eager_engine, str(tmp_path / "r"))
    for r in results:
        assert r.mfu_percent == -1.0  # no GPU peak TFLOPs
        assert r.power_watts == -1.0
        assert r.gpu_mem_gb == 0.0
