"""Benchmark sweep runner and result schema.

The :class:`BenchmarkResult` schema and :func:`calculate_mfu` are CPU-safe and
fully implemented here. The :func:`run_sweep` driver is implemented in Phase 6.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.profiling.profiler import profile_generation
from src.serving.inference_engine import InferenceEngine
from src.utils.config import load_benchmark_config, load_model_config
from src.utils.env import get_gpu_name
from src.utils.logger import get_logger

logger = get_logger(__name__)

# GPU peak dense TFLOPs by model (for MFU). Matched against get_gpu_name().
_GPU_PEAK_TFLOPS: dict[str, float] = {
    "H100": 989.0,
    "A100": 312.0,
    "L4": 121.0,
    "V100": 125.0,
    "T4": 65.0,
}


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
    """Run a benchmark sweep over the batch_size × seq_len grid for one engine.

    The ``engine`` fixes the ``(backend, dtype, model)``; this sweeps the grid from
    the config and profiles each point. To compare backends, call ``run_sweep`` once
    per engine (the benchmark notebook loops over them). Results are written to
    ``output_dir`` as ``benchmark_<backend>_<dtype>.{csv,json}`` per ``output_format``.

    Args:
        config_path: Benchmark sweep config (name or path).
        engine: The inference engine to benchmark.
        output_dir: Directory where CSV/JSON results are written.

    Returns:
        A list of :class:`BenchmarkResult`, one per grid point.
    """
    cfg = load_benchmark_config(config_path)
    batch_sizes = cfg["batch_sizes"]
    seq_lens = cfg["seq_lens"]
    max_new_tokens = int(cfg.get("max_new_tokens", 128))
    warmup_iters = int(cfg.get("warmup_iters", 3))
    output_formats = cfg.get("output_format", ["csv", "json"])

    # MFU inputs + model context are constant across the grid — resolve once.
    metadata = _model_metadata(engine)
    gpu_tflops = _gpu_peak_tflops()
    max_ctx = _max_context(engine)
    if metadata is None or gpu_tflops <= 0:
        logger.warning(
            "MFU unavailable (model metadata=%s, gpu_peak_tflops=%s); reporting -1.",
            metadata is not None,
            gpu_tflops,
        )

    results: list[BenchmarkResult] = []
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # Input + output must fit the model's context window. vLLM errors hard
            # on overflow (eager only warns), so skip rather than crash the sweep.
            if max_ctx is not None and seq_len + max_new_tokens > max_ctx:
                logger.warning(
                    "Skipping batch=%d seq_len=%d: seq_len + max_new_tokens (%d) "
                    "exceeds model context %d.",
                    batch_size,
                    seq_len,
                    seq_len + max_new_tokens,
                    max_ctx,
                )
                continue
            prompts = _synth_prompts(batch_size, seq_len)
            profile = profile_generation(engine, prompts, max_new_tokens, warmup_iters)
            mfu = _mfu_percent(metadata, profile.throughput_tps, gpu_tflops)
            results.append(
                BenchmarkResult(
                    model=engine.model_path,
                    backend=engine.backend,
                    dtype=engine.dtype,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    ttft_ms=profile.ttft_ms,
                    tpot_ms=profile.tpot_ms,
                    throughput_tps=profile.throughput_tps,
                    mfu_percent=mfu,
                    gpu_mem_gb=profile.gpu_mem_gb,
                    power_watts=profile.power_watts,
                )
            )
            logger.info(
                "Swept batch=%d seq_len=%d: %.1f tok/s, MFU=%.1f%%",
                batch_size,
                seq_len,
                profile.throughput_tps,
                mfu,
            )

    _write_results(results, engine, Path(output_dir), output_formats)
    return results


def _synth_prompts(batch_size: int, seq_len: int) -> list[str]:
    """Build ``batch_size`` prompts of approximately ``seq_len`` tokens.

    Uses a repeated common token; a leading index keeps prompts distinct so caches
    (e.g. vLLM prefix caching) don't skew per-request timing. Length is approximate
    (whitespace-token heuristic), which is fine for relative benchmarking.

    Args:
        batch_size: Number of prompts.
        seq_len: Target token length per prompt.

    Returns:
        A list of ``batch_size`` prompt strings.
    """
    prompts: list[str] = []
    for i in range(batch_size):
        words = ["the"] * max(seq_len, 1)
        words[0] = str(i)  # keep prompts distinct without changing the length
        prompts.append(" ".join(words))
    return prompts


def _metadata_from_cfg(cfg: dict) -> tuple[int, int, int] | None:
    """Extract ``(params, num_layers, hidden_dim)`` from a model config mapping."""
    try:
        return (
            int(cfg["num_parameters_b"] * 1e9),
            int(cfg["num_layers"]),
            int(cfg["hidden_dim"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _config_by_model_id(model_id: str) -> dict | None:
    """Find a model config in ``configs/model_configs/`` whose ``model_id`` matches.

    Lets an HF id (the engine's ``model_path``) resolve to its project config so
    MFU works for any backend (e.g. vLLM, where the model isn't introspectable).

    Args:
        model_id: The HF model id to match against each config's ``model_id``.

    Returns:
        The matching config mapping, or ``None``.
    """
    from src.utils.config import MODEL_CONFIG_DIR

    for path in sorted(MODEL_CONFIG_DIR.glob("*.y*ml")):
        try:
            cfg = load_model_config(path.stem)
        except (FileNotFoundError, ValueError):
            continue
        if cfg.get("model_id") == model_id:
            return cfg
    return None


def _max_context(engine: InferenceEngine) -> int | None:
    """Resolve the model's max context length, or ``None`` if unknown.

    Prefers a project config's ``max_seq_len`` (by name/path or model_id); falls
    back to the loaded model's ``max_position_embeddings``.

    Args:
        engine: The benchmarked engine.

    Returns:
        The max context length in tokens, or ``None``.
    """
    cfg: dict | None = None
    try:
        cfg = load_model_config(engine.model_path)
    except (FileNotFoundError, ValueError):
        cfg = _config_by_model_id(engine.model_path)
    if cfg is not None and cfg.get("max_seq_len") is not None:
        try:
            return int(cfg["max_seq_len"])
        except (TypeError, ValueError):
            pass

    model_cfg = getattr(getattr(engine, "_model", None), "config", None)
    mpe = getattr(model_cfg, "max_position_embeddings", None)
    if mpe is not None:
        try:
            return int(mpe)
        except (TypeError, ValueError):
            pass
    return None


def _model_metadata(engine: InferenceEngine) -> tuple[int, int, int] | None:
    """Resolve ``(params, num_layers, hidden_dim)`` for MFU, or ``None`` if unknown.

    Prefers a project model config (``configs/model_configs/``); falls back to
    introspecting the loaded model's HF config.

    Args:
        engine: The benchmarked engine.

    Returns:
        ``(params, num_layers, hidden_dim)`` or ``None``.
    """
    # 1) A project model config — matched by name/path, or by its model_id (so an
    #    HF id like "meta-llama/Meta-Llama-3-8B-Instruct" resolves too). This path
    #    works for every backend, including vLLM where the model isn't a torch
    #    module we can introspect.
    cfg: dict | None = None
    try:
        cfg = load_model_config(engine.model_path)
    except (FileNotFoundError, ValueError):
        cfg = _config_by_model_id(engine.model_path)
    if cfg is not None:
        meta = _metadata_from_cfg(cfg)
        if meta is not None:
            return meta

    # 2) Fall back to introspecting the loaded model (eager/onnx).
    model = getattr(engine, "_model", None)
    if model is None:
        return None
    model_cfg = getattr(model, "config", None)
    if model_cfg is None:
        return None
    layers = getattr(model_cfg, "num_hidden_layers", None)
    hidden = getattr(model_cfg, "hidden_size", None)
    params: int | None = None
    try:
        if hasattr(model, "num_parameters"):
            params = int(model.num_parameters())
        elif hasattr(model, "parameters"):
            params = int(sum(p.numel() for p in model.parameters()))
    except Exception as exc:  # noqa: BLE001 -- introspection varies by backend
        logger.warning("Could not count model parameters: %s", exc)
    if params is None or layers is None or hidden is None:
        return None
    return params, int(layers), int(hidden)


def _gpu_peak_tflops() -> float:
    """Return the current GPU's peak TFLOPs for MFU, or ``-1.0`` if unknown/CPU."""
    name = get_gpu_name()
    for key, tflops in _GPU_PEAK_TFLOPS.items():
        if key in name:
            return tflops
    return -1.0


def _mfu_percent(
    metadata: tuple[int, int, int] | None, tokens_per_second: float, gpu_tflops: float
) -> float:
    """Compute MFU% from resolved metadata, or ``-1.0`` if inputs are unavailable."""
    if metadata is None or gpu_tflops <= 0 or tokens_per_second <= 0:
        return -1.0
    params, num_layers, hidden_dim = metadata
    return calculate_mfu(params, tokens_per_second, gpu_tflops, num_layers, hidden_dim)


def _write_results(
    results: list[BenchmarkResult],
    engine: InferenceEngine,
    output_dir: Path,
    output_formats: list[str],
) -> None:
    """Write sweep results to ``output_dir`` as CSV and/or JSON.

    Filenames are tagged with backend + dtype so per-engine sweeps don't overwrite
    each other when looping backends.

    Args:
        results: The sweep results.
        engine: The benchmarked engine (provides the filename tag).
        output_dir: Destination directory (created if needed).
        output_formats: Subset of ``{"csv", "json"}``.
    """
    if not results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark_{engine.backend}_{engine.dtype}"
    rows = [asdict(r) for r in results]

    if "json" in output_formats:
        path = output_dir / f"{stem}.json"
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        logger.info("Wrote %s", path)
    if "csv" in output_formats:
        path = output_dir / f"{stem}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Wrote %s", path)
