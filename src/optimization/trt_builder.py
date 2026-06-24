"""TensorRT engine builder.

Builds a TensorRT engine from the Phase 2 ONNX graph (a decoder *with* KV-cache,
exported via ``text-generation-with-past``). That graph has dynamic axes on batch,
sequence length, and every ``past_key_values.*`` input, so the build constructs a
single optimization profile covering all of them — the static dims (num_heads,
head_dim) are read straight from the parsed ONNX inputs, so no model config is
needed here.

GPU-only. ``tensorrt``/``pycuda`` must only be imported *after* the GPU guard
passes (they cannot be installed on Mac). Precision validation and the
not-yet-implemented int8/fp8 paths are checked *before* the guard so they remain
unit-testable on CPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.env import is_cuda_available
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Every precision the contract accepts.
_SUPPORTED_PRECISIONS = ("fp32", "fp16", "int8", "fp8")

# Precisions actually implemented in this phase. int8 needs an entropy
# calibrator and fp8 needs explicit Q/DQ nodes in the ONNX graph (+H100) — both
# are deferred to a later pass.
_IMPLEMENTED_PRECISIONS = ("fp32", "fp16")

_GPU_REQUIRED_MSG = (
    "GPU required for this operation. "
    "Run on Google Colab (runtime > change runtime type > GPU) "
    "or a CUDA-enabled machine."
)


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
            ``"int8"`` and ``"fp8"`` are not yet implemented.
        max_batch_size: Maximum batch size for the optimization profile.
        max_seq_len: Maximum sequence length for the optimization profile.
        workspace_gb: Builder workspace size in gigabytes.

    Returns:
        The path to the written ``.engine`` file.

    Raises:
        ValueError: If ``precision`` is not a supported value.
        NotImplementedError: If ``precision`` is ``"int8"`` or ``"fp8"``.
        RuntimeError: If no CUDA GPU is available, or the build fails.
    """
    if precision not in _SUPPORTED_PRECISIONS:
        raise ValueError(
            f"Unsupported precision {precision!r}. Expected one of {_SUPPORTED_PRECISIONS}."
        )
    if precision not in _IMPLEMENTED_PRECISIONS:
        raise NotImplementedError(
            f"precision={precision!r} is not implemented yet. int8 requires an entropy "
            "calibrator and fp8 requires explicit Q/DQ nodes in the ONNX graph (and an "
            f"H100). Implemented precisions: {_IMPLEMENTED_PRECISIONS}."
        )
    if not is_cuda_available():
        raise RuntimeError(_GPU_REQUIRED_MSG)

    src = Path(onnx_path)
    if not src.exists():
        raise RuntimeError(f"ONNX model not found: {src}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    serialized = _build_serialized_engine(src, precision, max_batch_size, max_seq_len, workspace_gb)
    out.write_bytes(serialized)
    logger.info("Engine written: %s (%.1f MB)", out, len(serialized) / 1e6)

    _verify_engine(out)
    return str(out)


def _build_serialized_engine(
    onnx_path: Path,
    precision: str,
    max_batch_size: int,
    max_seq_len: int,
    workspace_gb: int,
) -> bytes:
    """Parse the ONNX graph and build a serialized TensorRT engine.

    Args:
        onnx_path: Source ``.onnx`` path.
        precision: ``"fp32"`` or ``"fp16"`` (already validated by the caller).
        max_batch_size: Maximum batch size for the optimization profile.
        max_seq_len: Maximum sequence length for the optimization profile.
        workspace_gb: Builder workspace size in gigabytes.

    Returns:
        The serialized engine bytes.

    Raises:
        RuntimeError: If ONNX parsing or engine building fails.
    """
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    # Explicit-batch network. The EXPLICIT_BATCH flag was required on TRT < 10,
    # deprecated on 10.x, and removed on 11.x (explicit batch is the only mode, so
    # the flag is 0). Guard on the attribute so one call works across versions.
    network_flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing ONNX graph: %s", onnx_path)
    with onnx_path.open("rb") as handle:
        if not parser.parse(handle.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Failed to parse ONNX graph {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if precision == "fp16":
        # NB: builder.platform_has_fast_fp16 was removed in TRT 10/11; modern
        # targets (incl. T4) all have fast FP16, so we just enable the flag.
        config.set_flag(trt.BuilderFlag.FP16)

    profile = _build_optimization_profile(builder, network, max_batch_size, max_seq_len)
    config.add_optimization_profile(profile)

    logger.info(
        "Building TensorRT engine (precision=%s, max_batch=%d, max_seq=%d, workspace=%dGB)...",
        precision,
        max_batch_size,
        max_seq_len,
        workspace_gb,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "TensorRT engine build returned None. Check the builder log above for the "
            "failing layer or an unsatisfiable optimization profile."
        )
    return bytes(serialized)


def _build_optimization_profile(
    builder: Any,  # trt.Builder (typed loosely — no hard import on CPU)
    network: Any,  # trt.INetworkDefinition
    max_batch_size: int,
    max_seq_len: int,
) -> Any:
    """Build one optimization profile covering every dynamic input of the graph.

    Dynamic dims (reported as ``-1``) are assigned ranges by position: dim 0 is the
    batch dimension; any other dynamic dim is a sequence/past-length dimension.
    KV-cache inputs (``past_key_values.*``) get a sequence ``min`` of 0 so the
    prefill step (empty cache) is covered. Static dims (num_heads, head_dim) are
    kept exactly as the ONNX graph declares them.

    Args:
        builder: The TensorRT ``Builder``.
        network: The parsed ``INetworkDefinition``.
        max_batch_size: Upper bound for the batch dimension.
        max_seq_len: Upper bound for any sequence/past-length dimension.

    Returns:
        A populated ``IOptimizationProfile``.
    """
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = list(tensor.shape)
        is_past = "past" in name.lower()

        min_shape, opt_shape, max_shape = [], [], []
        for axis, dim in enumerate(shape):
            if dim != -1:  # static dim (e.g. num_heads, head_dim) — keep as-is
                min_shape.append(dim)
                opt_shape.append(dim)
                max_shape.append(dim)
            elif axis == 0:  # batch
                min_shape.append(1)
                opt_shape.append(max_batch_size)
                max_shape.append(max_batch_size)
            else:  # sequence / past length
                seq_min = 0 if is_past else 1
                min_shape.append(seq_min)
                opt_shape.append(max_seq_len)
                max_shape.append(max_seq_len)

        profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
        logger.info("Profile %-28s min=%s opt=%s max=%s", name, min_shape, opt_shape, max_shape)

    return profile


def _verify_engine(engine_path: Path) -> None:
    """Deserialize the written engine and log its I/O bindings as a sanity check.

    Args:
        engine_path: Path to the serialized ``.engine`` file.

    Raises:
        RuntimeError: If the engine fails to deserialize.
    """
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Engine verification FAILED: could not deserialize {engine_path}.")

    bindings = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    logger.info("Engine verification PASSED: %d I/O tensors %s", len(bindings), bindings)
