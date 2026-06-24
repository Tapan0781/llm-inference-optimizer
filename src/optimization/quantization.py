"""Model quantization wrappers (INT8 / FP8 / AWQ / GPTQ).

Each method is a thin wrapper around its *maintained* ecosystem library,
dispatched by name. The heavy, GPU-only libraries are imported lazily inside each
helper so this module imports cleanly on CPU/Mac and method/argument validation
stays unit-testable.

Backend choices (the originally-planned autoawq / auto-gptq are both deprecated
and broken against current transformers, so they were swapped out):

* ``awq``  -> ``llmcompressor`` (the vLLM project's successor to AutoAWQ)
* ``gptq`` -> ``gptqmodel`` (maintained successor to auto-gptq)
* ``int8`` -> ``bitsandbytes`` LLM.int8() (calibration-free)
* ``fp8``  -> Transformer Engine, gated to Hopper (SM 9.0+) and otherwise rejected
"""

from __future__ import annotations

from pathlib import Path

from src.utils.env import is_cuda_available
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SUPPORTED_METHODS = ("int8", "fp8", "awq", "gptq")

_GPU_REQUIRED_MSG = (
    "GPU required for this operation. "
    "Run on Google Colab (runtime > change runtime type > GPU) "
    "or a CUDA-enabled machine."
)

# Small offline calibration corpus, used when no concrete HF dataset id is given
# (the default "pileval" sentinel). Keeps the smoke test deterministic and
# network-free; real runs should pass a proper dataset id.
_DEFAULT_CALIB_SAMPLES = (
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "The mitochondrion is the powerhouse of the cell.",
    "Energy equals mass times the speed of light squared.",
    "To be, or not to be, that is the question.",
    "The rain in Spain stays mainly in the plain.",
    "A language model predicts the next token from the preceding context.",
    "Quantization reduces the numerical precision of a model's weights.",
    "Gradient descent iteratively minimizes a differentiable loss function.",
    "Attention lets each token attend to every other token in the sequence.",
    "The capital of France is Paris, a city on the river Seine.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
    "Supply and demand jointly determine the equilibrium market price.",
    "A binary search halves the search space on every comparison.",
)


def quantize_model(
    model_name_or_path: str,
    method: str,
    output_path: str,
    calibration_dataset: str = "pileval",
) -> str:
    """Quantize a model using the specified method.

    Args:
        model_name_or_path: HF model id or local path.
        method: One of ``"int8"``, ``"fp8"``, ``"awq"``, ``"gptq"``.
        output_path: Destination directory for the quantized model.
        calibration_dataset: HF dataset id for calibration (``awq``/``gptq``). The
            default ``"pileval"`` selects a small built-in offline corpus; pass a
            real dataset id for production runs. Ignored by ``int8``.

    Returns:
        The path to the quantized model directory.

    Raises:
        ValueError: If ``method`` is not supported.
        RuntimeError: If no CUDA GPU is available (or, for ``fp8``, no Hopper GPU).
    """
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported quantization method {method!r}. Expected one of {_SUPPORTED_METHODS}."
        )
    if not is_cuda_available():
        raise RuntimeError(_GPU_REQUIRED_MSG)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Quantizing %s with method=%s -> %s", model_name_or_path, method, out_dir)
    dispatch = {
        "awq": _quantize_awq,
        "gptq": _quantize_gptq,
        "int8": _quantize_int8,
        "fp8": _quantize_fp8,
    }
    dispatch[method](model_name_or_path, out_dir, calibration_dataset)

    logger.info("Quantization complete: %s", out_dir)
    return str(out_dir)


def _calibration_texts(calibration_dataset: str, n: int) -> list[str]:
    """Return up to ``n`` calibration text samples.

    If ``calibration_dataset`` names a real HF dataset, stream its first text
    column; otherwise (the default ``"pileval"`` sentinel) fall back to the
    built-in offline corpus, cycled to length ``n``.

    Args:
        calibration_dataset: HF dataset id, or ``"pileval"`` for the offline corpus.
        n: Desired number of samples.

    Returns:
        A list of calibration strings.
    """
    if calibration_dataset and calibration_dataset != "pileval":
        from datasets import load_dataset

        stream = load_dataset(calibration_dataset, split="train", streaming=True)
        texts: list[str] = []
        for row in stream:
            text = row.get("text") or next(
                (v for v in row.values() if isinstance(v, str) and v.strip()), None
            )
            if text:
                texts.append(text)
            if len(texts) >= n:
                break
        if texts:
            return texts
        logger.warning("Dataset %r yielded no text; using built-in corpus.", calibration_dataset)

    base = _DEFAULT_CALIB_SAMPLES
    return [base[i % len(base)] for i in range(n)]


def _quantize_awq(model_name_or_path: str, out_dir: Path, calibration_dataset: str) -> None:
    """Quantize to 4-bit AWQ (W4A16) via ``llmcompressor`` and save to ``out_dir``.

    Args:
        model_name_or_path: HF model id or local path.
        out_dir: Destination directory.
        calibration_dataset: HF dataset id, or ``"pileval"`` for the offline corpus.
    """
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    texts = _calibration_texts(calibration_dataset, n=128)
    calib = Dataset.from_dict({"text": texts})
    recipe = AWQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])

    oneshot(
        model=model_name_or_path,
        dataset=calib,
        recipe=recipe,
        output_dir=str(out_dir),
        max_seq_length=512,
        num_calibration_samples=len(texts),
    )


def _quantize_gptq(model_name_or_path: str, out_dir: Path, calibration_dataset: str) -> None:
    """Quantize to 4-bit GPTQ via ``gptqmodel`` and save to ``out_dir``.

    Args:
        model_name_or_path: HF model id or local path.
        out_dir: Destination directory.
        calibration_dataset: HF dataset id, or ``"pileval"`` for the offline corpus.
    """
    from gptqmodel import GPTQModel, QuantizeConfig

    texts = _calibration_texts(calibration_dataset, n=256)
    quant_config = QuantizeConfig(bits=4, group_size=128)
    model = GPTQModel.load(model_name_or_path, quant_config)
    model.quantize(texts)
    model.save(str(out_dir))


def _quantize_int8(model_name_or_path: str, out_dir: Path, calibration_dataset: str) -> None:
    """Quantize to INT8 via bitsandbytes LLM.int8() (calibration-free).

    Args:
        model_name_or_path: HF model id or local path.
        out_dir: Destination directory.
        calibration_dataset: Unused — LLM.int8() needs no calibration pass.
    """
    del calibration_dataset  # not used by bitsandbytes LLM.int8()
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto", quantization_config=bnb_config
    )

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))


def _quantize_fp8(model_name_or_path: str, out_dir: Path, calibration_dataset: str) -> None:
    """Reject FP8 unless running on a Hopper (SM 9.0+) GPU.

    FP8 weight quantization is provided by NVIDIA Transformer Engine and is only
    meaningful on H100-class hardware — not the T4/A100 GPUs available on Colab.

    Args:
        model_name_or_path: HF model id or local path.
        out_dir: Destination directory.
        calibration_dataset: Unused.

    Raises:
        RuntimeError: Always on non-Hopper GPUs; FP8 is not implemented for the
            Colab-available T4/A100 tiers.
    """
    del out_dir, calibration_dataset
    import torch

    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(
            f"fp8 quantization requires an H100-class (Hopper, SM 9.0+) GPU, but the "
            f"current device reports SM {major}.x. Use awq/gptq/int8 on T4/A100 instead."
        )
    raise NotImplementedError(
        "fp8 quantization on Hopper via Transformer Engine is not wired up yet "
        f"(model={model_name_or_path!r})."
    )
