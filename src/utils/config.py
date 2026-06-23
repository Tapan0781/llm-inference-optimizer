"""Config loading utilities.

Loads model and benchmark configs from the ``configs/`` tree so callers never
hardcode model ids or sweep parameters. CPU-safe: pure YAML parsing, no GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root resolved from this file: src/utils/config.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_DIR = _REPO_ROOT / "configs" / "model_configs"
BENCHMARK_CONFIG_DIR = _REPO_ROOT / "configs" / "benchmark_configs"

# Keys every model config must define (matches the schema in CLAUDE.md).
_REQUIRED_MODEL_KEYS: tuple[str, ...] = (
    "model_id",
    "architecture",
    "num_parameters_b",
    "dtype",
    "max_seq_len",
    "vocab_size",
    "num_layers",
    "num_heads",
    "hidden_dim",
    "tp_degree",
    "hf_token_required",
)


def _resolve(name_or_path: str, config_dir: Path) -> Path:
    """Resolve a config reference to a concrete YAML path.

    Accepts a bare name (``"llama3_8b"``), a filename (``"llama3_8b.yaml"``),
    or an explicit filesystem path.

    Args:
        name_or_path: Config reference.
        config_dir: Directory to search for bare names / filenames.

    Returns:
        The resolved path to an existing YAML file.

    Raises:
        FileNotFoundError: If no matching config file exists.
    """
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        return candidate

    stem = candidate.name.removesuffix(".yaml").removesuffix(".yml")
    for ext in (".yaml", ".yml"):
        path = config_dir / f"{stem}{ext}"
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No config found for {name_or_path!r} in {config_dir}. "
        f"Available: {sorted(p.name for p in config_dir.glob('*.y*ml'))}"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Args:
        path: Path to a YAML file.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If the file does not contain a top-level mapping.
    """
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(
            f"Config {path} must contain a top-level mapping, got {type(data).__name__}."
        )
    return data


def load_model_config(name_or_path: str) -> dict[str, Any]:
    """Load and validate a model config from ``configs/model_configs/``.

    Args:
        name_or_path: ``"llama3_8b"``, ``"llama3_8b.yaml"``, or a path.

    Returns:
        The validated model config mapping.

    Raises:
        FileNotFoundError: If the config does not exist.
        ValueError: If required keys are missing.
    """
    path = _resolve(name_or_path, MODEL_CONFIG_DIR)
    config = _load_yaml(path)

    missing = [key for key in _REQUIRED_MODEL_KEYS if key not in config]
    if missing:
        raise ValueError(f"Model config {path.name} is missing required keys: {missing}")
    return config


def load_benchmark_config(name_or_path: str = "default_sweep") -> dict[str, Any]:
    """Load a benchmark sweep config from ``configs/benchmark_configs/``.

    Args:
        name_or_path: ``"default_sweep"``, ``"default_sweep.yaml"``, or a path.

    Returns:
        The benchmark config mapping.

    Raises:
        FileNotFoundError: If the config does not exist.
    """
    path = _resolve(name_or_path, BENCHMARK_CONFIG_DIR)
    return _load_yaml(path)
