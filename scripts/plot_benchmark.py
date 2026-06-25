#!/usr/bin/env python3
"""Render docs/benchmarks/backend_comparison.png from a benchmark CSV.

Reads docs/benchmarks/tinyllama_t4_seq128.csv (eager vs vLLM, throughput + TPOT
by batch size) and draws a two-panel comparison chart for the README.

Run:  python scripts/plot_benchmark.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs" / "benchmarks" / "tinyllama_t4_seq128.csv"
OUT = ROOT / "docs" / "benchmarks" / "backend_comparison.png"

COLORS = {"eager": "#9aa0a6", "vllm": "#2563af"}


def load() -> dict[str, dict[str, list[float]]]:
    """Load the CSV grouped by backend into {backend: {batch, tps, tpot}} lists."""
    data: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"batch": [], "tps": [], "tpot": []}
    )
    with CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            d = data[row["backend"]]
            d["batch"].append(float(row["batch_size"]))
            d["tps"].append(float(row["throughput_tps"]))
            d["tpot"].append(float(row["tpot_ms"]))
    return data


def main() -> None:
    data = load()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for backend, d in sorted(data.items()):
        label = "vLLM" if backend == "vllm" else "eager (baseline)"
        color = COLORS.get(backend, None)
        ax1.plot(d["batch"], d["tps"], marker="o", label=label, color=color, linewidth=2)
        ax2.plot(d["batch"], d["tpot"], marker="o", label=label, color=color, linewidth=2)

    ax1.set_title("Throughput (higher is better)")
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("tokens / second")
    ax2.set_title("Per-token latency (lower is better)")
    ax2.set_xlabel("batch size")
    ax2.set_ylabel("TPOT (ms)")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xscale("log", base=2)
        ax.set_xticks(data["eager"]["batch"])
        ax.set_xticklabels([str(int(b)) for b in data["eager"]["batch"]])

    fig.suptitle(
        "TinyLlama-1.1B on a free Colab T4 (seq=128, 128 output tokens): eager vs vLLM",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
