#!/usr/bin/env python3
"""Render docs/architecture.png — the project's pipeline architecture diagram.

Pure matplotlib (no system deps). Run:  python scripts/make_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"

NAVY = "#172a4d"
BLUE = "#2563af"
LIGHT = "#eaf1fb"
GREY = "#5a5a5a"
GREEN = "#1f7a4d"
LIGHTGREEN = "#e7f4ee"

# Stage boxes: (id, title, detail, folder, x, y)
BOX_W, BOX_H = 4.2, 2.5
STAGES = [
    ("1. Export", "HuggingFace -> ONNX\n(Optimum, KV-cache)", "src/export", 0.5, 7.3),
    ("2. Optimize", "TensorRT engine +\nQuantization (AWQ/GPTQ/INT8)", "src/optimization", 5.4, 7.3),
    ("3. Serve", "InferenceEngine\neager / ONNX / vLLM", "src/serving", 10.3, 7.3),
    ("4. Profile", "TTFT / TPOT / throughput\nNVML power + memory", "src/profiling", 10.3, 3.1),
    ("5. Benchmark", "sweep grid -> CSV/JSON\n+ MFU", "src/benchmarking", 5.4, 3.1),
    ("6. Analyze", "charts + speedups", "notebook 04", 0.5, 3.1),
]


def box(ax, x, y, title, detail, folder, face=LIGHT, edge=BLUE) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), BOX_W, BOX_H, boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=2, edgecolor=edge, facecolor=face, zorder=2,
        )
    )
    cx = x + BOX_W / 2
    ax.text(cx, y + BOX_H - 0.55, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color=NAVY, zorder=3)
    ax.text(cx, y + BOX_H / 2 - 0.18, detail, ha="center", va="center",
            fontsize=9.5, color="#222", zorder=3)
    ax.text(cx, y + 0.32, folder, ha="center", va="center",
            fontsize=8.5, style="italic", color=GREY, zorder=3)


def arrow(ax, x1, y1, x2, y2) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20,
            linewidth=2, color=GREY, zorder=1,
        )
    )


def label(ax, x, y, text) -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=BLUE,
            fontweight="bold", zorder=4,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"})


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # Title
    ax.text(7.5, 10.5, "LLM Inference Optimizer - Pipeline Architecture",
            ha="center", fontsize=18, fontweight="bold", color=NAVY)
    ax.text(7.5, 10.0, "export -> optimize -> serve -> profile -> benchmark -> analyze "
            "(each stage validated on a real GPU)",
            ha="center", fontsize=10.5, color=GREY)

    for title, detail, folder, x, y in STAGES:
        box(ax, x, y, title, detail, folder)

    # Centers helper
    def c(i: int) -> tuple[float, float]:
        _, _, _, x, y = STAGES[i]
        return x + BOX_W / 2, y + BOX_H / 2

    # Input -> stage 1
    ax.text(2.6, 9.9, "HuggingFace model\n(LLaMA 3 / TinyLlama)", ha="center",
            fontsize=9.5, color=NAVY, fontweight="bold")
    arrow(ax, 2.6, 9.55, 2.6, 7.3 + BOX_H)

    # Flow arrows (snake): 1->2->3 (top, L->R), 3->4 (down), 4->5->6 (bottom, R->L)
    arrow(ax, 0.5 + BOX_W, 8.55, 5.4, 8.55)          # 1 -> 2
    label(ax, 4.9, 8.9, "model.onnx")
    arrow(ax, 5.4 + BOX_W, 8.55, 10.3, 8.55)         # 2 -> 3
    label(ax, 9.85, 8.9, ".engine /\nquantized")
    arrow(ax, 10.3 + BOX_W / 2, 7.3, 10.3 + BOX_W / 2, 3.1 + BOX_H)  # 3 -> 4 (down)
    label(ax, 10.3 + BOX_W / 2, 6.0, "generations")
    arrow(ax, 10.3, 4.35, 5.4 + BOX_W, 4.35)         # 4 -> 5
    label(ax, 9.85, 4.7, "metrics")
    arrow(ax, 5.4, 4.35, 0.5 + BOX_W, 4.35)          # 5 -> 6
    label(ax, 4.9, 4.7, "results")

    # Output from stage 6
    ax.text(2.6, 1.9, "Speedup charts & report\n(docs/benchmarks/)", ha="center",
            fontsize=9.5, color=GREEN, fontweight="bold")
    arrow(ax, 2.6, 3.1, 2.6, 2.4)

    # Environment band
    ax.add_patch(
        FancyBboxPatch(
            (0.4, 0.15), 14.2, 0.95, boxstyle="round,pad=0.02,rounding_size=0.1",
            linewidth=1.5, edgecolor=GREEN, facecolor=LIGHTGREEN, zorder=0,
        )
    )
    ax.text(7.5, 0.83, "Runs in two places", ha="center", fontsize=9.5,
            fontweight="bold", color=GREEN)
    ax.text(
        7.5, 0.42,
        "Local (Mac, CPU): code + tests + eager backend      |      "
        "Colab GPU: gpu.txt (export/TRT/INT8) . gpu-quant.txt (AWQ/GPTQ) . gpu-serve.txt (vLLM)",
        ha="center", fontsize=8.7, color="#234",
    )

    fig.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
