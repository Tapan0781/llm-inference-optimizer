#!/usr/bin/env python3
"""Generate docs/PROJECT_OVERVIEW.pdf — a plain-English project explainer.

Pure-Python (fpdf2), no system dependencies. Run:  python scripts/make_overview_pdf.py
Keep this in sync with docs/PROJECT_OVERVIEW.md (same content, prose form).
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

NAVY = (23, 42, 77)
ACCENT = (37, 99, 175)
GREY = (90, 90, 90)


class Overview(FPDF):
    def header(self) -> None:  # noqa: D102
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, "LLM Inference Optimizer - Plain-English Overview", align="L")
        self.ln(10)

    def footer(self) -> None:  # noqa: D102
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def h1(pdf: FPDF, text: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def body(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.5)


def bullet(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(20, 20, 20)
    pdf.set_x(pdf.l_margin + 4)
    pdf.multi_cell(0, 6, f"-  {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(0.5)


def main() -> None:
    pdf = Overview()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # ---- Title block ----
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(0, 11, "LLM Inference Optimizer", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(*ACCENT)
    pdf.multi_cell(0, 8, "A Plain-English Overview", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 6, "A guide anyone can follow - no technical background required.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    h1(pdf, "1. The problem, in everyday terms")
    body(
        pdf,
        "A modern AI language model (the kind behind chatbots) is like a brilliant but "
        "slow-to-respond expert. It knows a great deal, but every time you ask it something "
        "it must do an enormous amount of mental arithmetic to produce each word. On a "
        "computer that arithmetic is slow and expensive - it ties up costly graphics "
        "processors (GPUs) and makes you wait. Two questions matter to anyone running these "
        "models:",
    )
    bullet(pdf, "How do we make the model answer faster and serve more people at once?")
    bullet(pdf, "How do we prove it actually got faster - with real numbers, not guesses?")
    body(pdf, "This project answers both.")

    h1(pdf, "2. What this project is")
    body(
        pdf,
        "It is an assembly line for taking a large AI model and making it run faster on a "
        "GPU, plus a measurement lab that times everything precisely so the speed-ups are "
        "proven, not claimed. Think of a workshop that tunes a fast-but-thirsty car engine "
        "for efficiency, then puts it on a dynamometer to measure the exact horsepower gain.",
    )

    h1(pdf, "3. How it works - the six stations on the assembly line")
    body(pdf, "Each stage takes the model one step closer to running faster; each feeds the next.")
    bullet(
        pdf,
        "Export - convert the model into a standard, portable format (ONNX) that "
        "optimization tools understand. Like translating a recipe into a universal format.",
    )
    bullet(
        pdf,
        "Optimize - re-compile the model for the exact GPU (TensorRT), and shrink it by "
        "storing numbers with less precision (quantization). Like compressing a huge photo "
        "into a smaller file that still looks fine: ~3x smaller and faster.",
    )
    bullet(
        pdf,
        "Serve - one unified way to run the model, with three interchangeable engines: "
        "eager (the standard baseline), ONNX (the portable build), and vLLM (a "
        "state-of-the-art server that handles many requests at once - the fast one).",
    )
    bullet(
        pdf,
        "Profile - put a stopwatch and a power meter on the model: time to first word, "
        "speed of each following word, and how much electricity the GPU draws.",
    )
    bullet(
        pdf,
        "Benchmark - run the model under many workloads (small vs large batches, short vs "
        "long inputs) and record every result to a spreadsheet automatically.",
    )
    bullet(pdf, "Analyze - turn those spreadsheets into charts and a clear bottom line.")

    h1(pdf, "4. What was achieved")
    body(
        pdf,
        "Everything was built and proven on a real GPU (a free Google Colab T4), end to "
        "end. Highlights from the measured results:",
    )
    bullet(
        pdf,
        "~2.3x more throughput and ~57% less waiting per word using the vLLM engine versus "
        "the standard baseline - identical hardware and model.",
    )
    bullet(
        pdf,
        "~24x more throughput from batching alone: serving 32 requests together instead of "
        "one at a time (28 -> 686 words per second).",
    )
    bullet(
        pdf,
        "A model made ~3x smaller via quantization (2.2 GB shrunk to ~0.76 GB) while still "
        "running correctly.",
    )
    bullet(
        pdf,
        "Real power and memory captured automatically (~45-67 watts under load) - so "
        "efficiency, not just speed, is quantified.",
    )
    body(
        pdf,
        "These numbers come from a small model on modest free hardware; the same pipeline "
        "is built to run a much larger model on a powerful GPU, where the gains are larger.",
    )

    h1(pdf, "5. The hard part (and why it shows real skill)")
    body(
        pdf,
        "Building the pipeline was only half the work. The other half was making it run on "
        "real machines, which fought back constantly:",
    )
    bullet(
        pdf,
        "Mismatched software versions - different stages needed conflicting versions of a "
        "shared library. Fixed by giving each stage its own clean workspace.",
    )
    bullet(
        pdf,
        "GPU driver mismatches - several libraries shipped builds for a newer GPU platform "
        "than the machine had, causing cryptic crashes. Each was pinned to the right build.",
    )
    bullet(
        pdf,
        "A multi-layer puzzle to get the fast (vLLM) engine running - five separate issues "
        "unwound in sequence, then baked into the code so it self-heals.",
    )
    body(
        pdf,
        "Every one of these was found by actually running the system, fixed, and locked in "
        "with automated tests so it stays fixed.",
    )

    h1(pdf, "6. How quality was kept high")
    bullet(pdf, "Automated tests run on every change, on an ordinary laptop, catching mistakes early.")
    bullet(pdf, "Every GPU-only piece was validated on a real GPU before being marked done.")
    bullet(pdf, "Every fix and decision is documented and version-controlled - fully reproducible.")

    h1(pdf, "7. Why it matters")
    body(
        pdf,
        "Faster, cheaper AI inference means lower running costs, snappier responses, and "
        "serving more people with the same hardware. This project delivers a complete, "
        "measured, reproducible path to that - and the evidence to back it up.",
    )

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(*GREY)
    pdf.multi_cell(
        0,
        5,
        "Tech stack: PyTorch, CUDA, TensorRT, ONNX, vLLM, NVML.  "
        "Status: end-to-end pipeline complete and validated on GPU.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    out = Path(__file__).resolve().parents[1] / "docs" / "PROJECT_OVERVIEW.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
