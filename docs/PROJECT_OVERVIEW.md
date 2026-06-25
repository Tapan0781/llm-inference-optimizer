# LLM Inference Optimizer — A Plain-English Overview

*A guide anyone can follow, no technical background required.*

---

## 1. The problem, in everyday terms

A modern AI language model (the kind behind chatbots) is like a brilliant but
slow-to-respond expert. It knows a great deal, but every time you ask it
something, it has to do an enormous amount of mental arithmetic to produce each
word of its answer. On a computer, that arithmetic is **slow** and **expensive** —
it ties up costly graphics processors (GPUs) and makes you wait.

Two questions matter to anyone running these models:

- **How do we make the model answer faster and serve more people at once?**
- **How do we *prove* it actually got faster — with real numbers, not guesses?**

This project answers both.

---

## 2. What this project is

It is an **assembly line** for taking a large AI model and making it run faster on
a GPU — plus a **measurement lab** that times everything precisely so the speed-ups
are proven, not claimed.

Think of it like a workshop that takes a fast-but-thirsty car engine and tunes it
for efficiency, then puts it on a dynamometer to measure the exact horsepower gain.

---

## 3. How it works — the six stations on the assembly line

Each stage takes the model one step closer to running faster. The output of one
stage feeds the next.

1. **Export** — Convert the model from its original format into a standard,
   portable one (ONNX) that optimization tools understand. *Like translating a
   recipe into a universal format any kitchen can use.*

2. **Optimize** — Two techniques:
   - **TensorRT engine build**: re-compile the model specifically for the GPU it
     will run on, so it uses the hardware as efficiently as possible.
   - **Quantization**: shrink the model by storing its numbers with less precision
     (e.g. 4-bit instead of 16-bit). *Like compressing a huge photo to a smaller
     file that still looks fine* — the model gets ~3x smaller and faster, with
     little quality loss.

3. **Serve** — A single, unified way to actually run the model, with three
   interchangeable "engines" under the hood:
   - **eager** — the plain, standard way (the baseline to beat).
   - **ONNX** — runs the exported, portable model.
   - **vLLM** — a state-of-the-art server that cleverly handles many requests at
     once (this is the fast one).

4. **Profile** — Put a stopwatch and a power meter on the model: how long until the
   first word appears, how fast each following word comes, and how much electricity
   the GPU draws.

5. **Benchmark** — Run the model under many different workloads (small vs large
   batches of requests, short vs long inputs) and record every result to a
   spreadsheet automatically.

6. **Analyze** — Turn those spreadsheets into charts and a clear bottom line:
   *how much faster did we get?*

---

## 4. What was achieved

Everything was built **and proven on a real GPU** (a free Google Colab T4),
end to end. Highlights from the measured results:

- **~3x more throughput** and **~65% less waiting per word** when using the
  vLLM engine versus the standard baseline — on identical hardware and model
  (peaking at ~1,150 words per second).
- **~24x more throughput** from "batching" alone: serving many requests together
  instead of one at a time (28 -> 686 words per second) — the model does far more
  useful work per second when kept busy.
- **A model made ~3x smaller** through quantization (a 2.2 GB model shrunk to
  ~0.76 GB) while still running correctly.
- **Real power and memory measurements** captured automatically (e.g. ~45-67
  watts of GPU draw under load) — so efficiency, not just speed, is quantified.

The numbers above come from a small model on modest free hardware; the *same
pipeline* is built to run a much larger model on a powerful GPU, where the gains
are larger still.

---

## 5. The hard part (and why it shows real skill)

Building the pipeline was only half the work. The other half was making it
actually run on real machines, which fought back constantly:

- **Mismatched software versions.** The newest tools for one stage demanded a
  different version of a shared library than the tools for another stage. The
  solution was to give each stage its **own clean workspace** instead of forcing
  everything into one — the way a kitchen separates prep, cooking, and plating.

- **GPU driver mismatches.** Several libraries shipped builds for a *newer* GPU
  platform than the test machine had, causing cryptic "file not found" crashes.
  Each was diagnosed and pinned to the correct matching version.

- **A multi-layer puzzle to get the fast (vLLM) engine running.** It took
  unwinding five separate issues in sequence — wrong package build, a missing
  system file, a library-loading path problem, and a process-startup conflict —
  before it worked. The fixes were then **baked into the code so it self-heals**
  and nobody has to repeat that ordeal.

Every one of these was found by *actually running the system*, fixed, and locked
in with automated tests so it stays fixed.

---

## 6. How quality was kept high

- **Automated tests** run on every change, on an ordinary laptop, catching
  mistakes early.
- The trickier GPU-only pieces were each **validated on a real GPU** before being
  marked "done" — nothing was assumed to work.
- Every fix and decision is **documented and version-controlled**, so the project
  is reproducible and the reasoning is preserved.

---

## 7. Why it matters

Faster, cheaper AI inference means lower running costs, snappier responses for
users, and the ability to serve more people with the same hardware. This project
delivers a complete, measured, reproducible path to that — and the evidence to
back it up.

---

*Tech stack: PyTorch, CUDA, TensorRT, ONNX, vLLM, NVML. Status: end-to-end
pipeline complete and validated on GPU. (This document is generated from
`scripts/make_overview_pdf.py`.)*
