# llm-inference-optimizer

End-to-end GPU-accelerated LLM inference optimization pipeline with a benchmarking
framework, targeting **LLaMA 3 (8B / 70B)**.

**Stack:** PyTorch 2.3 · CUDA · TensorRT · ONNX · vLLM · Nsight

## Goals

- 45% latency reduction vs baseline PyTorch eager
- 2.3× throughput improvement
- GPU MFU from ~25% → 60–75%
- Full benchmark sweep: TTFT, TPOT, tokens/sec, MFU, power (NVML)

## Pipeline at a glance

```
HuggingFace LLaMA 3
   │  export_to_onnx          (Phase 2)  →  model.onnx (decoder + KV-cache)
   ▼
ONNX graph
   │  build_trt_engine        (Phase 3)  →  .engine   (TensorRT)
   │  quantize_model          (Phase 3)  →  AWQ / GPTQ / INT8 checkpoint
   ▼
InferenceEngine               (Phase 4)  →  eager / onnx / vllm backends
   │  profiler                (Phase 5)  →  latency + NVML power
   ▼
benchmark sweep               (Phase 6)  →  TTFT / TPOT / tok-s / MFU  →  plots (Phase 7)
```

Each stage is chained by **artifacts on disk** (an ONNX dir, a `.engine`, a quantized
checkpoint), which is what lets the stages run in separate environments (see below).

## Environment model

Development is local (CPU); all GPU work runs on Colab. The GPU side is split into
**three environments** because the stages need mutually incompatible dependency
pins — they are distinct pipeline stages chained by on-disk artifacts, so this is
correct structure, not a workaround.

| Environment | Used for | Install |
|---|---|---|
| **Local (Mac, CPU)** | code, configs, tests, git, the `eager` backend | `make setup-local` (`requirements/base.txt` + `dev.txt`) |
| **Colab GPU — main** | ONNX export, TensorRT build, INT8 quant | `requirements/gpu.txt` (transformers <5.0) |
| **Colab GPU — quant** | AWQ + GPTQ | `requirements/gpu-quant.txt` (transformers 5.x: `llmcompressor`, `gptqmodel`) |
| **Colab GPU — serve** | vLLM serving backend | `requirements/gpu-serve.txt` (vLLM pins its own torch/transformers) |

Every GPU code path guards itself with `is_cuda_available()` and fails loudly on CPU —
it never silently falls back. Start GPU sessions from `notebooks/00_colab_setup.ipynb`.

> ⚠️ **Do not mix the three GPU environments in one kernel** — `optimum` (export)
> needs transformers <5.0, while `llmcompressor`/`gptqmodel` need 5.x, and vLLM pins
> its own. Use a separate Colab session per stage.

## Quickstart (local)

```bash
make setup-local     # install CPU deps + dev tooling
make lint            # ruff + black --check + mypy
make test            # CPU-safe unit tests (incl. real eager generation)
make format          # auto-fix formatting
```

## Module layout & phase status

```
configs/             model + benchmark sweep configs
src/utils/           env detection, logging, config loaders   ✅ Phase 1 (done)
src/export/          HuggingFace → ONNX (optimum, KV-cache)    ✅ Phase 2 (done, Colab-validated)
src/optimization/    TensorRT engine build + quantization      ✅ Phase 3 (done, Colab-validated)
src/serving/         unified InferenceEngine                   ✅ Phase 4 (eager/onnx/vllm validated)
src/profiling/       PyTorch profiler + NVML power             ⏳ Phase 5
src/benchmarking/    sweep runner + MFU                        ⏳ Phase 6 (MFU + result schema done)
notebooks/           Colab GPU workflows (00–04)
tests/unit/          CPU-safe, run in CI
tests/integration/   GPU-only, auto-skipped on CPU
```

## Roadmap

- ✅ **Phase 1 — Scaffold + environment.** Repo, configs, CI, `env.py`/`logger.py`,
  config loaders, `calculate_mfu` + `BenchmarkResult` (CPU-safe).
- ✅ **Phase 2 — ONNX export.** `export_to_onnx` via HF Optimum
  (`text-generation-with-past`, KV-cache + dynamic axes). PyTorch-vs-ONNX parity
  verified on a tiny model (Colab T4).
- ✅ **Phase 3 — TensorRT + quantization.** *Validated on Colab.*
  - `build_trt_engine` (fp16/fp32): parses the ONNX, builds one optimization
    profile over batch/sequence/past-length, serializes + verifies the `.engine`.
  - `quantize_model`: **INT8** (bitsandbytes), **AWQ** (llmcompressor),
    **GPTQ** (gptqmodel); **FP8** gated to H100. INT8/FP8 TRT-engine precisions
    are deferred (need a calibrator / ONNX Q-DQ).
- ✅ **Phase 4 — Serving runtime.** *Validated.* `InferenceEngine` with a single
  `generate()` contract over backends:
  - **eager** (HF `.generate`, CPU+GPU) — unit-tested on CPU (real generation).
  - **onnx** (optimum `ORTModelForCausalLM`) — validated on Colab.
  - **vllm** (continuous batching + chunked prefill) — validated on Colab
    (TinyLlama-1.1B). Self-heals Colab's cu13-vLLM-on-cu12-torch runtime
    (lib preload + `LD_LIBRARY_PATH` + spawn workers).
  - **trt** serving (hand-rolled decode loop) and **Medusa** speculative decoding
    are deferred to follow-ups.
- ⏳ **Phase 5 — Profiling.** PyTorch profiler + NVML power wrapper.
- ⏳ **Phase 6 — Benchmarking.** Full sweep (`run_sweep`): TTFT, TPOT, tokens/sec,
  MFU, power → CSV/JSON.
- ⏳ **Phase 7 — Nsight.** Requires bare-metal GPU (Lambda Labs / RunPod).

## Engineering notes (hard-won, GPU-validated)

These surfaced during real Colab runs and are baked into the code & requirements:

- **TensorRT 11** removed the precision `BuilderFlag`s (`FP16`/`INT8`/`FP8`),
  `EXPLICIT_BATCH`, and `platform_has_fast_fp16`. Precision now comes from the ONNX
  tensor types via a **strongly-typed network**; the builder detects this and adapts.
- **Install `tensorrt-cu12`, not bare `tensorrt`** (the latter pulls a CUDA-13 build →
  `libcudart.so.13` mismatch on Colab's CUDA 12). Same trap on `onnxruntime-gpu` — the
  export uses **CPU `onnxruntime`** (the parity check is correctness-only).
- **`autoawq` and `auto-gptq` are dead** (import-break / build-fail on current stacks);
  replaced by `llmcompressor` / `gptqmodel`, which require transformers 5.x — hence the
  separate quant environment.
- **Quantization needs a real model, not the tiny test model** — AWQ/GPTQ group quant
  (`group_size=128`) requires weight dims divisible by 128. Validation uses
  `TinyLlama-1.1B`; `tiny-random` is only for export/TRT graph structure.

## More

See **`CLAUDE.md`** for full module contracts, config schemas, coding standards, and
the detailed phase log.
