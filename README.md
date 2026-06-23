# llm-inference-optimizer

End-to-end GPU-accelerated LLM inference optimization pipeline with a benchmarking
framework, targeting **LLaMA 3 (8B / 70B)**.

**Stack:** PyTorch 2.3 · CUDA · TensorRT · ONNX · vLLM · Nsight

## Goals

- 45% latency reduction vs baseline PyTorch eager
- 2.3× throughput improvement
- GPU MFU from ~25% → 60–75%
- Full benchmark sweep: TTFT, TPOT, tokens/sec, MFU, power (NVML)

## Two-environment model

| Environment | Used for | Install |
|---|---|---|
| **Local (Mac, CPU)** | code, configs, tests, git | `make setup-local` (`requirements/base.txt` + `dev.txt`) |
| **Google Colab (GPU)** | export, TRT build, quantization, benchmarking | `requirements/gpu.txt`, start at `notebooks/00_colab_setup.ipynb` |

Every GPU code path guards itself with `is_cuda_available()` and fails loudly on CPU —
it never silently falls back.

## Quickstart (local)

```bash
make setup-local     # install CPU deps + dev tooling
make lint            # ruff + black --check + mypy
make test            # CPU-safe unit tests
make format          # auto-fix formatting
```

## Layout

```
configs/      model + benchmark sweep configs
src/utils/    env detection + logging   (Phase 1 — done)
src/export/   HuggingFace → ONNX        (Phase 2)
src/optimization/  quantization + TRT   (Phase 3)
src/serving/  unified inference engine  (Phase 4)
src/profiling/  PyTorch profiler + NVML (Phase 5)
src/benchmarking/  sweep runner + MFU   (Phase 6)
notebooks/    Colab GPU workflows
tests/unit/         CPU-safe, run in CI
tests/integration/  GPU-only, auto-skipped on CPU
```

## Project phase

**Phase 1 — Scaffold + environment setup** (current). Repo structure, configs, CI,
and `src/utils/env.py` are in place; everything runs cleanly on Mac (CPU only).
See `CLAUDE.md` for full module contracts and the phase roadmap.
