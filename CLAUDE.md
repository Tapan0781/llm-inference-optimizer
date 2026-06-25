# CLAUDE.md — LLM Inference Optimizer

This file gives Claude Code full project context. Read this before touching any file.

---

## Project identity

**Name**: llm-inference-optimizer
**Goal**: End-to-end GPU-accelerated LLM inference optimization pipeline with benchmarking framework
**Target model**: LLaMA 3 (8B and 70B)
**Stack**: PyTorch 2.3, CUDA, TensorRT, ONNX, vLLM, Nsight

**Key results to achieve**:
- 45% latency reduction vs baseline PyTorch eager
- 2.3× throughput improvement
- GPU MFU from ~25% → 60–75%
- Full benchmark sweep: TTFT, TPOT, tokens/sec, MFU, power (NVML)

---

## Environment model

This project runs in TWO environments. Every decision must respect both.

### Local (Mac, CPU only)
- Used for: writing code, scaffolding, configs, tests, git operations
- No CUDA, no TensorRT, no GPU
- Python 3.11, install from `requirements/base.txt` + `requirements/dev.txt`
- Run with: `make setup-local`

### Google Colab (GPU execution)
- Used for: ONNX export, TRT engine build, quantization, benchmarking
- GPU: T4 (free) or A100 (Pro/Pro+)
- Entry point: `notebooks/00_colab_setup.ipynb`
- **The GPU stack is split across TWO environments** (they need incompatible
  transformers majors and are separate pipeline stages chained by on-disk
  checkpoints — do not try to install both in one kernel):
  - `requirements/gpu.txt` — export / TensorRT / INT8 (transformers <5.0)
  - `requirements/gpu-quant.txt` — AWQ + GPTQ only (transformers 5.x, via
    `llmcompressor` + `gptqmodel`)
  - `requirements/gpu-serve.txt` — vLLM serving backend (pins its own
    torch/transformers; continuous batching + chunked prefill)

### Rule: every GPU code path must guard itself

```python
from src.utils.env import is_cuda_available

if not is_cuda_available():
    raise RuntimeError(
        "GPU required for this operation. "
        "Run on Google Colab (runtime > change runtime type > GPU) "
        "or a CUDA-enabled machine."
    )
```

Never silently fall back on CPU for GPU-only operations. Fail loudly with a clear message.

---

## Repository structure

```
llm-inference-optimizer/
├── CLAUDE.md                        ← you are here
├── README.md
├── pyproject.toml                   # setuptools, ruff, black, mypy config
├── Makefile                         # setup-local, lint, test, test-all, clean
├── .gitignore
├── .pre-commit-config.yaml          # black + ruff + mypy hooks
│
├── .github/workflows/
│   ├── ci.yml                       # lint + unit tests on every push (CPU, ubuntu-latest)
│   └── gpu_tests.yml                # integration tests on self-hosted GPU runner only
│
├── requirements/
│   ├── base.txt                     # CPU-safe: torch (cpu), transformers, onnx, pyyaml
│   ├── gpu.txt                      # GPU: torch (cu121), tensorrt, onnxruntime-gpu, vllm
│   └── dev.txt                      # black, ruff, mypy, pytest, ipykernel, pre-commit
│
├── configs/
│   ├── model_configs/
│   │   ├── llama3_8b.yaml
│   │   └── llama3_70b.yaml
│   └── benchmark_configs/
│       └── default_sweep.yaml
│
├── src/
│   ├── utils/
│   │   ├── env.py                   # is_cuda_available, is_colab, get_device, get_gpu_name
│   │   └── logger.py                # structured logging
│   ├── export/
│   │   └── onnx_exporter.py         # HuggingFace → ONNX export
│   ├── optimization/
│   │   ├── quantization.py          # INT8 / FP8 / AWQ / GPTQ wrappers
│   │   └── trt_builder.py           # TensorRT engine builder
│   ├── serving/
│   │   └── inference_engine.py      # unified backend: eager / ONNX / TRT
│   ├── profiling/
│   │   └── profiler.py              # PyTorch profiler + NVML power wrapper
│   └── benchmarking/
│       └── benchmark_runner.py      # sweep runner, saves results to results/
│
├── tests/
│   ├── unit/                        # CPU-safe, always run in CI
│   └── integration/                 # GPU-required, skipped if no CUDA
│
├── scripts/
│   ├── setup_local.sh
│   ├── setup_colab.sh
│   └── run_benchmark.sh
│
├── notebooks/
│   ├── 00_colab_setup.ipynb         # bootstrap: clone → install → verify GPU
│   ├── 01_export.ipynb              # LLaMA 3 → ONNX
│   ├── 02_optimize.ipynb            # quantization + TRT engine
│   ├── 03_benchmark.ipynb           # full sweep
│   └── 04_analysis.ipynb            # plots and results
│
└── results/                         # gitignored — benchmark outputs land here
```

---

## Module contracts

These are the exact interfaces. Do not change signatures without updating this file.

### `src/utils/env.py`

```python
def is_cuda_available() -> bool
    """Returns True if a CUDA GPU is accessible."""

def is_colab() -> bool
    """Returns True if running inside Google Colab."""

def get_device() -> str
    """Returns 'cuda' if GPU available, else 'cpu'."""

def get_gpu_name() -> str
    """Returns GPU model name, or 'CPU-only' if no GPU."""

def get_env_info() -> dict
    """Returns dict with: device, gpu_name, cuda_version, torch_version, is_colab."""
```

### `src/utils/config.py`

```python
def load_model_config(name_or_path: str) -> dict
    """Loads + validates a model config from configs/model_configs/.
    Accepts a bare name ('llama3_8b'), filename, or path. CPU-safe."""

def load_benchmark_config(name_or_path: str = "default_sweep") -> dict
    """Loads a sweep config from configs/benchmark_configs/. CPU-safe."""
```

Never read config YAML ad hoc — always go through these loaders.

### `src/export/onnx_exporter.py`

```python
def export_to_onnx(
    model_name_or_path: str,
    output_path: str,            # destination DIRECTORY for ONNX artifacts
    dtype: str = "fp16",          # "fp32" | "fp16"
    opset_version: int = 17,
    verify_export: bool = True,
) -> str
    """Exports HuggingFace LLM to ONNX. Returns path to model.onnx. GPU required."""
```

**Export decisions (Phase 2):**
- Backend: **HF Optimum** (`optimum.exporters.onnx.main_export`).
- Task: **`text-generation-with-past`** — decoder *with* KV-cache + dynamic
  axes. A cacheless export is useless for fast inference / the TRT build.
- `verify_export=True` runs PyTorch-vs-ONNXRuntime last-token logit parity;
  tolerances `fp32: atol=1e-4`, `fp16: atol=1e-2`.
- Dev/validate on the ungated tiny model
  `hf-internal-testing/tiny-random-LlamaForCausalLM` (free T4, no token);
  real 8B needs `HF_TOKEN` + A100.

### `src/optimization/trt_builder.py`

```python
def build_trt_engine(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",      # "fp32" | "fp16" | "int8" | "fp8"
    max_batch_size: int = 32,
    max_seq_len: int = 2048,
    workspace_gb: int = 8,
) -> str
    """Builds TensorRT engine from ONNX. Returns .engine path. GPU required."""
```

### `src/optimization/quantization.py`

```python
def quantize_model(
    model_name_or_path: str,
    method: str,                  # "int8" | "fp8" | "awq" | "gptq"
    output_path: str,
    calibration_dataset: str = "pileval",
) -> str
    """Quantizes model using specified method. Returns path to quantized model."""
```

### `src/serving/inference_engine.py`

```python
class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        backend: str = "eager",   # "eager" | "onnx" | "trt" | "vllm"
        dtype: str = "fp16",
        device: str = "auto",
    )
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
    ) -> list[str]

    def warmup(self, n_iters: int = 5) -> None
```

### `src/profiling/profiler.py`

```python
def get_power_watts() -> float
    """Instantaneous GPU board power via NVML, or -1.0 if unavailable."""

@dataclass
class ProfileResult:
    ttft_ms: float            # time to first token
    tpot_ms: float            # time per output token
    throughput_tps: float     # generated tokens/sec across the batch
    gpu_mem_gb: float         # peak CUDA memory (0.0 on CPU)
    power_watts: float        # mean board power (-1.0 if NVML unavailable)
    power_peak_watts: float   # peak board power (-1.0 if unavailable)

def profile_generation(
    engine,                   # InferenceEngine (any backend)
    prompts: list[str],
    max_new_tokens: int = 128,
    warmup_iters: int = 3,
) -> ProfileResult
    """Black-box latency/power/memory profile. TTFT = single-token-run latency;
    TPOT = marginal per-token cost from the full run. CUDA-synced timing.
    Runs on CPU (timing) with GPU-only metrics degrading to sentinels."""
```

### `src/benchmarking/benchmark_runner.py`

```python
@dataclass
class BenchmarkResult:
    model: str
    backend: str
    dtype: str
    batch_size: int
    seq_len: int
    ttft_ms: float          # time to first token
    tpot_ms: float          # time per output token
    throughput_tps: float   # tokens per second
    mfu_percent: float      # model FLOP utilization
    gpu_mem_gb: float
    power_watts: float      # from NVML, -1 if unavailable

def run_sweep(
    config_path: str,
    engine: InferenceEngine,
    output_dir: str = "results/",
) -> list[BenchmarkResult]
    """Runs full benchmark sweep from config. Saves CSV + JSON to output_dir."""
```

---

## Config schemas

### `configs/model_configs/llama3_8b.yaml`

```yaml
model_id: "meta-llama/Meta-Llama-3-8B-Instruct"
architecture: "llama"
num_parameters_b: 8
dtype: "fp16"
max_seq_len: 8192
vocab_size: 128256
num_layers: 32
num_heads: 32
hidden_dim: 4096
tp_degree: 1              # tensor parallel degree (1 = single GPU)
hf_token_required: true   # needs HF_TOKEN env var
```

### `configs/benchmark_configs/default_sweep.yaml`

```yaml
batch_sizes: [1, 4, 8, 16, 32]
seq_lens: [128, 512, 1024, 2048]
dtypes: ["fp16", "int8"]
backends: ["eager", "trt"]
warmup_iters: 5
benchmark_iters: 50
save_traces: false
output_format: ["csv", "json"]
```

---

## Coding standards

- **Python 3.11+** — use `match`, `tomllib`, `Self` type where appropriate
- **Type hints everywhere** — all function signatures, all class attributes
- **Docstrings** — Google style, every public function and class
- **Line length**: 100 characters (black + ruff configured in pyproject.toml)
- **Imports**: stdlib → third-party → local, separated by blank lines (ruff enforces)
- **No bare `except:`** — always catch specific exceptions
- **No `print()`** in `src/` — use `src.utils.logger` instead
- **GPU guards first** — check `is_cuda_available()` at the top of any GPU function, before any imports of tensorrt/pycuda
- **Test every public function** — unit tests in `tests/unit/` must be CPU-safe and runnable with `pytest` locally

---

## Makefile targets

```
make setup-local    # pip install requirements/base.txt requirements/dev.txt
make lint           # ruff check src/ tests/ + black --check + mypy src/
make test           # pytest tests/unit/ -v
make test-all       # pytest tests/ -v (skips GPU tests if no CUDA)
make clean          # remove __pycache__, .pytest_cache, dist/, *.egg-info
make format         # black src/ tests/ + ruff --fix src/ tests/
```

---

## GitHub Actions

### `ci.yml` — runs on every push and PR to `main`
- OS: `ubuntu-latest`
- Python: `3.11`
- Steps: checkout → install base+dev → `make lint` → `make test`
- Must pass before merge

### `gpu_tests.yml` — runs only on self-hosted runner tagged `gpu`
- Triggered manually or on push to `gpu-*` branches
- Steps: checkout → install gpu requirements → `pytest tests/integration/ -v`

---

## Notebook conventions

Every notebook must start with this cell:

```python
# ── Environment check ──────────────────────────────────────────
import sys
sys.path.insert(0, "/content/llm-inference-optimizer")  # Colab path
sys.path.insert(0, "..")                                  # local path fallback

from src.utils.env import get_env_info
info = get_env_info()
print(f"Device   : {info['device']}")
print(f"GPU      : {info['gpu_name']}")
print(f"CUDA     : {info['cuda_version']}")
print(f"PyTorch  : {info['torch_version']}")
print(f"Colab    : {info['is_colab']}")
```

Every notebook must end with this cell:

```python
# ── Save results to GitHub ─────────────────────────────────────
import subprocess
subprocess.run(["git", "add", "results/", "notebooks/"], check=True)
subprocess.run(["git", "commit", "-m", f"results: {NOTEBOOK_NAME} run"], check=True)
subprocess.run(["git", "push"], check=True)
print("Results pushed to GitHub.")
```

---

## New technology additions (beyond original spec)

These are the additions that differentiate this project. Implement them in this order:

| Technology | Module | Notes |
|---|---|---|
| FP8 quantization | `src/optimization/quantization.py` | H100 only via Transformer Engine |
| Flash Attention v2 | `src/optimization/trt_builder.py` | use `flash-attn` package |
| AWQ quantization | `src/optimization/quantization.py` | via `llmcompressor` (autoawq is deprecated/broken) |
| GPTQ quantization | `src/optimization/quantization.py` | via `gptqmodel` (auto-gptq fails to build) |
| Speculative decoding (Medusa) | `src/serving/inference_engine.py` | via `medusa-llm` |
| Continuous batching | `src/serving/inference_engine.py` | via vLLM backend (inherent to its scheduler) |
| Chunked prefill | `src/serving/inference_engine.py` | vLLM `enable_chunked_prefill=True` |
| NVML power tracking | `src/profiling/profiler.py` | via `pynvml` |
| MFU calculation | `src/benchmarking/benchmark_runner.py` | formula below |

### MFU formula to implement

```python
def calculate_mfu(
    model_params: int,
    tokens_per_second: float,
    gpu_peak_tflops: float,        # A100 = 312, H100 = 989, T4 = 65
    num_layers: int,
    hidden_dim: int,
) -> float:
    """
    MFU = actual_tflops / theoretical_peak_tflops
    actual_tflops = 6 * num_params * tokens_per_second / 1e12
    (factor of 6: 2 for matmul FLOPs × 3 for fwd+bwd, but inference = fwd only so use 2)
    """
    actual_tflops = 2 * model_params * tokens_per_second / 1e12
    return (actual_tflops / gpu_peak_tflops) * 100  # as percentage
```

---

## Current project phase

> **Phase 6 — Benchmarking sweep (`src/benchmarking/`)** ← current
> `run_sweep`: drive `InferenceEngine` across the config grid (batch × seq ×
> dtype × backend), call `profile_generation`, compute MFU via `calculate_mfu`,
> and write `BenchmarkResult` rows to CSV + JSON. `calculate_mfu` +
> `BenchmarkResult` already exist (Phase 1, CPU-safe).

### Phase status

- [x] **Phase 1 — Scaffold + environment setup** *(done — CI green on `main`)*
  - Repo structure, configs, CI, `src/utils/env.py` + `logger.py` fully implemented.
  - Later-phase modules exist as contract-accurate stubs with GPU guards.
  - `calculate_mfu` + `BenchmarkResult` implemented (CPU-safe).
- [x] **Phase 2 — ONNX export pipeline (`src/export/`)** *(done — validated on Colab T4)*
  - `export_to_onnx` via HF Optimum (`text-generation-with-past`, KV-cache).
  - `src/utils/config.py` loaders; tiny-model parity verified (max abs diff ~4e-8).
  - Ecosystem notes: optimum 2.x → install `optimum-onnx`; onnxruntime-gpu CUDA
    major must match torch's, else verification falls back to CPU ORT provider.
- [x] **Phase 3 — TensorRT engine builder + quantization (`src/optimization/`)** *(done — fully validated on Colab: TRT 11.1 engine + INT8 + AWQ + GPTQ)*
  - `build_trt_engine`: **fp16/fp32 implemented + validated on Colab (TensorRT
    11.1)** — parses ONNX, builds a single optimization profile over
    batch/sequence/past-length (static KV dims read from the graph), serializes +
    deserialize-verifies the `.engine`. `int8` (entropy calibrator) and `fp8`
    (ONNX Q/DQ + H100) raise `NotImplementedError` *before* the GPU guard.
  - `quantize_model`: all validated on Colab with TinyLlama-1.1B —
    **awq** via `llmcompressor` (~762 MB) + **gptq** via `gptqmodel` (~767 MB) in
    the separate `gpu-quant.txt` env; **int8** via bitsandbytes (~1.2 GB) in the
    `gpu.txt` env; **fp8** gated to Hopper (SM 9.0+).
  - mypy overrides added: `llmcompressor.*`, `gptqmodel.*`, `bitsandbytes.*`,
    `datasets.*`, `transformer_engine.*`.
  - **Colab/TensorRT 11 ecosystem notes (hard-won):**
    - Install **`tensorrt-cu12`**, never bare `tensorrt` (resolves to a cu13 build
      → `libcudart.so.13` mismatch on Colab's CUDA 12). Same trap as
      `onnxruntime-gpu` — use **CPU `onnxruntime`** for export (parity check is
      correctness-only).
    - **TensorRT 11 removed** `NetworkDefinitionCreationFlag.EXPLICIT_BATCH`,
      `builder.platform_has_fast_fp16`, and the precision `BuilderFlag`s
      (`FP16`/`INT8`/`FP8`). Precision now comes from the ONNX tensor types via a
      **`STRONGLY_TYPED`** network. `trt_builder.py` detects the regime
      (`BuilderFlag.FP16` absent) and adapts; on TRT 11 the engine follows the
      export dtype, so build at the same dtype the ONNX was exported with.
    - **autoawq and auto-gptq are dead** (autoawq import-breaks on current
      transformers; auto-gptq won't build). Replaced by `llmcompressor`/`gptqmodel`,
      which require **transformers 5.x** — hence the two-env split (`gpu.txt` vs
      `gpu-quant.txt`); AWQ/GPTQ run in their own Colab session.
    - **Quantization needs a real model, not `tiny-random`.** AWQ/GPTQ do
      group-wise 4-bit quant (`group_size=128`), which requires weight dims
      divisible by 128; the tiny-random model (hidden=16) fails with
      `unflatten ... don't multiply up`, and its toy embedding vs real tokenizer
      also overflows GPTQ calibration. Validate AWQ/GPTQ on a small *real* Llama
      (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, ungated). tiny-random is fine for
      export/TRT (graph structure) but not for quantization.
  - `notebooks/02_optimize.ipynb` wired: config → ONNX → engine → quantize.
- [x] **Phase 4 — Serving runtime + vLLM (`src/serving/`)** *(done — eager/onnx/vllm validated; eager on CPU+CI, onnx/vllm on Colab)*
  - `InferenceEngine`: **eager** (HF `.generate`, CPU+GPU) — real CPU unit tests
    (generation, greedy determinism, batch, warmup), not just guard rails.
  - **onnx** via `ORTModelForCausalLM.generate` (shares the HF generate path,
    consumes the Phase 2 export dir) — validated on Colab.
  - **vllm** via `vllm.LLM` + `SamplingParams`, continuous batching + chunked
    prefill — validated on Colab (TinyLlama-1.1B, ~133 tok/s).
  - **trt** serving backend deferred — raw engine execution needs a hand-rolled
    autoregressive decode loop over the KV-cache bindings (separate follow-up).
    Medusa speculative decoding also deferred.
  - vLLM isolated in `requirements/gpu-serve.txt` (own torch/transformers pin).
  - **Colab vLLM notes:** install via `uv pip install vllm --torch-backend=auto`
    (plain `pip` pulls a cu13 wheel that won't load). `_prepare_vllm_runtime()`
    self-heals the cu13-on-cu12 runtime: preloads the bundled `nvidia/cu*` libs,
    adds them to `LD_LIBRARY_PATH` for spawned workers, and forces
    `VLLM_WORKER_MULTIPROC_METHOD=spawn` (CUDA can't init in a forked child).
- [x] **Phase 5 — Profiling wrapper (`src/profiling/`)** *(done — validated on Colab T4)*
  - `profile_generation` (black-box over any `InferenceEngine` backend): TTFT =
    single-token-run latency, TPOT = marginal per-token cost, throughput, peak
    CUDA mem, mean/peak NVML power. CUDA-synced timing; CPU-runnable with the
    eager backend (real CPU unit tests, GPU metrics degrade to sentinels).
  - `_PowerSampler`: background NVML thread (mean + peak) over the timed run.
  - Colab T4 validation (TinyLlama-1.1B): TTFT 90.8ms, TPOT 26.9ms, 69 tok/s,
    mem 2.21GB (matches fp16 weights), power 45.2W mean / 53.2W peak.
  - torch.profiler op-level trace capture (`save_traces`) deferred to a follow-up.
- [~] **Phase 6 — Benchmarking sweep (`src/benchmarking/`)** ← current *(core validated on CPU; GPU sweep Colab-pending)*
  - `run_sweep(config_path, engine, output_dir)`: sweeps the batch_size × seq_len
    grid for one engine (the engine fixes backend/dtype/model), profiles each via
    `profile_generation`, computes MFU (`load_model_config` or model-config
    introspection × `get_gpu_name()` → peak TFLOPs), writes
    `benchmark_<backend>_<dtype>.{csv,json}`. Cross-backend = call once per engine.
  - CPU-runnable with eager → real unit tests (grid, schema, CSV/JSON written;
    MFU/power/mem sentinels off-GPU). `default_sweep.yaml` reconciled to
    implemented backends (`eager`/`onnx`/`vllm`, `fp16`) + `max_new_tokens`.
  - Pending: GPU sweep validation + `03_benchmark.ipynb` backend loop.
- [ ] Phase 7: Nsight integration (requires bare-metal GPU — Lambda Labs / RunPod)

### Cross-cutting conventions established in Phase 1

- **mypy checks first-party code only.** `pyproject.toml` sets
  `follow_imports = "skip"` for heavy third-party libs (`torch`, `transformers`,
  …). This prevents mypy from analyzing/ crashing inside their stubs. Add any new
  GPU-only dependency to that override list.
- **GPU guards fail loudly** with the standard `"GPU required ..."` message — see
  the stubs in `src/export`, `src/optimization`, `src/profiling`.

---

## Common commands reference

```bash
# Local dev
make setup-local          # first time setup on Mac
make lint                 # check code quality
make test                 # run CPU-safe unit tests
make format               # auto-fix formatting

# Git
git add . && git commit -m "feat: <description>"
git push origin main

# Colab (run in notebook cell)
!git clone https://github.com/YOUR_USERNAME/llm-inference-optimizer.git
!pip install -r llm-inference-optimizer/requirements/gpu.txt
```

---

## Do not do these things

- Do not install `tensorrt`, `pycuda`, or `flash-attn` in `requirements/base.txt` — Mac cannot install them
- Do not call `torch.cuda.*` outside a `is_cuda_available()` guard
- Do not hardcode model paths — always load from `configs/model_configs/`
- Do not commit `results/`, `*.onnx`, `*.engine`, or `*.pt` files — they are gitignored
- Do not use `print()` in `src/` modules — use the logger
- Do not write notebooks that only work on Colab — the first two cells must work locally too
