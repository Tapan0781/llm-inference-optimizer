#!/usr/bin/env bash
# Run the default benchmark sweep. GPU required (Phase 6).
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="${1:-configs/benchmark_configs/default_sweep.yaml}"

python -m src.benchmarking.benchmark_runner --config "$CONFIG"
