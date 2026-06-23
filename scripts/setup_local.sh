#!/usr/bin/env bash
# Local (Mac / CPU) setup. No CUDA.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
pip install -r requirements/base.txt -r requirements/dev.txt
pip install -e .
pre-commit install || true

echo "Local setup complete. Run 'make test' to verify."
