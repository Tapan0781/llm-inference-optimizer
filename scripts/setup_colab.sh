#!/usr/bin/env bash
# Google Colab (GPU) setup. Run inside a GPU runtime.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
pip install -r requirements/gpu.txt -r requirements/dev.txt
pip install -e .

python -c "from src.utils.env import get_env_info; print(get_env_info())"
echo "Colab setup complete."
