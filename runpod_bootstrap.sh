#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[1/4] Creating virtualenv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"

echo "[2/4] Upgrading pip"
python -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing PyTorch CUDA 12.8 build"
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1

echo "[4/4] Installing project dependencies"
python -m pip install \
  "kernels>=0.11.7" \
  "matplotlib>=3.10.8" \
  "numpy>=2.2.6" \
  "pandas>=2.3.3" \
  "pyarrow>=21.0.0" \
  "requests>=2.32.0" \
  "rustbpe>=0.1.0" \
  "tiktoken>=0.11.0"

echo
echo "Bootstrap complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"
echo "Then run for example: bash runpod_vocab_fixed_d_sweep.sh 300"
