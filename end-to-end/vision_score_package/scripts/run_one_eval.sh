#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${PKG_DIR}/config.yaml}"
MODEL="${2:?Usage: bash scripts/run_one_eval.sh [config.yaml] model_name}"
PYTHON_BIN="/home/T7/public/miniconda3/envs/finetune1/bin/python"

cd "${PKG_DIR}"
"${PYTHON_BIN}" tools/run_eval.py --config "${CONFIG}" --models "${MODEL}"
