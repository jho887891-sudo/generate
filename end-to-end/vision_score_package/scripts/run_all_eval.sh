#!/usr/bin/env bash
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${PKG_DIR}/config.yaml}"
PYTHON_BIN="/home/T7/public/miniconda3/envs/finetune1/bin/python"
OUT_DIR="/home/T7/wkq/end-to-end/vision_score_outputs"

mkdir -p "${OUT_DIR}"
cd "${PKG_DIR}"

echo "[eval] package: ${PKG_DIR}"
echo "[eval] config : ${CONFIG}"
echo "[eval] output : ${OUT_DIR}"

"${PYTHON_BIN}" tools/check_paths.py --config "${CONFIG}"
"${PYTHON_BIN}" tools/run_eval.py --config "${CONFIG}"

echo "[eval] done"
