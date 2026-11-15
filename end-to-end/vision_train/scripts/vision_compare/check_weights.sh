#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml}"
PYTHON_BIN="/home/T7/public/miniconda3/envs/finetune1/bin/python"
PROJECT_ROOT="/home/T7/wkq/end-to-end/vision_train"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" tools/vision_compare/03_check_outputs.py --config "${CONFIG}"
