#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml}"
PYTHON_BIN="/home/T7/public/miniconda3/envs/finetune1/bin/python"
PROJECT_ROOT="/home/T7/wkq/end-to-end/vision_train"
LOG_DIR="/home/T7/wkq/end-to-end/logs/vision_compare"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

echo "[prepare_data] checking environment"
"${PYTHON_BIN}" tools/vision_compare/00_check_env.py --config "${CONFIG}" 2>&1 | tee "${LOG_DIR}/prepare_00_check_env.log"

echo "[prepare_data] cleaning raw dataset"
"${PYTHON_BIN}" tools/vision_compare/01_clean_dataset.py --config "${CONFIG}" 2>&1 | tee "${LOG_DIR}/prepare_01_clean_dataset.log"

echo "[prepare_data] building unified splits and model formats"
"${PYTHON_BIN}" tools/vision_compare/02_build_splits_and_formats.py --config "${CONFIG}" 2>&1 | tee "${LOG_DIR}/prepare_02_build_splits_and_formats.log"

echo "[prepare_data] done"
