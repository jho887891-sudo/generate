#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml}"
RESUME_FLAG="${2:-}"
PYTHON_BIN="/home/T7/public/miniconda3/envs/finetune1/bin/python"
PROJECT_ROOT="/home/T7/wkq/end-to-end/vision_train"
LOG_DIR="/home/T7/wkq/end-to-end/logs/vision_compare"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"
if [[ "${RESUME_FLAG}" == "--resume" ]]; then
  "${PYTHON_BIN}" train/vision_compare/train_mask_rcnn.py --config "${CONFIG}" --resume 2>&1 | tee -a "${LOG_DIR}/train_mask_rcnn.log"
else
  "${PYTHON_BIN}" train/vision_compare/train_mask_rcnn.py --config "${CONFIG}" 2>&1 | tee -a "${LOG_DIR}/train_mask_rcnn.log"
fi
