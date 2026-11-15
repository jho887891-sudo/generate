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
  "${PYTHON_BIN}" train/vision_compare/train_yolo_seg.py --config "${CONFIG}" --model-name yolov8n_seg --resume 2>&1 | tee -a "${LOG_DIR}/train_yolov8n_seg.log"
else
  "${PYTHON_BIN}" train/vision_compare/train_yolo_seg.py --config "${CONFIG}" --model-name yolov8n_seg 2>&1 | tee -a "${LOG_DIR}/train_yolov8n_seg.log"
fi
