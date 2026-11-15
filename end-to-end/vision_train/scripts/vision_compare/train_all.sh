#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-/home/T7/wkq/end-to-end/vision_train/configs/vision_compare.yaml}"
RESUME_FLAG="${2:-}"
PROJECT_ROOT="/home/T7/wkq/end-to-end/vision_train"
LOG_DIR="/home/T7/wkq/end-to-end/logs/vision_compare"
mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

run_step() {
  local name="$1"
  local script="$2"
  echo "[train_all] starting ${name}"
  if [[ "${RESUME_FLAG}" == "--resume" ]]; then
    bash "${script}" "${CONFIG}" --resume > "${LOG_DIR}/${name}.log" 2>&1
  else
    bash "${script}" "${CONFIG}" > "${LOG_DIR}/${name}.log" 2>&1
  fi
  echo "[train_all] finished ${name}"
}

run_step "train_unet" "scripts/vision_compare/train_unet.sh"
run_step "train_deeplabv3plus" "scripts/vision_compare/train_deeplabv3plus.sh"
run_step "train_mask_rcnn" "scripts/vision_compare/train_mask_rcnn.sh"
run_step "train_yolov8n_seg" "scripts/vision_compare/train_yolov8n_seg.sh"
run_step "train_yolov11n_seg" "scripts/vision_compare/train_yolov11n_seg.sh"

bash scripts/vision_compare/check_weights.sh "${CONFIG}"
echo "[train_all] all done"
