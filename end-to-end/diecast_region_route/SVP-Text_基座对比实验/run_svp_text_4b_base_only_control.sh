#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="${BASE_DIR:-/home/T7/wkq/end-to-end/diecast_region_route/SVP-Text_基座对比实验}"
PYTHON_BIN="${PYTHON_BIN:-/home/T7/public/miniconda3/envs/finetune1/bin/python}"
MAKE_CONFIG="${BASE_DIR}/make_4b_base_only_control_config.py"
SCRIPT="${BASE_DIR}/e2e_svp_text_300q_multisize_graphrag.py"
CONFIG="${OUTPUT_CONFIG:-${BASE_DIR}/configs/e2e_token_text_config_4b_300q_base_only_control.json}"
LOG_DIR="${LOG_DIR:-/home/T7/wkq/end-to-end/compare_outputs/logs_svp_text_4b_base_only_control}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

mkdir -p "${LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] python not executable: ${PYTHON_BIN}"
  exit 1
fi
if [[ ! -f "${MAKE_CONFIG}" ]]; then
  echo "[ERROR] config maker not found: ${MAKE_CONFIG}"
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "[ERROR] script not found: ${SCRIPT}"
  exit 1
fi

"${PYTHON_BIN}" "${MAKE_CONFIG}"

LOG_FILE="${LOG_DIR}/4b_base_only_control_$(date +%Y%m%d_%H%M%S).log"
echo "[run] config=${CONFIG}"
echo "[run] log=${LOG_FILE}"

E2E_CONFIG_JSON="${CONFIG}" stdbuf -oL -eL "${PYTHON_BIN}" "${SCRIPT}" > "${LOG_FILE}" 2>&1
status=$?
if [[ "${status}" -eq 0 ]]; then
  echo "[done] 4B base-only control finished. log=${LOG_FILE}"
else
  echo "[failed] 4B base-only control failed. exit_code=${status}, log=${LOG_FILE}"
fi
exit "${status}"
