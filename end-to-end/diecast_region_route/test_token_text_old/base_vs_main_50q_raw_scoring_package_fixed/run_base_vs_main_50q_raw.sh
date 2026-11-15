#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/T7/public/miniconda3/envs/finetune1/bin/python}"
ROOT_DIR="/home/T7/wkq/end-to-end/diecast_region_route/test_token_text"
SCRIPT="${ROOT_DIR}/base_vs_main_50q_raw_scoring.py"
CONFIG="${ROOT_DIR}/base_vs_main_50q_config.json"
LOG_DIR="/home/T7/wkq/end-to-end/compare_outputs/logs_base_vs_main_50q"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# 可选：如果配置里的 base_llm_path 不对，可以在运行时覆盖：
# BASE_LLM_PATH=/path/to/unfinetuned/Qwen3-4B bash run_base_vs_main_50q_raw.sh

echo "============================================================"
echo "Base vs Main 50Q raw scoring test"
echo "SCRIPT=${SCRIPT}"
echo "CONFIG=${CONFIG}"
echo "BASE_LLM_PATH=${BASE_LLM_PATH:-<use config base_llm_path/candidates>}"
echo "LOG_DIR=${LOG_DIR}"
echo "============================================================"

if [[ ! -f "${SCRIPT}" ]]; then
  echo "[ERROR] script not found: ${SCRIPT}"
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] config not found: ${CONFIG}"
  exit 1
fi

E2E_CONFIG_JSON="${CONFIG}" "${PYTHON_BIN}" "${SCRIPT}" 2>&1 | tee "${LOG_DIR}/base_vs_main_50q_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "DONE"
echo "Result: /home/T7/wkq/end-to-end/compare_outputs/base_vs_main_50q_sample_38071/all_modes_results.json"
echo "============================================================"
