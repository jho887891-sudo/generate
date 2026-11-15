#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/T7/public/miniconda3/envs/finetune1/bin/python"
SCRIPT="/home/T7/wkq/end-to-end/diecast_region_route/test_token_text/e2e_token_text_only_600q_4b_graphrag.py"
CONFIG="/home/T7/wkq/end-to-end/diecast_region_route/test_token_text/e2e_token_text_config_4b_600q_sample38063_graphrag.json"

LOG_DIR="/home/T7/wkq/end-to-end/compare_outputs/logs_token_text_600q_graphrag"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "============================================================"
echo "Token_text only stability test: 4B, 600 questions, GraphRAG enabled"
echo "SCRIPT=${SCRIPT}"
echo "CONFIG=${CONFIG}"
echo "LOG_DIR=${LOG_DIR}"
echo "============================================================"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] config not found: ${CONFIG}"
  exit 1
fi

E2E_CONFIG_JSON="${CONFIG}" "${PYTHON_BIN}" "${SCRIPT}" 2>&1 | tee "${LOG_DIR}/4b_600q_graphrag_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "DONE"
echo "Result: /home/T7/wkq/end-to-end/compare_outputs/e2e_4b_sample_38063_600q_token_text_graphrag/all_modes_results.json"
echo "============================================================"
