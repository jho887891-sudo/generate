#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-/home/T7/public/miniconda3/envs/finetune1/bin/python}"
RESULT_JSON="/home/T7/wkq/end-to-end/compare_outputs/base_vs_main_50q_sample_38071/all_modes_results.json"
SCORER="${SCORER:-/home/T7/wkq/end-to-end/diecast_region_route/test_token_text/standard_e2e_scoring_v2.py}"
OUT_DIR="/home/T7/wkq/end-to-end/compare_outputs/base_vs_main_50q_sample_38071_score_raw"

if [[ ! -f "${SCORER}" ]]; then
  echo "[ERROR] scoring script not found: ${SCORER}"
  echo "请把 standard_e2e_scoring_v2.py 放到该路径，或用 SCORER=/path/to/standard_e2e_scoring_v2.py 覆盖。"
  exit 1
fi
"${PYTHON_BIN}" "${SCORER}" --input "${RESULT_JSON}" --output-dir "${OUT_DIR}"
echo "score output: ${OUT_DIR}"
