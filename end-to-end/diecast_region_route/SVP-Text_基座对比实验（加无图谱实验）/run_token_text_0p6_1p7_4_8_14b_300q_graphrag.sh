#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="${BASE_DIR:-/home/T7/wkq/end-to-end/diecast_region_route/SVP-Text_基座对比实验}"
PYTHON_BIN="${PYTHON_BIN:-/home/T7/public/miniconda3/envs/finetune1/bin/python}"
SCRIPT="${BASE_DIR}/e2e_svp_text_300q_multisize_graphrag.py"
MAKE_CONFIGS="${BASE_DIR}/make_5size_300q_configs.py"
CONFIG_DIR="${BASE_DIR}/configs"
LOG_DIR="${LOG_DIR:-/home/T7/wkq/end-to-end/compare_outputs/logs_svp_text_5size_300q}"
SUMMARY_JSON="${LOG_DIR}/run_summary_$(date +%Y%m%d_%H%M%S).jsonl"

# 默认顺序：小模型先跑，确认流程没有问题，再跑大模型。
SIZES_DEFAULT=("0.6b" "1.7b" "4b" "8b" "14b")
if [[ "$#" -gt 0 ]]; then
  SIZES=("$@")
else
  SIZES=("${SIZES_DEFAULT[@]}")
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${LOG_DIR}"

printf '%s\n' "============================================================"
printf '%s\n' "SVP-Text 5-size test: 0.6B / 1.7B / 4B / 8B / 14B"
printf '%s\n' "No multimodal heads or prefix tokens will be loaded."
printf '%s\n' "Each size uses the same fixed 300 questions."
printf '%s\n' "BASE_DIR=${BASE_DIR}"
printf '%s\n' "SCRIPT=${SCRIPT}"
printf '%s\n' "CONFIG_DIR=${CONFIG_DIR}"
printf '%s\n' "LOG_DIR=${LOG_DIR}"
printf '%s\n' "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
printf '%s\n' "SIZES=${SIZES[*]}"
printf '%s\n' "============================================================"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] python not executable: ${PYTHON_BIN}"
  exit 1
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "[ERROR] script not found: ${SCRIPT}"
  exit 1
fi
if [[ ! -f "${MAKE_CONFIGS}" ]]; then
  echo "[ERROR] config maker not found: ${MAKE_CONFIGS}"
  exit 1
fi

"${PYTHON_BIN}" "${MAKE_CONFIGS}"

check_paths() {
  local cfg="$1"
  "${PYTHON_BIN}" - "$cfg" <<'PY'
import json, os, sys
cfg_path = sys.argv[1]
with open(cfg_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
keys = [
    'llm_base_path', 'llm_stage2_adapter_path', 'visual_model_path', 'calibration_json',
    'region_types_json', 'standard_json', 'template_folder', 'mask_folder', 'image_dir'
]
missing = []
for k in keys:
    v = cfg.get(k, '')
    if v and not os.path.exists(v):
        missing.append((k, v))
print(f"[check] model_size={cfg.get('model_size')} result_output_dir={cfg.get('result_output_dir')}")
for k, v in missing:
    print(f"[check][WARN] missing {k}: {v}")
if missing:
    print("[check][WARN] 如果这些路径名称和你服务器不一致，先改 model_jobs_5size_300q.json 后重新运行。")
PY
}

for size in "${SIZES[@]}"; do
  case "${size}" in
    0p6b) size="0.6b" ;;
    1p7b) size="1.7b" ;;
  esac
  CONFIG="${CONFIG_DIR}/e2e_token_text_config_${size}_300q_graphrag.json"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] config not found for size=${size}: ${CONFIG}" | tee -a "${SUMMARY_JSON}"
    continue
  fi

  RESULT_DIR=$("${PYTHON_BIN}" - "$CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print(cfg.get('result_output_dir', ''))
PY
)
  LOG_FILE="${LOG_DIR}/${size}_300q_graphrag_$(date +%Y%m%d_%H%M%S).log"

  if [[ "${SKIP_DONE:-1}" == "1" && -f "${RESULT_DIR}/all_modes_results.json" ]]; then
    echo "[skip] ${size} already done: ${RESULT_DIR}/all_modes_results.json" | tee -a "${SUMMARY_JSON}"
    continue
  fi

  echo "============================================================"
  echo "[run] size=${size}"
  echo "[run] config=${CONFIG}"
  echo "[run] result=${RESULT_DIR}"
  echo "[run] log=${LOG_FILE}"
  echo "============================================================"
  check_paths "${CONFIG}"

  set +e
  E2E_CONFIG_JSON="${CONFIG}" stdbuf -oL -eL "${PYTHON_BIN}" "${SCRIPT}" 2>&1 | tee "${LOG_FILE}"
  status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -eq 0 ]]; then
    echo "{\"size\":\"${size}\",\"status\":\"ok\",\"result\":\"${RESULT_DIR}/all_modes_results.json\",\"log\":\"${LOG_FILE}\"}" >> "${SUMMARY_JSON}"
    echo "[done] ${size}: ${RESULT_DIR}/all_modes_results.json"
  else
    echo "{\"size\":\"${size}\",\"status\":\"failed\",\"exit_code\":${status},\"result\":\"${RESULT_DIR}/all_modes_results.json\",\"log\":\"${LOG_FILE}\"}" >> "${SUMMARY_JSON}"
    echo "[failed] ${size}, exit_code=${status}, log=${LOG_FILE}"
    if [[ "${STOP_ON_ERROR:-0}" == "1" ]]; then
      exit "${status}"
    fi
  fi

  # 给下一轮释放显存和文件句柄的缓冲时间。
  sleep "${SLEEP_BETWEEN_RUNS:-10}"
done

echo "============================================================"
echo "ALL REQUESTED RUNS FINISHED"
echo "Summary: ${SUMMARY_JSON}"
echo "Logs: ${LOG_DIR}"
echo "============================================================"
