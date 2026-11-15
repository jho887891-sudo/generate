#!/usr/bin/env bash
set -uo pipefail

BASE_DIR="${BASE_DIR:-/home/T7/wkq/end-to-end/diecast_region_route/SVP-Text_基座对比实验}"
PYTHON_BIN="${PYTHON_BIN:-/home/T7/public/miniconda3/envs/finetune1/bin/python}"
SCRIPT="${BASE_DIR}/e2e_svp_text_300q_multisize_graphrag.py"
MAKE_CONFIGS="${BASE_DIR}/make_5size_300q_configs.py"
CONFIG_DIR="${BASE_DIR}/configs"
LOG_DIR="${LOG_DIR:-/home/T7/wkq/end-to-end/compare_outputs/logs_svp_text_5size_300q}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

mkdir -p "${LOG_DIR}"

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

run_one() {
  local size="$1"
  local config="${CONFIG_DIR}/e2e_token_text_config_${size}_300q_graphrag.json"
  local log_file="${LOG_DIR}/${size}_svp_text_$(date +%Y%m%d_%H%M%S).log"

  if [[ ! -f "${config}" ]]; then
    echo "[ERROR] missing config for ${size}: ${config}"
    return 1
  fi

  local result_dir
  result_dir=$("${PYTHON_BIN}" - "$config" <<'PY'
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = json.load(f)
print(cfg.get('result_output_dir', ''))
PY
)

  if [[ "${SKIP_DONE:-1}" == "1" && -f "${result_dir}/all_modes_results.json" ]]; then
    echo "[skip] ${size} already done: ${result_dir}/all_modes_results.json"
    return 0
  fi

  echo "============================================================"
  echo "[run] size=${size}"
  echo "[run] config=${config}"
  echo "[run] result=${result_dir}"
  echo "[run] log=${log_file}"
  echo "============================================================"

  E2E_CONFIG_JSON="${config}" stdbuf -oL -eL "${PYTHON_BIN}" "${SCRIPT}" > "${log_file}" 2>&1
  local status=$?
  if [[ "${status}" -eq 0 ]]; then
    echo "[done] ${size}: ${result_dir}/all_modes_results.json"
  else
    echo "[failed] ${size}, exit_code=${status}, log=${log_file}"
  fi
  return "${status}"
}

run_parallel_group() {
  local group_name="$1"
  shift
  local sizes=("$@")
  local pids=()
  local labels=()

  echo "============================================================"
  echo "[group] ${group_name}: ${sizes[*]}"
  echo "============================================================"

  for size in "${sizes[@]}"; do
    run_one "${size}" &
    pids+=("$!")
    labels+=("${size}")
    sleep "${LAUNCH_STAGGER_SEC:-20}"
  done

  local failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      echo "[group][ok] ${labels[$i]}"
    else
      echo "[group][failed] ${labels[$i]}"
      failed=1
    fi
  done
  return "${failed}"
}

run_sequential_group() {
  local group_name="$1"
  shift
  local sizes=("$@")
  local failed=0

  echo "============================================================"
  echo "[group] ${group_name}: ${sizes[*]}"
  echo "============================================================"

  for size in "${sizes[@]}"; do
    run_one "${size}"
    local status=$?
    if [[ "${status}" -ne 0 ]]; then
      failed=1
      if [[ "${STOP_ON_ERROR:-0}" == "1" ]]; then
        return "${status}"
      fi
    fi
    sleep "${SLEEP_BETWEEN_RUNS:-30}"
  done
  return "${failed}"
}

echo "SVP-Text staged run"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Batch 1 parallel: 0.6b 1.7b 4b"
echo "Batch 2 default sequential: 8b 14b"
echo "Set RUN_LARGE_PARALLEL=1 to run 8b and 14b in parallel."

run_parallel_group "small_models_parallel" "0.6b" "1.7b" "4b"
small_status=$?

if [[ "${small_status}" -ne 0 && "${STOP_ON_ERROR:-0}" == "1" ]]; then
  echo "[stop] small model group failed."
  exit "${small_status}"
fi

if [[ "${RUN_LARGE_PARALLEL:-0}" == "1" ]]; then
  run_parallel_group "large_models_parallel" "8b" "14b"
  large_status=$?
else
  run_sequential_group "large_models_sequential" "8b" "14b"
  large_status=$?
fi

if [[ "${small_status}" -ne 0 || "${large_status}" -ne 0 ]]; then
  echo "[finished] some runs failed. Check logs: ${LOG_DIR}"
  exit 1
fi

echo "[finished] all runs completed. Logs: ${LOG_DIR}"
