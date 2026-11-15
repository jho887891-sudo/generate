#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="/home/T7/wkq/end-to-end/compare_outputs/logs_base_vs_main_50q"
mkdir -p "${LOG_DIR}"
nohup bash run_base_vs_main_50q_raw.sh > "${LOG_DIR}/overnight_base_vs_main_50q_raw.log" 2>&1 &
echo "started: $!"
echo "tail -f ${LOG_DIR}/overnight_base_vs_main_50q_raw.log"
