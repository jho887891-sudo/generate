#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export SVP_TEXT_END_TO_END_ROOT="${SVP_TEXT_END_TO_END_ROOT:-/home/T7/wkq/end-to-end}"
export DIECAST_REGION_ROUTE_ROOT="${DIECAST_REGION_ROUTE_ROOT:-$SVP_TEXT_END_TO_END_ROOT/diecast_region_route}"
export SVP_TEXT_CONFIG_JSON="${SVP_TEXT_CONFIG_JSON:-$DIECAST_REGION_ROUTE_ROOT/e2e_compare_graph_config_hardcoded.json}"
export SVP_TEXT_RUNTIME_DIR="${SVP_TEXT_RUNTIME_DIR:-$SVP_TEXT_END_TO_END_ROOT/compare_outputs/svp_text_runtime}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

PYTHON_BIN="${PYTHON_BIN:-/home/T7/public/miniconda3/envs/finetune1/bin/python}"

"$PYTHON_BIN" -m uvicorn svp_text_app.main:app \
  --host "${SVP_TEXT_HOST:-0.0.0.0}" \
  --port "${SVP_TEXT_PORT:-8088}" \
  --workers 1

