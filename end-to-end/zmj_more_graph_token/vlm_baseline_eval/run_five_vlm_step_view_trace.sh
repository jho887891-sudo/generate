#!/usr/bin/env bash
set -euo pipefail

cd /home/T7/wkq/end-to-end/vlm_baseline_eval

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

PYTHON_BIN=${PYTHON_BIN:-/home/T7/public/miniconda3/envs/finetune1/bin/python}
CONFIG=${CONFIG:-/home/T7/wkq/end-to-end/vlm_baseline_eval/config_five_vlm_step_view_sample38069.json}
MAX_VIEWS=${MAX_VIEWS:-9}
MODELS=${MODELS:-qwen25_vl_7b,qwen3_vl_8b,internvl35_8b_hf,minicpm_v45,llava_onevision}

"$PYTHON_BIN" /home/T7/wkq/end-to-end/vlm_baseline_eval/run_five_vlm_step_view_trace.py \
  --config "$CONFIG" \
  --max-views "$MAX_VIEWS" \
  --models "$MODELS"
