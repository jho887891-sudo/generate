#!/usr/bin/env bash
set -euo pipefail

cd /home/T7/wkq/end-to-end/vlm_baseline_eval

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

PYTHON_BIN="${PYTHON_BIN:-/home/T7/public/miniconda3/envs/finetune1/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-/home/T7/wkq/end-to-end/vlm_baseline_eval/config_qwen25vl7b_9views_sample38069.json}"
MODEL_PATH="${MODEL_PATH:-/home/T7/wkq/end-to-end/vlm_baseline_eval/vlm_baselines/Qwen2.5-VL-7B-Instruct}"
# 一个工件默认九张X光检测图；每张检测图同时给模板图和蒙版图，所以视觉总结阶段共 9*3=27 张图。
MAX_VIEWS="${MAX_VIEWS:-9}"

"${PYTHON_BIN}" /home/T7/wkq/end-to-end/vlm_baseline_eval/run_qwen25vl7b_diecast_9views_call.py   --config "${CONFIG_PATH}"   --model-path "${MODEL_PATH}"   --max-views "${MAX_VIEWS}"
