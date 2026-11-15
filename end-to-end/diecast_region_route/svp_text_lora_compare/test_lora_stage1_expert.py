from __future__ import annotations

from common_svp_text_experiment import run_experiment


ADAPTER_PATH = "/home/T7/wkq/end-to-end/runs/qwen3_4b_stage1_expert_clean_best"


if __name__ == "__main__":
    run_experiment(
        experiment_name="lora_stage1_expert",
        adapter_path=ADAPTER_PATH,
    )

