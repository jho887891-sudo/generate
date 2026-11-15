from __future__ import annotations

from common_svp_text_experiment import run_experiment


ADAPTER_PATH = "/home/T7/wkq/end-to-end/runs/qwen3_4b_只进行第二次微调"


if __name__ == "__main__":
    run_experiment(
        experiment_name="lora_second_only",
        adapter_path=ADAPTER_PATH,
    )

