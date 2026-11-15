from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CONFIG = SCRIPT_DIR / "configs" / "e2e_token_text_config_4b_300q_graphrag.json"
DEFAULT_OUTPUT_CONFIG = SCRIPT_DIR / "configs" / "e2e_token_text_config_4b_300q_base_only_control.json"
COMPARE_ROOT = "/home/T7/wkq/end-to-end/compare_outputs"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    source_config = Path(os.environ.get("SOURCE_CONFIG", str(DEFAULT_SOURCE_CONFIG)))
    output_config = Path(os.environ.get("OUTPUT_CONFIG", str(DEFAULT_OUTPUT_CONFIG)))
    if not source_config.exists():
        raise FileNotFoundError(f"source config not found: {source_config}")

    cfg = load_json(source_config)
    workpiece_id = str(cfg.get("workpiece_id", cfg.get("sample_id", "unknown")))
    graph_suffix = "graphrag" if bool(cfg.get("use_graph_rag", False)) else "no_graph"

    cfg["llm_base_path"] = "/home/T7/wkq/end-to-end/models/Qwen3-4B-stage1-book-merged-4bit"
    cfg["use_lora_adapter"] = False
    cfg["llm_stage2_adapter_path"] = ""
    cfg["model_size"] = "4B-base-only"
    cfg["size_key"] = "4b_base_only"
    cfg["compare_modes"] = ["svp_text"]
    cfg["resume"] = False
    cfg["result_output_dir"] = (
        f"{COMPARE_ROOT}/e2e_4b_base_only_sample_{workpiece_id}_300q_svp_text_{graph_suffix}"
    )
    cfg["visual_output_dir"] = cfg["result_output_dir"] + "/visual"

    save_json(output_config, cfg)
    print(json.dumps({
        "source_config": str(source_config),
        "output_config": str(output_config),
        "llm_base_path": cfg["llm_base_path"],
        "use_lora_adapter": cfg["use_lora_adapter"],
        "use_graph_rag": cfg.get("use_graph_rag", False),
        "result_output_dir": cfg["result_output_dir"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
