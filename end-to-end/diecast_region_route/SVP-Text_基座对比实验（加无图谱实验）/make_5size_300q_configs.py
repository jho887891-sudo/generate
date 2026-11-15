from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_CONFIG = SCRIPT_DIR / "e2e_token_text_config_4b_600q_sample38063_graphrag.json"
QUESTIONS_FILE = SCRIPT_DIR / "questions_300_fixed_from_600q.json"
JOBS_FILE = SCRIPT_DIR / "model_jobs_5size_300q.json"
CONFIG_DIR = SCRIPT_DIR / "configs"

COMPARE_ROOT = "/home/T7/wkq/end-to-end/compare_outputs"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clean_question_list(obj: Any) -> List[str]:
    if isinstance(obj, dict):
        obj = obj.get("questions", [])
    if not isinstance(obj, list):
        raise ValueError("questions 必须是 list 或包含 questions 的 dict")
    out: List[str] = []
    for item in obj:
        if isinstance(item, dict):
            q = str(item.get("question", "")).strip()
        else:
            q = str(item).strip()
        if q:
            out.append(q)
    return out


def main() -> None:
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(f"base config not found: {BASE_CONFIG}")
    if not JOBS_FILE.exists():
        raise FileNotFoundError(f"jobs file not found: {JOBS_FILE}")

    base_cfg: Dict[str, Any] = load_json(BASE_CONFIG)

    if QUESTIONS_FILE.exists():
        questions = clean_question_list(load_json(QUESTIONS_FILE))[:300]
    else:
        questions = clean_question_list(base_cfg.get("questions", []))[:300]
        save_json(QUESTIONS_FILE, {
            "source": BASE_CONFIG.name,
            "selection": "first_300_questions",
            "count": len(questions),
            "questions": questions,
        })
    if len(questions) != 300:
        raise ValueError(f"固定问题数量不是300，当前为 {len(questions)}")

    jobs_obj = load_json(JOBS_FILE)
    jobs = jobs_obj.get("jobs", []) if isinstance(jobs_obj, dict) else []
    if not jobs:
        raise ValueError("model_jobs_5size_300q.json 中没有 jobs")

    workpiece_id = str(base_cfg.get("workpiece_id", base_cfg.get("sample_id", "unknown")))
    made = []
    for job in jobs:
        if not job.get("enabled", True):
            continue
        size_key = str(job["size_key"])
        result_dir = f"{COMPARE_ROOT}/e2e_{size_key}_sample_{workpiece_id}_300q_svp_text_graphrag"

        cfg = copy.deepcopy(base_cfg)
        # 所有尺寸共用同一个固定问题文件，保证300问完全一致。
        cfg.pop("questions", None)
        cfg["questions_file"] = "../questions_300_fixed_from_600q.json"
        cfg["max_questions"] = 300
        cfg["compare_modes"] = ["svp_text"]
        cfg["use_graph_rag"] = bool(cfg.get("use_graph_rag", True))
        cfg["save_intermediate_visual_files"] = False
        cfg["stream_to_stdout"] = False
        cfg["resume"] = True
        cfg["generation_max_time_sec"] = int(cfg.get("generation_max_time_sec", 240))
        cfg["max_new_tokens"] = int(cfg.get("max_new_tokens", 512))

        # 每个尺寸自己的底座和 Stage2 adapter。
        # SVP-Text 入口不再读取 mm_weight_path / mm_output_dir；保留字段仅用于旧脚本兼容。
        for key in [
            "model_size", "mm_weight_path", "mm_output_dir", "llm_base_path",
            "llm_stage2_adapter_path", "min_free_gpu_gb",
        ]:
            if key in job:
                cfg[key] = job[key]
        cfg["size_key"] = size_key
        cfg["result_output_dir"] = result_dir
        cfg["visual_output_dir"] = f"{result_dir}/visual"

        out_path = CONFIG_DIR / f"e2e_token_text_config_{size_key}_300q_graphrag.json"
        save_json(out_path, cfg)
        made.append(str(out_path))
        alias_key = size_key.replace(".", "p")
        if alias_key != size_key:
            alias_path = CONFIG_DIR / f"e2e_token_text_config_{alias_key}_300q_graphrag.json"
            save_json(alias_path, cfg)
            made.append(str(alias_path))

    index = {
        "question_file": str(QUESTIONS_FILE),
        "question_count": 300,
        "config_count": len(made),
        "configs": made,
    }
    save_json(CONFIG_DIR / "generated_config_index.json", index)
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
