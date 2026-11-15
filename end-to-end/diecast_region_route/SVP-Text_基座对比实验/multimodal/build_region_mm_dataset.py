from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


RUNS_DIR = Path("/home/T7/zyx/finetune/diecast_e2e_project_web/runs")
OUTPUT_JSONL = Path("/home/T7/zyx/finetune/diecast_region_route/data/region_mm_train.jsonl")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)

    if "温度(℃)" in out and "环境温度" not in out:
        out["环境温度"] = out["温度(℃)"]

    if "湿度(%)" in out and "环境湿度" not in out:
        out["环境湿度"] = out["湿度(%)"]

    if "探伤图编号" in out and "样本编号" not in out:
        out["样本编号"] = out["探伤图编号"]

    return out


def collect_images_from_dir(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ])


def build_image_index(workpiece_dir: Path) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}

    candidate_dirs = [
        workpiece_dir / "input",
        workpiece_dir / "package",
        workpiece_dir / "per_image",
    ]

    for d in candidate_dirs:
        for img_path in collect_images_from_dir(d):
            image_index.setdefault(img_path.name, img_path)
            image_index.setdefault(img_path.stem, img_path)

    return image_index


def match_view_image(view_info: Dict[str, Any], image_index: Dict[str, Path]) -> Optional[Path]:
    candidates = [
        str(view_info.get("image_name", "")).strip(),
        str(view_info.get("image_stem", "")).strip(),
        str(view_info.get("display_name", "")).strip(),
    ]

    for key in candidates:
        if key and key in image_index:
            return image_index[key]

    return None


def build_view_items(workpiece_dir: Path, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_index = build_image_index(workpiece_dir)
    view_items: List[Dict[str, Any]] = []

    images_meta = summary.get("images", [])
    for item in images_meta:
        matched = match_view_image(item, image_index)

        view_items.append({
            "view_index": item.get("view_index"),
            "image_name": item.get("image_name", ""),
            "display_name": item.get("display_name", ""),
            "image_stem": item.get("image_stem", ""),
            "image_path": str(matched) if matched else "",
            "total_defect_count": item.get("total_defect_count", 0),
            "total_area_mm2": item.get("total_area_mm2", 0),
            "max_eq_diameter_mm": item.get("max_eq_diameter_mm", 0),
            "region_count": item.get("region_count", {}),
            "visual_ready": item.get("visual_ready", False),
        })

    return view_items


def build_image_paths(view_items: List[Dict[str, Any]]) -> List[str]:
    image_paths: List[str] = []
    for item in view_items:
        p = item.get("image_path", "")
        if p:
            image_paths.append(p)
    return image_paths


def build_one_sample(workpiece_dir: Path) -> Dict[str, Any]:
    aggregate_dir = workpiece_dir / "aggregate"
    if not aggregate_dir.exists():
        raise FileNotFoundError(f"缺少 aggregate 目录: {aggregate_dir}")

    params_path = aggregate_dir / "params.json"
    summary_path = aggregate_dir / "workpiece_summary.json"

    if not params_path.exists():
        raise FileNotFoundError(f"缺少 params.json: {params_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"缺少 workpiece_summary.json: {summary_path}")

    params = normalize_params(load_json(params_path))
    summary = load_json(summary_path)

    question = read_text_if_exists(aggregate_dir / "question.txt").strip()
    if not question:
        question = "请基于当前样本进行缺陷溯源分析，并给出建议方向。"

    answer = read_text_if_exists(aggregate_dir / "llm_report.txt").strip()
    info_text = read_text_if_exists(aggregate_dir / "info_text.txt").strip()
    graph_context = read_text_if_exists(aggregate_dir / "graph_context.txt").strip()

    usage = {}
    usage_path = aggregate_dir / "usage.json"
    if usage_path.exists():
        usage = load_json(usage_path)

    graph_debug = {}
    graph_debug_path = aggregate_dir / "graph_debug.json"
    if graph_debug_path.exists():
        graph_debug = load_json(graph_debug_path)

    view_items = build_view_items(workpiece_dir, summary)
    image_paths = build_image_paths(view_items)

    return {
        "workpiece_id": workpiece_dir.name,
        "sample_no": params.get("样本编号", params.get("探伤图编号", "")),
        "image_paths": image_paths,
        "view_items": view_items,
        "summary": summary,
        "params": params,
        "question": question,
        "answer": answer,
        "info_text": info_text,
        "graph_context": graph_context,
        "usage": usage,
        "graph_debug": graph_debug,
    }


def find_workpiece_dirs(runs_dir: Path) -> List[Path]:
    workpiece_dirs: List[Path] = []
    for aggregate_dir in runs_dir.rglob("aggregate"):
        if aggregate_dir.is_dir():
            workpiece_dirs.append(aggregate_dir.parent)
    return sorted(workpiece_dirs)


def main() -> None:
    runs_dir = RUNS_DIR
    out_path = OUTPUT_JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        raise FileNotFoundError(f"RUNS_DIR 不存在: {runs_dir}")

    print("=" * 80)
    print("开始构造区域感知多模态训练集")
    print(f"RUNS_DIR    : {runs_dir}")
    print(f"OUTPUT_JSONL: {out_path}")
    print("=" * 80)

    workpiece_dirs = find_workpiece_dirs(runs_dir)
    print(f"递归找到 {len(workpiece_dirs)} 个候选工件目录")

    samples: List[Dict[str, Any]] = []

    for workpiece_dir in workpiece_dirs:
        try:
            sample = build_one_sample(workpiece_dir)

            if not sample["answer"]:
                print(f"[WARN] 跳过 {workpiece_dir.name}: 缺少 llm_report.txt 或答案为空")
                continue

            if len(sample["image_paths"]) == 0:
                print(f"[WARN] 跳过 {workpiece_dir.name}: 未找到任何视角图像")
                continue

            samples.append(sample)

        except Exception as exc:
            print(f"[WARN] 跳过 {workpiece_dir}: {exc}")

    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("=" * 80)
    print(f"已写出 {len(samples)} 条样本到 {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()