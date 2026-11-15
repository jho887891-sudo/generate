from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _short_dict(d: Dict[str, Any], max_items: int = 6) -> str:
    if not isinstance(d, dict) or not d:
        return ""
    items = list(d.items())[:max_items]
    return "；".join([f"{k}:{v}" for k, v in items])


def _top_view_lines(view_items: Sequence[Dict[str, Any]], max_items: int = 2) -> str:
    if not view_items:
        return ""
    lines = []
    sorted_items = sorted(
        view_items,
        key=lambda x: (
            -(float(x.get("total_defect_count", 0) or 0)),
            -(float(x.get("total_area_mm2", 0) or 0)),
        ),
    )
    for item in sorted_items[:max_items]:
        lines.append(
            f"视角{item.get('view_index', '?')}: "
            f"数量={item.get('total_defect_count', 0)}, "
            f"面积={item.get('total_area_mm2', 0)}, "
            f"最大等效直径={item.get('max_eq_diameter_mm', 0)}, "
            f"区域统计={_short_dict(item.get('region_count', {}), max_items=4)}"
        )
    return "\n".join(lines)


def _normalize_graph_evidence(graph_evidence: Any) -> List[str]:
    if not graph_evidence:
        return []
    if isinstance(graph_evidence, dict):
        ctx = str(graph_evidence.get("graph_context", "")).strip()
        if ctx:
            return [line.strip("- ").strip() for line in ctx.splitlines() if line.strip()][:4]
        return [f"{k}:{v}" for k, v in graph_evidence.items() if k != "graph_debug"][:4]
    if isinstance(graph_evidence, str):
        return [line.strip("- ").strip() for line in graph_evidence.splitlines() if line.strip()][:4]
    return [str(x).strip() for x in list(graph_evidence) if str(x).strip()][:4]


def build_training_prompt(
    workpiece_id: str,
    summary: Dict[str, Any],
    params: Dict[str, Any],
    question: str,
    view_items: Sequence[Dict[str, Any]],
    info_text: str = "",
    include_visual_summary: bool = True,
) -> str:
    lines = []
    lines.append(f"工件ID: {workpiece_id}")
    lines.append("视觉缺陷、坐标区域归类和分区域统计已通过模型前缀token输入；不要依赖外部JSON摘要复述。")
    lines.append("请基于前缀token、工艺参数和问题回答，回答必须严格使用【结论】【溯源分析】【建议方向】三段式。")

    if include_visual_summary:
        lines.append("")
        lines.append("【样本摘要】")
        lines.append(f"总缺陷条目={summary.get('total_defect_count', 0)}")
        lines.append(f"总面积(mm²)={summary.get('total_area_mm2', 0)}")
        lines.append(f"最大等效直径(mm)={summary.get('max_eq_diameter_mm', 0)}")
        lines.append(f"区域缺陷统计={_short_dict(summary.get('region_count', {}), max_items=8)}")
        lines.append(f"区域最大面积(mm²)={_short_dict(summary.get('region_max_area_mm2', {}), max_items=8)}")
        lines.append(f"区域最大等效直径(mm)={_short_dict(summary.get('region_max_eq_diameter_mm', {}), max_items=8)}")
        lines.append(f"检测类别汇总={_short_dict(summary.get('class_summary', {}), max_items=6)}")

    top_views = _top_view_lines(view_items, max_items=2) if include_visual_summary else ""
    if include_visual_summary and top_views:
        lines.append("")
        lines.append("【关键视角】")
        lines.append(top_views)

    short_params = {
        k: params.get(k, 0)
        for k in [
            "慢速速度", "高速速度", "料饼厚度", "铸造压力", "循环周期",
            "建压时间", "射料结束位置", "压射压力", "高速行程", "高速时间",
            "慢速行程", "环境温度", "环境湿度"
        ]
        if k in params
    }
    lines.append("")
    lines.append("【参数摘要】")
    lines.append(_short_dict(short_params, max_items=16))

    lines.append("")
    lines.append("【问题】")
    lines.append(question)

    return "\n".join(lines)


def build_inference_prompt(
    workpiece_id: str,
    summary: Dict[str, Any],
    params: Dict[str, Any],
    question: str,
    view_items: Sequence[Dict[str, Any]],
    info_text: str = "",
    graph_evidence: Any = None,
    include_visual_summary: bool = True,
) -> str:
    prompt = build_training_prompt(
        workpiece_id=workpiece_id,
        summary=summary,
        params=params,
        question=question,
        view_items=view_items,
        info_text=info_text,
        include_visual_summary=include_visual_summary,
    )
    graph_lines = _normalize_graph_evidence(graph_evidence)
    if graph_lines:
        evidence = "\n".join([f"- {x}" for x in graph_lines])
        prompt += "\n\n【图谱证据】\n" + evidence
    return prompt


def build_messages(system_prompt: str, user_prompt: str, answer: str | None = None) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages
