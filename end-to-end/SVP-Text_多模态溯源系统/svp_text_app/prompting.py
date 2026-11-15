from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .bootstrap import ensure_region_route_on_path


SVP_TEXT_SYSTEM_PROMPT = """你是一名铝合金压铸气孔缺陷溯源助手。

硬性规则：
1. 最终回答必须直接从【结论】开始。
2. 最终回答必须且只能包含【结论】【溯源分析】【建议方向】三部分。
3. 缺陷类型统一按气孔处理。
4. 牙孔区域、焊接区域、重点区域属于高影响区域；一般区域作为整体波动参考。
5. 回答必须依据当前输入中的视觉层文字总结、压铸参数、问题和图谱辅助证据。
6. 图谱证据只辅助机理判断，不能覆盖当前样本；最终不要输出图谱命中列表、规则编号或调试信息。
7. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。

当前方法：SVP-Text。
SVP-Text 的数据流是：图片先由视觉层模型识别为缺陷区域统计和视觉文字总结，再与压铸参数、用户问题和可选图谱证据共同写入提示词，由语言模型完成溯源分析。

请直接作答，不要解释格式要求。"""


PARAM_PROMPT_ORDER = [
    "慢速速度", "高速速度", "料饼厚度", "铸造压力", "循环周期",
    "建压时间", "射料结束位置", "压射压力", "高速行程", "高速时间",
    "慢速行程", "环境温度", "环境湿度", "温度", "湿度",
    "探伤图编号", "样本编号",
]


def _e2e_helpers():
    ensure_region_route_on_path()
    try:
        from e2e_compare_three_modes_graph_4b_v3 import (
            build_info_text_for_prompt,
            build_question_type_hint,
            build_visual_fact_block,
            format_params_for_prompt,
            sanitize_graph_context_for_prompt,
        )

        return {
            "build_info_text_for_prompt": build_info_text_for_prompt,
            "build_question_type_hint": build_question_type_hint,
            "build_visual_fact_block": build_visual_fact_block,
            "format_params_for_prompt": format_params_for_prompt,
            "sanitize_graph_context_for_prompt": sanitize_graph_context_for_prompt,
        }
    except Exception:
        return None


def _format_params_for_prompt(params: Dict[str, Any] | None) -> str:
    if not isinstance(params, dict) or not params:
        return ""
    lines: List[str] = []
    used = set()
    for key in PARAM_PROMPT_ORDER:
        if key in params and params[key] not in ("", None):
            lines.append(f"{key}={params[key]}")
            used.add(key)
    for key, value in params.items():
        if key not in used and value not in ("", None):
            lines.append(f"{key}={value}")
    return "\n".join(lines)


def _build_info_text_for_prompt(cfg_json: Dict[str, Any]) -> str:
    sample_id = cfg_json.get("sample_id", cfg_json.get("workpiece_id", ""))
    lines: List[str] = []
    if sample_id not in ("", None):
        lines.append(f"【样本编号】{sample_id}")
    params_text = _format_params_for_prompt(cfg_json.get("params", {}))
    if params_text:
        lines.append("【压铸参数摘要】")
        lines.append(params_text)
    return "\n".join(lines)


def _build_visual_fact_block(summary: Dict[str, Any]) -> str:
    region_count = summary.get("region_count", {}) or {}
    region_area = summary.get("region_max_area_mm2", {}) or {}
    region_dia = summary.get("region_max_eq_diameter_mm", {}) or {}
    return "\n".join([
        "【视觉层文字总结】",
        "缺陷类型统一按气孔处理；视觉统计仅包含有效工艺区域。",
        f"过滤后共保留{summary.get('total_defect_count', 0)}条有效气孔，总面积约{summary.get('total_area_mm2', 0)}mm²，最大等效直径约{summary.get('max_eq_diameter_mm', 0)}mm。",
        "区域气孔数量分布为：" + json.dumps(region_count, ensure_ascii=False) + "。",
        "各区域最大面积为：" + json.dumps(region_area, ensure_ascii=False) + "。",
        "各区域最大等效直径为：" + json.dumps(region_dia, ensure_ascii=False) + "。",
    ])


def _sanitize_graph_context(text: str, max_chars: int = 700) -> str:
    text = (text or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n（图谱辅助证据已截断）"
    return text


def _limit_text(text: str, max_chars: int, label: str) -> str:
    text = (text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + f"\n（{label}已截断，优先保留当前样本核心事实）"


def build_visual_text(summary: Dict[str, Any]) -> str:
    helpers = _e2e_helpers()
    if helpers:
        return helpers["build_visual_fact_block"](summary)
    return _build_visual_fact_block(summary)


def classify_question(question: str) -> Tuple[str, str, List[str], List[str]]:
    helpers = _e2e_helpers()
    if helpers:
        return helpers["build_question_type_hint"](question)
    return "open_diagnosis", "当前问题属于开放诊断类。先抓数量主导趋势和风险关注趋势，再分析参数链和优先级。", [], []


def build_svp_text_prompt(
    cfg_json: Dict[str, Any],
    question: str,
    visual_fact_block: str,
    graph_context: str = "",
    question_type_hint: str = "",
) -> str:
    sections = build_prompt_sections(
        cfg_json=cfg_json,
        question=question,
        visual_fact_block=visual_fact_block,
        graph_context=graph_context,
        question_type_hint=question_type_hint,
    )
    return sections["user_prompt"]


def build_prompt_sections(
    cfg_json: Dict[str, Any],
    question: str,
    visual_fact_block: str,
    graph_context: str = "",
    question_type_hint: str = "",
) -> Dict[str, str]:
    helpers = _e2e_helpers()
    if helpers:
        info_text = helpers["build_info_text_for_prompt"](cfg_json)
        graph_context = helpers["sanitize_graph_context_for_prompt"](
            graph_context,
            max_chars=int(cfg_json.get("prompt_graph_max_chars", 700)),
        )
    else:
        info_text = _build_info_text_for_prompt(cfg_json)
        graph_context = _sanitize_graph_context(graph_context, int(cfg_json.get("prompt_graph_max_chars", 700)))

    visual_fact_block = _limit_text(
        visual_fact_block,
        int(cfg_json.get("prompt_visual_max_chars", 3000)),
        "视觉事实块",
    )
    info_text = _limit_text(
        info_text,
        int(cfg_json.get("prompt_params_max_chars", 1800)),
        "压铸参数摘要",
    )
    graph_block = f"\n【图谱辅助证据】\n{graph_context}\n" if graph_context else ""
    qtype_block = f"\n【当前问题类型提示】\n{question_type_hint.strip()}\n" if question_type_hint else ""

    user_prompt = f"""{visual_fact_block}

【压铸参数】
{info_text}
{qtype_block}{graph_block}
【问题】
{question.strip()}

【作答边界】
1. 当前方法为 SVP-Text：视觉信息只来自视觉层文字总结，压铸参数只通过文本提供。
2. 必须使用视觉层文字总结中的区域数量分布、区域最大面积和区域最大等效直径形成风险判断。
3. 缺陷统一按气孔处理，不引入缩孔、缩松、裂纹、冷隔、夹杂、夹渣等非气孔缺陷。
4. 图谱证据只能辅助机理判断，不能覆盖当前样本视觉统计和参数事实。
5. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。

【必须遵守的输出格式】
第一行必须是【结论】。
【结论】必须直接给当前样本判断，包含排查优先级或风险优先级。
【溯源分析】下面必须写 1. 2. 3. 三个编号点：1解释区域证据，2分析参数链，3给出综合优先级。
【建议方向】下面必须写 1. 2. 3. 三个编号点：1写优先排查链路，2写联动复核，3写同料号、同视角、同工况验证。

现在开始作答。"""

    return {
        "visual_fact_block": visual_fact_block,
        "params_block": info_text,
        "question_type_hint": question_type_hint.strip(),
        "graph_context": graph_context,
        "question": question.strip(),
        "user_prompt": user_prompt,
    }
