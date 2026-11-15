from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


SECTION_PATTERN = re.compile(
    r"【结论】(?P<conclusion>.*?)【溯源分析】(?P<analysis>.*?)【建议方向】(?P<suggestion>.*)",
    re.DOTALL,
)


QUESTION_TYPE_HINTS_V2 = {
    "effect_analysis": "当前问题属于影响分析类。重点分析指定参数或条件对缺陷数量、区域分布和缺陷极值的影响。",
    "cause_explanation": "当前问题属于原因解释类。重点解释缺陷较多或异常表现形成的可能原因，并区分主因链与次因链。",
    "relation_mechanism": "当前问题属于参数/机理关系类。重点说明参数之间的关系，以及这种关系如何影响充型、排气、补缩或压力传递。",
    "adjustment_validation": "当前问题属于调整/验证类。重点说明优先排查方向、验证指标和前后样本对比方式。",
    "inspection_identification": "当前问题属于探伤识别/区分类。重点说明影像表现、区域特征和判定依据。",
    "yes_no_judgment": "当前问题属于是非判断类。必须先明确回答“可能/不宜直接判断/暂不支持”，再说明依据和限制条件。",
    "binary_decision": "当前问题属于二选一判断类。必须明确更偏向哪一项，并说明另一项为何不是当前优先判断。",
    "open_diagnosis": "当前问题属于开放诊断类。先判断数量主导区域和风险关注区域，再分析参数链和排查优先级。",
}


def save_text(path: str | Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text is not None else "", encoding="utf-8")


def save_json(path: str | Path, obj: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def classify_question_v2(q: str) -> str:
    q = (q or "").strip()

    # 二选一要最先判定
    if re.search(r"还是|优先.*还是|二选一|哪一种|哪类更|更偏向", q):
        return "binary_decision"

    # 是非判断：补上“是否可能/是否有关/能不能判断”等旧版漏掉的表达
    if re.search(r"(是否可能|是否.*有关|是否.*相关|有没有关系|有关系吗|有关吗|相关吗|会不会|能否|是否可以|是否需要|是否合理|是否同步|正常吗)", q):
        return "yes_no_judgment"

    # 机制关系
    if re.search(r"(关系|比值|区别|如何计算|协同影响|交互作用|设定顺序|乘积|匹配|关联)", q):
        return "relation_mechanism"

    # 识别
    if re.search(r"(如何准确区分|如何区分|如何判定|影像特征|是什么缺陷|是什么现象|探伤图上的解释|什么意思|如何评级)", q):
        return "inspection_identification"

    # 调整验证
    if re.search(r"(如何调整|如何改善|如何通过.*改善|如何参数调整|如何预防|应如何调整|如何判断最佳值|如何监测|如何现场快速验证|如何优化|如何确保|如何选择|如何避免|如何排查|针对性解决|怎么办|如何平衡|如何确定|如何解决|验证|对比哪些指标)", q):
        return "adjustment_validation"

    # 原因
    if re.search(r"(为什么|原因|可能是什么原因|可能原因|有何后果|会导致什么|导致什么|会产生什么缺陷|后果是什么|怎么形成|主要成因|是什么原因)", q):
        return "cause_explanation"

    # 影响
    if re.search(r"(有何影响|对.+影响|如何变化|有什么特征|效果如何|好处)", q):
        return "effect_analysis"

    return "open_diagnosis"


def detect_mechanism_candidates_v2(q: str) -> List[str]:
    candidates = []
    for term in ["卷气类缺陷保留", "补缩失效类缺陷保留", "气孔", "缩孔", "缩松", "裂纹", "冷隔", "热裂", "变形"]:
        base = term.replace("类缺陷保留", "")
        if base in q or term in q:
            candidates.append(term)
    if "卷气" in q and "补缩" in q:
        candidates = ["卷气类缺陷保留", "补缩失效类缺陷保留"]
    return unique_keep_order(candidates)


def region_flags_v2(q: str) -> List[str]:
    regs = []
    for term in ["一般区域", "牙孔区域", "焊接区域", "重点区域", "厚大区域", "背景", "热节", "分型面", "浇口附近"]:
        if term in q:
            regs.append(term)
    return regs


def build_question_type_hint_v2(question: str) -> Tuple[str, str, List[str], List[str]]:
    qt = classify_question_v2(question)
    regions = region_flags_v2(question)
    mechanisms = detect_mechanism_candidates_v2(question)

    lines = [f"【问题类型】{qt}"]
    lines.append(f"【类型提示】{QUESTION_TYPE_HINTS_V2.get(qt, QUESTION_TYPE_HINTS_V2['open_diagnosis'])}")
    if regions:
        lines.append("【关注区域】" + "；".join(regions))
    if mechanisms:
        lines.append("【候选机理】" + "；".join(mechanisms))

    lines.append(
        "【输出格式硬约束】输出必须且只能包含【结论】【溯源分析】【建议方向】三部分；"
        "【溯源分析】必须1.2.3三点；【建议方向】必须1.2.3三点；"
        "禁止输出图谱命中列表、命中关系链、命中结论、调试信息、原始片段。"
    )
    return qt, "\n".join(lines), regions, mechanisms


def normalize_output_v2(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<\s*/?\s*think\s*>", "", text, flags=re.I)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
    text = text.strip()

    # 如果模型前面泄露了图谱内容，只截取三段式开头之后的内容
    m = re.search(r"【结论】", text)
    if m:
        text = text[m.start():].strip()

    # 不允许把异常片段放入最终答案
    text = re.sub(r"\n?【生成异常片段】.*$", "", text, flags=re.S)
    text = re.sub(r"\n?【图谱证据】.*$", "", text, flags=re.S)
    text = re.sub(r"\n?三、命中关系链[:：]?.*$", "", text, flags=re.S)
    text = re.sub(r"\n?四、命中结论[:：]?.*$", "", text, flags=re.S)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_valid_three_sections_v2(text: str) -> bool:
    text = normalize_output_v2(text)
    m = SECTION_PATTERN.search(text or "")
    if not m:
        return False
    c = m.group("conclusion").strip()
    a = m.group("analysis").strip()
    s = m.group("suggestion").strip()
    if len(c) < 10 or len(a) < 80 or len(s) < 50:
        return False
    if not all(re.search(rf"(^|\n)\s*{i}[\.、]", a) for i in [1, 2, 3]):
        return False
    if not all(re.search(rf"(^|\n)\s*{i}[\.、]", s) for i in [1, 2, 3]):
        return False
    forbidden = ["命中关系链", "命中结论", "图谱调试信息", "生成异常片段"]
    if any(x in text for x in forbidden):
        return False
    return True


def has_repeated_sections_v2(text: str) -> bool:
    m = SECTION_PATTERN.search(text or "")
    if not m:
        return False
    parts = [re.sub(r"\s+", "", m.group(x)) for x in ["conclusion", "analysis", "suggestion"]]
    return len(set(parts)) != len(parts)


def get_dominant_and_risk(summary: Dict[str, Any]) -> Tuple[str, str]:
    region_count = summary.get("region_count", {}) or {}
    region_dia = summary.get("region_max_eq_diameter_mm", {}) or {}
    dominant_region = max(region_count.items(), key=lambda x: x[1])[0] if region_count else "未知"
    risk_region = max(region_dia.items(), key=lambda x: x[1])[0] if region_dia else dominant_region
    return dominant_region, risk_region


def build_visual_fact_block_v2(summary: Dict[str, Any]) -> str:
    region_count = summary.get("region_count", {}) or {}
    region_area = summary.get("region_max_area_mm2", {}) or {}
    region_dia = summary.get("region_max_eq_diameter_mm", {}) or {}
    class_summary = summary.get("class_summary", {}) or {}

    rc = "；".join(f"{k}:{v}" for k, v in region_count.items()) or "无"
    ra = "；".join(f"{k}:{v}" for k, v in region_area.items()) or "无"
    rd = "；".join(f"{k}:{v}" for k, v in region_dia.items()) or "无"
    cs = "；".join(f"{k}:{v}" for k, v in class_summary.items()) or "无"

    return f"""【当前样本硬事实】
总缺陷条目={summary.get("total_defect_count", 0)}
总面积(mm²)={summary.get("total_area_mm2", 0)}
最大等效直径(mm)={summary.get("max_eq_diameter_mm", 0)}
区域缺陷统计={rc}
区域最大面积(mm²)={ra}
区域最大等效直径(mm)={rd}
检测类别汇总={cs}
IsPore统计={summary.get("is_pore_count", 0)}"""


def extract_question_terms(question: str) -> List[str]:
    candidates = [
        "高速速度", "高速行程", "高速时间", "射料结束位置", "慢速速度", "慢速行程",
        "建压时间", "压射压力", "铸造压力", "料饼厚度", "循环周期", "环境温度", "环境湿度",
        "充型", "填充", "卷气", "冷隔", "排气", "压力传递", "补缩", "缺陷", "区域分布"
    ]
    return [x for x in candidates if x in question]


def compress_graph_context_for_question(graph_context: str, question: str, max_items: int = 6) -> str:
    """
    不再把整段GraphRAG证据塞给模型，只保留和当前问题强相关的短证据。
    这能显著减少“命中关系链/命中结论”泄露到最终回答。
    """
    if not graph_context:
        return ""

    terms = extract_question_terms(question)
    if not terms:
        terms = ["高速速度", "高速行程", "缺陷", "充型", "卷气", "压力"]

    useful = []
    for raw_line in graph_context.splitlines():
        line = raw_line.strip("- ").strip()
        if not line:
            continue
        if any(bad in line for bad in ["【图谱证据】", "命中检索词", "命中实体", "关联规则", "以下内容来自"]):
            continue
        if any(t in line for t in terms):
            useful.append(line)

    useful = unique_keep_order(useful)[:max_items]
    if not useful:
        return ""

    return "【图谱压缩证据】\n" + "\n".join(f"- {x}" for x in useful)


def build_instruction_v2(question_type_hint: str) -> str:
    return f"""你是一名铝合金压铸缺陷诊断助手。
你的任务是根据多模态前缀token中的视觉缺陷/坐标区域统计、工艺参数、图谱压缩证据和用户问题，输出工程化诊断结果。

{question_type_hint}

【严格禁止】
1. 禁止输出<think>或思考过程。
2. 禁止输出“命中关系链”“命中结论”“图谱证据列表”等调试内容。
3. 禁止把图谱内容原样复制到答案。
4. 禁止在【建议方向】后追加任何异常片段。
"""


def build_user_prompt_v2(
    info_text: str,
    question: str,
    graph_context: str,
    visual_fact_block: str,
    question_type_hint: str,
) -> str:
    compact_graph = compress_graph_context_for_question(graph_context, question)
    return f"""【样本信息】
{question_type_hint}

{info_text.strip()}

{visual_fact_block.strip()}

【当前样本数字硬约束】
正式回答中的总缺陷条目、总面积、最大等效直径、各区域缺陷数、各区域最大面积和各区域最大等效直径，必须严格来自上面的当前样本硬事实；禁止使用训练语料旧样本中的数字，禁止自行估算或改写这些数字。

{compact_graph}

视觉缺陷、坐标区域归类和分区域统计已由多模态前缀token输入；上面的硬事实是由当前视觉token聚合得到的文本锚点，不是外部JSON提示文件。

【问题】
{question}

【回答要求】
1. 必须直接回答问题，不要只做泛化诊断。
2. 如果问题问“是否可能有关”，必须先明确回答“可能有关/暂不支持/不能直接判断”。
3. 必须使用多模态前缀token中的区域分布和缺陷极值作为判断依据。
4. 图谱只可作为辅助证据，不可原样复制。
5. 输出必须且只能包含【结论】【溯源分析】【建议方向】三部分。
"""
