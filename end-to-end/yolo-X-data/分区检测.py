# -*- coding: utf-8 -*-
import os
import cv2
import json
import time
import math
import shutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


# =========================================================
# 0. 初始化 SAHI 模型（全局只初始化一次）
# =========================================================
def init_detection_model() -> AutoDetectionModel:
    return AutoDetectionModel.from_pretrained(
        model_type="yolov11",
        model_path="best_LR_75.pt",
        confidence_threshold=0.3,
        device="cuda:1",
        image_size=1280
    )


detection_model = init_detection_model()


# =========================================================
# 1. 标准解析与工具函数
# =========================================================
def parse_diameter_range(dia_str: str) -> Tuple[float, float]:
    s = dia_str.replace("mm", "").strip()
    if "~" in s:
        a, b = s.split("~")
        return float(a), float(b)
    if ">" in s:
        a = s.replace(">", "").strip()
        return float(a), float("inf")
    v = float(s)
    return v, v


def build_sorted_ranges(standards: Dict[str, int]) -> List[Tuple[float, float, str]]:
    ranges = []
    for k in standards.keys():
        mn, mx = parse_diameter_range(k)
        ranges.append((mn, mx, k))
    ranges.sort(key=lambda x: x[0])
    return ranges


def get_diameter_range_key(
    diameter_mm: float,
    sorted_ranges: List[Tuple[float, float, str]]
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    # 前闭后开 [min, max)；最后一个区间右闭
    for i, (mn, mx, key) in enumerate(sorted_ranges):
        is_last = (i == len(sorted_ranges) - 1)
        if is_last:
            if mn <= diameter_mm <= mx:
                return key, mn, mx
        else:
            if mn <= diameter_mm < mx:
                return key, mn, mx
    return None, None, None


def get_min_diameter_threshold(sorted_ranges: List[Tuple[float, float, str]]) -> float:
    return min((r[0] for r in sorted_ranges), default=0.0)


def get_max_diameter_threshold(sorted_ranges: List[Tuple[float, float, str]]) -> float:
    finite_max = [mx for _, mx, _ in sorted_ranges if mx != float("inf")]
    return max(finite_max) if finite_max else float("inf")


def parse_unit_area(unit_str: str) -> float:
    s = unit_str.lower().strip()
    try:
        if "cm²" in s or "cm2" in s:
            num = float(s.replace("cm²", "").replace("cm2", "").strip())
            return num * 100.0
        if "mm²" in s or "mm2" in s:
            num = float(s.replace("mm²", "").replace("mm2", "").strip())
            return num
        return float(s)
    except Exception:
        return 1000.0


def safe_find_contours(binary_u8: np.ndarray) -> List[np.ndarray]:
    ret = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = ret[0] if len(ret) == 2 else ret[1]
    return contours or []


def extract_digits(s: str) -> str:
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits or "output"


# =========================================================
# 1.1 去重：SAHI 切片重叠重复实例 NMS（bbox IoU）
# =========================================================
def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms_dedup_predictions(
    preds: List[Dict[str, Any]],
    iou_thr: float = 0.55
) -> List[Dict[str, Any]]:
    """
    preds: [{"bbox":(x1,y1,x2,y2), "score":float, "mask_bool":np.ndarray}, ...]
    逻辑：按 score 降序，IoU>thr 的视为重复抑制
    """
    if not preds:
        return []

    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    kept: List[Dict[str, Any]] = []
    suppressed = np.zeros(len(preds), dtype=bool)

    for i in range(len(preds)):
        if suppressed[i]:
            continue
        kept.append(preds[i])
        bi = preds[i]["bbox"]
        for j in range(i + 1, len(preds)):
            if suppressed[j]:
                continue
            bj = preds[j]["bbox"]
            if bbox_iou(bi, bj) >= iou_thr:
                suppressed[j] = True
    return kept


# =========================================================
# 2. 判定逻辑（按整图计数规则，不改）
# =========================================================
def judge_image_by_total_counts(
    defects: List[Dict[str, Any]],
    standards: Dict[str, int],
    sorted_ranges: List[Tuple[float, float, str]],
    min_diameter_threshold: float,
    max_standard_d: float,
):
    range_counts = {k: 0 for k in standards.keys()}
    over_max_defects = []
    out_of_range_defects = []

    for d in defects:
        dia = d["Diameter (mm)"]
        if dia < min_diameter_threshold:
            continue
        if dia > max_standard_d:
            over_max_defects.append(d)
            continue
        key, _, _ = get_diameter_range_key(dia, sorted_ranges)
        if key is None:
            out_of_range_defects.append(d)
        else:
            range_counts[key] += 1

    reasons = []
    if over_max_defects:
        reasons.append(f"存在 {len(over_max_defects)} 个缺陷直径 > {max_standard_d}mm，直接判定NG")
    if out_of_range_defects:
        reasons.append(f"存在 {len(out_of_range_defects)} 个缺陷直径不在任何标准区间内，判定NG")

    for key, allowed in standards.items():
        cnt = range_counts.get(key, 0)
        if cnt > allowed:
            reasons.append(f"区间 {key} 缺陷总数 {cnt} > 允许 {allowed}，判定NG")

    current_status = "NG" if reasons else "OK"
    if not reasons:
        reasons.append("所有缺陷均符合整图计数标准")

    for d in defects:
        dia = d["Diameter (mm)"]

        if dia < min_diameter_threshold:
            d["Status"] = "忽略"
            d["Diameter Range"] = d.get("Diameter Range") or "忽略"
            d["Judgment Reason"] = f"直径{dia:.3f}mm < {min_diameter_threshold}mm，忽略"
            continue

        if dia > max_standard_d:
            d["Status"] = "NG"
            d["Diameter Range"] = "超出范围"
            d["Judgment Reason"] = f"直径{dia:.3f}mm > {max_standard_d}mm，直接NG"
            continue

        key, _, _ = get_diameter_range_key(dia, sorted_ranges)
        if key is None:
            d["Status"] = "NG"
            d["Diameter Range"] = "超出范围"
            d["Judgment Reason"] = f"直径{dia:.3f}mm不在标准范围内，判定NG"
            continue

        d["Diameter Range"] = key
        cnt = range_counts[key]
        allowed = standards[key]
        if cnt > allowed:
            d["Status"] = "NG"
            d["Judgment Reason"] = f"区间{key}整图数量{cnt} > 允许{allowed}，判定NG"
        else:
            d["Status"] = "OK"
            d["Judgment Reason"] = f"区间{key}整图数量{cnt}/{allowed}，OK"

    return current_status, reasons, range_counts


# =========================================================
# 3. Excel：轻量样式工具
# =========================================================
def _apply_header_style(ws, header_row_idx: int = 1):
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")
    align = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for cell in ws[header_row_idx]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = align


def _set_column_widths(ws, widths: Dict[int, float]):
    for col_idx, w in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = w


# =========================================================
# 3.1 单图 Excel：Defect Details 为主表，其余表用公式联动生成
# =========================================================
def generate_single_excel_report_fast(
    excel_path: str,
    img_name: str,
    level: int,
    description: str,
    unit_area_str: str,
    unit_area_mm2: float,
    unit_radius_mm: float,
    min_diameter_threshold: float,
    max_diameter_threshold: float,
    actual_size_str: str,
    standards: Dict[str, int],
    defects: List[Dict[str, Any]],
    current_status: str,
    judgment_reasons: List[str],
    range_counts: Dict[str, int],
):
    wb = Workbook(write_only=False)

    # -----------------------------
    # 1) 主表：Defect Details
    # -----------------------------
    ws_details = wb.active
    ws_details.title = "Defect Details"
    ws_details.append(["ID", "直径(mm)", "直径范围", "面积(mm²)", "X坐标", "Y坐标", "状态", "判定依据"])
    for d in defects:
        ws_details.append([
            d["Defect ID"],
            d["Diameter (mm)"],
            d["Diameter Range"] or "超出范围",
            d["Area (mm²)"],
            d["X Position"],
            d["Y Position"],
            d["Status"],
            d["Judgment Reason"]
        ])
    _apply_header_style(ws_details, 1)
    _set_column_widths(ws_details, {1: 8, 2: 12, 3: 14, 4: 12, 5: 10, 6: 10, 7: 10, 8: 48})

    # 主表整列引用（便于后续人工增删改行也联动）
    col_id = "'Defect Details'!A:A"
    col_dia = "'Defect Details'!B:B"
    col_area = "'Defect Details'!D:D"
    col_range = "'Defect Details'!C:C"
    col_status = "'Defect Details'!G:G"

    # -----------------------------
    # 2) Summary：由公式从主表统计（含最大/最小直径与面积）
    # -----------------------------
    ws_summary = wb.create_sheet("Summary")
    ws_summary.append([
        "图片名称", "检测标准等级",
        "最大缺陷直径(mm)", "最小缺陷直径(mm)",
        "最大缺陷面积(mm²)", "最小缺陷面积(mm²)",
        "图片实际尺寸(mm)",
        "总缺陷数", "忽略缺陷数", "NG缺陷数",
        "总体状态"
    ])

    # 说明：
    # - 最大/最小直径/面积：排除“忽略”行（状态 <> 忽略）
    # - 若没有满足条件的数据，用 IFERROR 返回 0
    ws_summary.append([
        img_name,
        level,
        f'=IFERROR(MAXIFS({col_dia},{col_status},"<>忽略"),0)',
        f'=IFERROR(MINIFS({col_dia},{col_status},"<>忽略"),0)',
        f'=IFERROR(MAXIFS({col_area},{col_status},"<>忽略"),0)',
        f'=IFERROR(MINIFS({col_area},{col_status},"<>忽略"),0)',
        actual_size_str,
        f"=MAX(0,COUNTA({col_id})-1)",
        f'=COUNTIF({col_status},"忽略")',
        f'=COUNTIF({col_status},"NG")',
        f'=IF(J2>0,"NG","OK")'
    ])

    _apply_header_style(ws_summary, 1)
    _set_column_widths(ws_summary, {
        1: 28, 3: 16, 4: 16, 5: 16, 6: 16, 7: 20, 11: 12
    })

    # -----------------------------
    # 3) Diameter Range Stats：由公式 COUNTIFS 从主表统计
    # -----------------------------
    ws_range = wb.create_sheet("Diameter Range Stats")
    ws_range.append(["直径范围", "允许数量", "实际数量(整图, 非忽略)", "状态"])

    start_row = 2
    for i, (dia_range, allowed) in enumerate(standards.items()):
        r = start_row + i
        ws_range.append([
            dia_range,
            int(allowed),
            f'=COUNTIFS({col_range},A{r},{col_status},"<>忽略")',
            f'=IF(C{r}>B{r},"NG (超出)","OK ("&C{r}&"/"&B{r}&")")'
        ])
    _apply_header_style(ws_range, 1)
    _set_column_widths(ws_range, {1: 18, 2: 12, 3: 28, 4: 18})

    # -----------------------------
    # 4) Judgment Basis：可联动判定依据（引用/公式）
    # -----------------------------
    ws_judgment = wb.create_sheet("Judgment Basis")
    ws_judgment.append(["判定项目", "判定结果", "详细依据(公式联动)"])

    ws_judgment.append([
        "整图状态",
        "=Summary!K2",
        "若主表中存在任意 NG，则整图为 NG；否则 OK"
    ])
    ws_judgment.append(["NG缺陷数", "=Summary!J2", "来自主表状态列 COUNTIF"])
    ws_judgment.append(["忽略缺陷数", "=Summary!I2", "来自主表状态列 COUNTIF"])
    ws_judgment.append([
        "是否存在超出范围缺陷",
        f'=IF(COUNTIF({col_range},"超出范围")>0,"是","否")',
        f'=COUNTIF({col_range},"超出范围")'
    ])
    ws_judgment.append([
        "是否存在区间数量超限",
        '=IF(COUNTIF(\'Diameter Range Stats\'!D:D,"NG*")>0,"是","否")',
        "由 Diameter Range Stats 自动判断"
    ])
    _apply_header_style(ws_judgment, 1)
    _set_column_widths(ws_judgment, {1: 22, 2: 16, 3: 70})

    # -----------------------------
    # 5) Standards：把那4个字段也换成“最大/最小直径/面积”，并引用 Summary（联动）
    # -----------------------------
    ws_std = wb.create_sheet("Standards")
    ws_std.append([
        "标准等级", "描述",
        "最大缺陷直径(mm)", "最小缺陷直径(mm)",
        "最大缺陷面积(mm²)", "最小缺陷面积(mm²)",
        "图片实际尺寸(mm)",
        "直径范围", "允许数量"
    ])

    for dia_range, max_allowed in standards.items():
        ws_std.append([
            level,
            description,
            "=Summary!C2",
            "=Summary!D2",
            "=Summary!E2",
            "=Summary!F2",
            "=Summary!G2",
            dia_range,
            int(max_allowed)
        ])

    _apply_header_style(ws_std, 1)
    _set_column_widths(ws_std, {2: 30, 3: 16, 4: 16, 5: 16, 6: 16, 7: 18, 8: 16, 9: 12})

    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    wb.save(excel_path)


# =========================================================
# 3.2 汇总 Excel（跨图片汇总，保持原逻辑）
# =========================================================
def generate_excel_summary_fast(
    excel_summary_path: str,
    all_images_summary: List[Dict[str, Any]],
    level: int,
    description: str,
    unit_area_str: str,
    unit_area_mm2: float,
    unit_radius_mm: float,
    min_diameter_threshold: float,
    max_diameter_threshold: float,
    standards: Dict[str, int],
    overall_status: str,
):
    wb = Workbook(write_only=False)

    ws_info = wb.active
    ws_info.title = "总体信息"
    ws_info.append(["孔隙检测总体报告"])
    ws_info.append(["检测标准等级", level])
    ws_info.append(["描述", description])
    ws_info.append(["单位面积(显示用)", f"{unit_area_str} ({unit_area_mm2}mm²)"])
    ws_info.append(["单位圆半径(显示用)", f"{unit_radius_mm:.2f}mm"])
    ws_info.append(["最小直径阈值(忽略)", f"{min_diameter_threshold}mm"])
    ws_info.append(["最大直径阈值(直接NG)", f"{max_diameter_threshold}mm"])
    ws_info.append(["总体状态", overall_status])
    ws_info.append(["处理图片数量", len(all_images_summary)])
    ws_info.append([])
    ws_info.append(["直径范围标准(整图计数):"])
    ws_info.append(["直径范围", "允许数量"])
    for dia_range, max_allowed in standards.items():
        ws_info.append([dia_range, int(max_allowed)])
    _set_column_widths(ws_info, {1: 22, 2: 38})

    ws_results = wb.create_sheet("检测结果汇总")
    ws_results.append(["序号", "图片名称", "状态", "总缺陷数", "忽略缺陷数", "NG缺陷数", "判定依据(前3条)"])
    for idx, img in enumerate(all_images_summary, 1):
        jr = "; ".join(img["judgment_reasons"][:3])
        if len(img["judgment_reasons"]) > 3:
            jr += f" 等{len(img['judgment_reasons'])}条"
        ws_results.append([
            idx, img["img_name"], img["status"],
            img["total_defects"], img["ignored_defects"], img["ng_defects"], jr
        ])
    _apply_header_style(ws_results, 1)
    _set_column_widths(ws_results, {1: 6, 2: 30, 3: 10, 4: 10, 5: 12, 6: 10, 7: 60})

    ws_stats = wb.create_sheet("统计信息")
    ws_stats.append(["统计项目", "数量"])
    ws_stats.append(["总图片数", len(all_images_summary)])
    ok_cnt = sum(1 for img in all_images_summary if img["status"] == "OK")
    ng_cnt = sum(1 for img in all_images_summary if img["status"] == "NG")
    ws_stats.append(["合格图片数", ok_cnt])
    ws_stats.append(["不合格图片数", ng_cnt])
    rate = (ok_cnt / len(all_images_summary) * 100.0) if all_images_summary else 0.0
    ws_stats.append(["合格率", f"{rate:.1f}%"])
    _apply_header_style(ws_stats, 1)
    _set_column_widths(ws_stats, {1: 18, 2: 18})

    os.makedirs(os.path.dirname(excel_summary_path), exist_ok=True)
    wb.save(excel_summary_path)

def to_float_score(score_obj: Any, default: float = 1.0) -> float:
    """
    兼容 SAHI 不同版本的 score 类型：
    - float / int
    - PredictionScore 对象（常见属性：value / score / confidence）
    - dict 或其他可取值结构
    """
    if score_obj is None:
        return float(default)

    # 已经是数值
    if isinstance(score_obj, (float, int, np.floating, np.integer)):
        return float(score_obj)

    # SAHI PredictionScore 常见：score_obj.value / score_obj.score / score_obj.confidence
    for attr in ("value", "score", "confidence"):
        if hasattr(score_obj, attr):
            v = getattr(score_obj, attr)
            try:
                return float(v)
            except Exception:
                pass

    # 可能是 dict
    if isinstance(score_obj, dict):
        for k in ("value", "score", "confidence"):
            if k in score_obj:
                try:
                    return float(score_obj[k])
                except Exception:
                    pass

    # 最后兜底：转字符串再转
    try:
        return float(str(score_obj))
    except Exception:
        return float(default)



# =========================================================
# 4. 主流程（输出目录结构统一 + 直径改为面积等效直径 + 去重）
#    输出目录：outputs/<ts>/{images, reporter}
# =========================================================
def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs", ts)
    out_images = os.path.join(out_dir, "images")
    out_reporter = os.path.join(out_dir, "reporter")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_reporter, exist_ok=True)

    # 读取校准
    with open("calibration_result.json", "r", encoding="utf-8") as f:
        calibration_data = json.load(f)
        pixel_size = float(calibration_data["pixel_size_mm_per_pixel"])
        print(f"校准文件信息: {calibration_data}")

    # 读取标准
    with open("standard2.json", "r", encoding="utf-8") as f:
        standard_data = json.load(f)

    level = int(standard_data.get("level", 1))
    description = standard_data.get("description", "孔隙等级标准")
    standards: Dict[str, int] = standard_data.get("standards", {})
    unit_area_str = standard_data.get("unit_area", "10cm²")

    sorted_ranges = build_sorted_ranges(standards)
    min_diameter_threshold = get_min_diameter_threshold(sorted_ranges)
    max_diameter_threshold = get_max_diameter_threshold(sorted_ranges)

    unit_area_mm2 = parse_unit_area(unit_area_str)
    unit_radius_mm = math.sqrt(unit_area_mm2 / math.pi)

    print(f"使用标准: Level {level}")
    print(f"描述: {description}")
    print(f"单位面积: {unit_area_str} = {unit_area_mm2}mm² (半径: {unit_radius_mm:.2f}mm) [仅用于显示/报告]")
    print(f"直径范围标准: {standards}")
    print(f"忽略阈值: < {min_diameter_threshold}mm")
    print(f"直接NG阈值: > {max_diameter_threshold}mm")
    print("判定方式: 整张图片按区间统计总数，超出允许即NG")

    img_dir = "38048 OK"
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    if not img_files:
        print("未找到图片文件！")
        return

    overall_status = "OK"
    all_images_summary: List[Dict[str, Any]] = []

    for img_idx, img_name in enumerate(img_files, 1):
        print(f"\n{'=' * 60}")
        print(f"处理图片 [{img_idx}/{len(img_files)}]: {img_name}")
        print(f"{'=' * 60}")

        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图片: {img_name}")
            continue

        h, w = image.shape[:2]
        actual_width_mm = w * pixel_size
        actual_height_mm = h * pixel_size
        actual_size_str = f"{actual_width_mm:.1f} x {actual_height_mm:.1f}"

        # SAHI 检测
        print("正在检测缺陷...")
        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.3,
            overlap_width_ratio=0.3
        )

        # 先收集预测（bbox + score + mask）
        raw_preds: List[Dict[str, Any]] = []
        for obj in result.object_prediction_list:
            if obj.mask is None:
                continue

            mask_bool = obj.mask.bool_mask
            if mask_bool is None:
                continue

            # bbox（加速 & 去重用）
            bb = obj.bbox
            try:
                x1, y1, x2, y2 = int(bb.minx), int(bb.miny), int(bb.maxx), int(bb.maxy)
            except Exception:
                x1, y1, x2, y2 = map(int, bb.to_xyxy())

            # ✅ 修复：兼容 SAHI 的 PredictionScore
            score = to_float_score(getattr(obj, "score", None), default=1.0)

            raw_preds.append({"bbox": (x1, y1, x2, y2), "score": score, "mask_bool": mask_bool})

        # 去重（避免切片重叠重复缺陷）
        dedup_preds = nms_dedup_predictions(raw_preds, iou_thr=0.55)

        # 转 defect dict：直径=面积等效直径（替代 minEnclosingCircle）
        defects: List[Dict[str, Any]] = []
        defect_id = 1

        for p in dedup_preds:
            mask_bool = p["mask_bool"]

            area_px = int(mask_bool.sum())
            if area_px <= 0:
                continue

            area_mm2 = area_px * (pixel_size ** 2)

            # 面积等效直径：d = 2*sqrt(A/pi)
            dia_mm = 2.0 * math.sqrt(area_mm2 / math.pi)

            # 中心点：用 mask 的质心（快且稳）
            ys, xs = np.nonzero(mask_bool)
            if len(xs) == 0:
                continue
            cx = int(xs.mean())
            cy = int(ys.mean())

            # 轮廓只用于画图（可选），并且只找一次最大轮廓
            binary = (mask_bool.astype(np.uint8) * 255)
            contours = safe_find_contours(binary)
            contour_max = max(contours, key=cv2.contourArea) if contours else None

            dia_range_key, _, _ = get_diameter_range_key(dia_mm, sorted_ranges)

            defects.append({
                "Defect ID": defect_id,
                "Diameter (mm)": round(float(dia_mm), 3),
                "Area (mm²)": round(float(area_mm2), 3),
                "X Position": cx,
                "Y Position": cy,
                "Diameter Range": dia_range_key,
                "Status": "待判定",
                "Judgment Reason": "",
                "Contour": contour_max,
                "BBox": p["bbox"],
                "Score": round(float(p["score"]), 4),
            })
            defect_id += 1

        print(f"检测完成: 原始{len(raw_preds)}个 → 去重后{len(defects)}个缺陷")

        # 判定（整图计数规则）
        print("开始判定（整图计数规则）...")
        current_status, judgment_reasons, range_counts = judge_image_by_total_counts(
            defects=defects,
            standards=standards,
            sorted_ranges=sorted_ranges,
            min_diameter_threshold=min_diameter_threshold,
            max_standard_d=max_diameter_threshold
        )

        total_ignored = sum(1 for d in defects if d["Status"] == "忽略")
        total_ng = sum(1 for d in defects if d["Status"] == "NG")
        if current_status == "NG":
            overall_status = "NG"
            judgment_reasons.insert(0, f"检测到{total_ng}个NG缺陷")

        all_images_summary.append({
            "img_name": img_name,
            "status": current_status,
            "total_defects": len(defects),
            "ignored_defects": total_ignored,
            "ng_defects": total_ng,
            "judgment_reasons": judgment_reasons,
            "range_counts": range_counts
        })

        # 绘制结果
        print("绘制结果图像...")
        vis = image.copy()

        for d in defects:
            contour = d["Contour"]
            cx, cy = d["X Position"], d["Y Position"]
            diameter_mm = d["Diameter (mm)"]
            did = d["Defect ID"]
            st = d["Status"]

            if st == "忽略":
                contour_color = (255, 200, 0)
                text_color = (255, 200, 0)
            elif st == "NG":
                contour_color = (0, 0, 255)
                text_color = (0, 0, 255)
            else:
                contour_color = (0, 255, 0)
                text_color = (0, 255, 0)

            if contour is not None:
                # 只画轮廓，不再为 NG 额外画红色矩形框（避免遮挡）
                cv2.drawContours(vis, [contour], -1, contour_color, 2)
            else:
                # contour 为空才画 bbox（同样不加额外红框）
                x1, y1, x2, y2 = d["BBox"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), contour_color, 2)

            cv2.putText(vis, f"{did}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            if diameter_mm < 0.5:
                cv2.putText(vis, f"{diameter_mm:.2f}mm", (cx, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            else:
                cv2.putText(vis, f"{diameter_mm:.1f}mm", (cx, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # 左上角信息
        overlay = vis.copy()
        cv2.rectangle(overlay, (10, 10), (420, 260), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

        info_y = 40
        cv2.putText(vis, f"Level: {level}", (20, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        info_y += 30
        cv2.putText(vis, f"Defects: {len(defects)} ({total_ng} NG)", (20, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 25
        cv2.putText(vis, f"Ignored(<{min_diameter_threshold}): {total_ignored}", (20, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # 右上角状态
        status_color = (0, 0, 255) if current_status == "NG" else (0, 255, 0)
        status_x = w - 320
        status_y = 80
        cv2.putText(vis, "Status:", (status_x, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(vis, f"{current_status}", (status_x + 150, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, status_color, 5)

        # 标准信息（显示用）
        standard_text_y = 150
        cv2.putText(vis, f"Unit Area: {unit_area_str}", (w - 320, standard_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        standard_text_y += 20
        cv2.putText(vis, "(Note: judge by whole image counts)", (w - 320, standard_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        standard_text_y += 20
        cv2.putText(vis, f"Ignore < {min_diameter_threshold}mm", (w - 320, standard_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        standard_text_y += 20
        cv2.putText(vis, f"NG > {max_diameter_threshold}mm", (w - 320, standard_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        standard_text_y += 20
        for dia_range, max_allowed in standards.items():
            cnt = range_counts.get(dia_range, 0)
            cv2.putText(vis, f"{dia_range}: {cnt}/{max_allowed}",
                        (w - 320, standard_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            standard_text_y += 20

        out_img = os.path.join(out_images, f"{os.path.splitext(img_name)[0]}_{current_status}.png")
        cv2.imwrite(out_img, vis)
        print(f"结果图片已保存: {out_img}")

        # 单图 Excel（主表驱动联动）
        excel_path = os.path.join(out_reporter, f"{os.path.splitext(img_name)[0]}_report.xlsx")
        generate_single_excel_report_fast(
            excel_path=excel_path,
            img_name=img_name,
            level=level,
            description=description,
            unit_area_str=unit_area_str,
            unit_area_mm2=unit_area_mm2,
            unit_radius_mm=unit_radius_mm,
            min_diameter_threshold=min_diameter_threshold,
            max_diameter_threshold=max_diameter_threshold,
            actual_size_str=actual_size_str,
            standards=standards,
            defects=defects,
            current_status=current_status,
            judgment_reasons=judgment_reasons,
            range_counts=range_counts,
        )
        print(f"Excel报告已保存: {excel_path}")

        print(f"\n判定结果: {current_status}")
        for reason in (judgment_reasons[:3] if current_status == "NG" else judgment_reasons):
            print(f"  - {reason}")

    # 汇总 Excel（跨图片汇总）
    excel_summary_path = os.path.join(out_reporter, f"汇总报告_Level{level}.xlsx")
    generate_excel_summary_fast(
        excel_summary_path=excel_summary_path,
        all_images_summary=all_images_summary,
        level=level,
        description=description,
        unit_area_str=unit_area_str,
        unit_area_mm2=unit_area_mm2,
        unit_radius_mm=unit_radius_mm,
        min_diameter_threshold=min_diameter_threshold,
        max_diameter_threshold=max_diameter_threshold,
        standards=standards,
        overall_status=overall_status
    )
    print(f"Excel汇总报告已生成: {excel_summary_path}")

    # 输出目录统一重命名：outputs/<base>_<OK|NG>_LevelX
    base_num = extract_digits(os.path.basename(img_dir))
    final_dir = os.path.join("outputs", f"{base_num}_{overall_status}_Level{level}")

    if os.path.exists(final_dir):
        shutil.rmtree(final_dir, ignore_errors=True)

    try:
        os.rename(out_dir, final_dir)
        print(f"\n{'=' * 60}")
        print(f"所有处理完成 → {final_dir}")
        print("目录结构：")
        print(f"  - {os.path.join(final_dir, 'images')}     (结果图片)")
        print(f"  - {os.path.join(final_dir, 'reporter')}   (单图报告 + 汇总报告)")
    except Exception as e:
        print(f"重命名文件夹失败: {e}")
        print(f"结果保存在: {out_dir}")


if __name__ == "__main__":
    main()
