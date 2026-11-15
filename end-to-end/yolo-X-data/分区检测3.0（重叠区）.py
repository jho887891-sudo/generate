import cv2
import numpy as np
import os
import time
import json
import shutil
import torch

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


# =========================================================
# 0. 路径与参数配置
# =========================================================
ROOT_DIR = "/home/T7/zyx/finetune/yolo-X-data"

MODEL_PATH = os.path.join(ROOT_DIR, "best_LR_75.pt")
CALIBRATION_JSON = os.path.join(ROOT_DIR, "calibration_result.json")
REGION_TYPES_JSON = os.path.join(ROOT_DIR, "region_types.json")
STANDARD_JSON = os.path.join(ROOT_DIR, "standard31.json")

IMG_FOLDER = os.path.join(ROOT_DIR, "38048 OK")
TEMPLATE_FOLDER = os.path.join(ROOT_DIR, "11193 NG")
MASK_FOLDER = os.path.join(ROOT_DIR, "VK3-mask")
OUTPUT_ROOT = os.path.join(ROOT_DIR, "output")

CONFIDENCE_THRESHOLD = 0.3

# 恢复你原来那套更保精度的参数
MODEL_IMAGE_SIZE = 1280
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.3
OVERLAP_WIDTH_RATIO = 0.3

USE_MASK_ALIGNMENT = True
SAVE_VIS_IMAGE = True
GENERATE_EXCEL = True


# =========================================================
# 1. 基础优化
# =========================================================
cv2.setUseOptimized(True)

try:
    cv2.setNumThreads(max(1, min(8, os.cpu_count() or 8)))
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    gpu_count = torch.cuda.device_count()
    print(f"检测到 GPU 数量: {gpu_count}")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    DEVICE = "cuda:0"
else:
    print("未检测到 CUDA，使用 CPU")
    DEVICE = "cpu"


# =========================================================
# 2. 初始化 SAHI 模型
# =========================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov11",
    model_path=MODEL_PATH,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    device=DEVICE,
    image_size=MODEL_IMAGE_SIZE,
)


# =========================================================
# 3. 工具函数
# =========================================================
def count_defects_in_circle(center, defects_list, radius_mm,
                            pixel_size_mm_per_pixel, max_area=None, is_pore=False):
    count = 0
    cx, cy = center
    radius_px = radius_mm / pixel_size_mm_per_pixel

    for d in defects_list:
        dist = np.sqrt((d["X Position"] - cx) ** 2 + (d["Y Position"] - cy) ** 2)
        if (
            dist <= radius_px
            and (max_area is None or d["Area (mm²)"] <= max_area)
            and d["IsPore"] == is_pore
        ):
            count += 1

    return count


def set_excel_style(ws):
    thin = Side(style="thin")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4F81BD", end_color="4F81BD", fill_type="solid")

    for row in ws.iter_rows():
        for c in row:
            c.border = border
            c.alignment = Alignment(horizontal="center", vertical="center")

    if ws.max_row >= 1:
        for c in ws[1]:
            c.font = header_font
            c.fill = header_fill

    for col in ws.columns:
        values = ["" if cell.value is None else str(cell.value) for cell in col]
        length = max((len(v) for v in values), default=10)
        ws.column_dimensions[get_column_letter(col[0].column)].width = (length + 2) * 1.2


def align_mask_to_image(img, template, mask):
    """
    将模板和掩膜对齐到输入图像
    返回对齐后的掩膜
    这里保持你原来的高精度思路，不做激进降配
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_gray, None)
    kp2, des2 = sift.detectAndCompute(template_gray, None)

    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        print("特征点不足，使用原始掩膜")
        return cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        print("匹配点不足，使用原始掩膜")
        return cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    pts_img = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_template = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(pts_template, pts_img, cv2.RANSAC, 5.0)

    if M is None:
        print("无法计算变换矩阵，使用原始掩膜")
        return cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    h, w = img_gray.shape
    aligned_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    return aligned_mask


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 3.1 区域过滤规则
# =========================================================
# 当前蒙版约定：
#   0 / 6：背景或非工艺区域，不计入缺陷统计
#   1：一般区域
#   2：牙孔区域
#   3：焊接区域
#   4：重点区域
#   5：重叠区域，不计入缺陷统计
#
# 这里的过滤发生在 defects.append(...) 之前。
# 因此，落在背景/未知/重叠区域的检测结果不会进入：
# - 缺陷总数
# - NG 总判定
# - 可视化标注图
# - Excel 的 Region Summary / Defect Details
# =========================================================
BACKGROUND_UNKNOWN_REGION_KEYWORDS = ("背景", "未知", "重叠")
OVERLAP_REGION_VALUES = {"5", "5.0"}
BACKGROUND_REGION_VALUES = {"0", "0.0", "6", "6.0"}


def is_ignored_region(region_str, region_val=None):
    """
    判断当前区域是否应该从缺陷统计中剔除。

    过滤对象：
    1. 背景区域：蒙版值 0 或 6，或名称中包含“背景”
    2. 未知区域：名称中包含“未知”
    3. 重叠区域：蒙版值 5，或名称中包含“重叠”
    """
    s = str(region_str or "").strip()
    v = str(region_val or "").strip()

    if not s and not v:
        return True

    if any(k in s for k in BACKGROUND_UNKNOWN_REGION_KEYWORDS):
        return True

    if s in {"None", "null"}:
        return True

    if s in BACKGROUND_REGION_VALUES or s in OVERLAP_REGION_VALUES:
        return True

    if v in BACKGROUND_REGION_VALUES or v in OVERLAP_REGION_VALUES:
        return True

    return False


# =========================================================
# 4. 主流程
# =========================================================
def main():
    ensure_dir(OUTPUT_ROOT)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(OUTPUT_ROOT, ts)
    data_root = os.path.join(OUTPUT_ROOT, f"{ts}_data")

    ensure_dir(out_root)
    ensure_dir(data_root)

    pixel_size = safe_read_json(CALIBRATION_JSON)["pixel_size_mm_per_pixel"]
    region_types = safe_read_json(REGION_TYPES_JSON)
    standards = safe_read_json(STANDARD_JSON)

    # 兼容旧 region_types.json：如果还没有写入 5，这里默认把 5 当作重叠区域。
    # 建议 region_types.json 同步改为：
    # {
    #   "6": "背景",
    #   "1": "一般区域",
    #   "2": "牙孔区域",
    #   "3": "焊接区域",
    #   "4": "重点区域",
    #   "5": "重叠区域"
    # }
    region_types.setdefault("5", "重叠区域")

    ignored_region_standard = {
        "allow_defects": False,
        "allow_pores": False,
        "observation_diameter_mm": 0,
        "max_pores_per_observation": 0,
        "max_pore_area_mm2": 0,
        "max_defects_per_observation": 0,
        "max_defect_area_mm2": 0,
    }
    standards["背景"] = ignored_region_standard
    standards["重叠区域"] = ignored_region_standard

    img_files = sorted([f for f in os.listdir(IMG_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    template_files = sorted([f for f in os.listdir(TEMPLATE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    mask_files = sorted([f for f in os.listdir(MASK_FOLDER) if f.lower().endswith(".png")])

    if len(img_files) != len(template_files) or len(img_files) != len(mask_files):
        raise ValueError("图像、模板与掩膜数量不一致！")

    print("=" * 80)
    print(f"DEVICE={DEVICE}")
    print(f"MODEL_IMAGE_SIZE={MODEL_IMAGE_SIZE}")
    print(f"SLICE=({SLICE_HEIGHT}, {SLICE_WIDTH})")
    print(f"OVERLAP=({OVERLAP_HEIGHT_RATIO}, {OVERLAP_WIDTH_RATIO})")
    print(f"USE_MASK_ALIGNMENT={USE_MASK_ALIGNMENT}")
    print(f"GENERATE_EXCEL={GENERATE_EXCEL}")
    print("FILTERED_REGIONS=背景/未知/重叠区域，其中蒙版值5为重叠区域，不计入缺陷总计")
    print("=" * 80)

    overall_status = "OK"

    for img_name, template_name, mask_name in zip(img_files, template_files, mask_files):
        t0 = time.time()
        print(f"\n处理图像: {img_name}")

        # 4.1 读取
        image = cv2.imread(os.path.join(IMG_FOLDER, img_name))
        template = cv2.imread(os.path.join(TEMPLATE_FOLDER, template_name))
        mask = cv2.imread(os.path.join(MASK_FOLDER, mask_name), cv2.IMREAD_GRAYSCALE)

        if image is None or template is None or mask is None:
            print(f"无法读取图像、模板或掩膜文件: {img_name}, {template_name}, {mask_name}")
            continue

        t1 = time.time()

        # 4.2 掩膜对齐
        if USE_MASK_ALIGNMENT:
            mask_g = align_mask_to_image(image, template, mask)
        else:
            mask_g = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        t2 = time.time()

        # 4.3 SAHI 检测（核心推理走 GPU）
        result = get_sliced_prediction(
            image,
            detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        )

        t3 = time.time()

        # 4.4 缺陷结构化
        defects = []
        defect_id = 1

        for obj in result.object_prediction_list:
            if obj.mask is None:
                continue

            mask_bool = obj.mask.bool_mask
            area_px = int(np.sum(mask_bool))
            area_mm2 = area_px * (pixel_size ** 2)

            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8) * 255,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue

            best_contour = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(best_contour)
            cx, cy = int(x), int(y)
            radius_px = int(round(r))
            dia_mm = 2 * r * pixel_size

            cx_clamped = int(np.clip(cx, 0, mask_g.shape[1] - 1))
            cy_clamped = int(np.clip(cy, 0, mask_g.shape[0] - 1))

            region_val = str(mask_g[cy_clamped, cx_clamped])
            region_str = region_types.get(region_val, f"未知区域({region_val})")

            # 关键修改：
            # 缺陷中心点如果落在背景、未知或重叠区域，直接跳过。
            # 这样这些检测结果不会进入缺陷总计、NG 判定、可视化图和 Excel。
            if is_ignored_region(region_str, region_val):
                continue

            is_pore = obj.category.name.lower() in ("pore", "porosity")

            status = "OK"
            std = standards.get(region_str, {})

            if is_pore and not std.get("allow_pores", True):
                status = "NG"
            elif (not is_pore) and not std.get("allow_defects", True):
                status = "NG"

            if status == "OK":
                max_area = (
                    std.get("max_pore_area_mm2", float("inf"))
                    if is_pore else std.get("max_defect_area_mm2", float("inf"))
                )
                if area_mm2 > max_area:
                    status = "NG"
            else:
                max_area = None

            if status == "OK":
                obs_r = std.get("observation_diameter_mm", 0) / 2
                max_num = (
                    std.get("max_pores_per_observation", float("inf"))
                    if is_pore else std.get("max_defects_per_observation", float("inf"))
                )
                if max_num > 0 and obs_r > 0:
                    near = count_defects_in_circle(
                        (cx, cy),
                        defects,
                        obs_r,
                        pixel_size,
                        max_area=max_area,
                        is_pore=is_pore,
                    )
                    if near >= max_num:
                        status = "NG"

            defect_info = {
                "Defect ID": defect_id,
                "Region Type": region_str,
                "Region Value": region_val,
                "Diameter (mm)": round(dia_mm, 2),
                "Area (mm²)": round(area_mm2, 2),
                "X Position": cx,
                "Y Position": cy,
                "Radius (px)": radius_px,
                "IsPore": is_pore,
                "Status": status,
                "Contour": best_contour,
            }
            defects.append(defect_info)
            defect_id += 1

        t4 = time.time()

        # 4.5 绘图与总判定
        current_status = "OK"
        for defect in defects:
            if defect["Status"] == "NG":
                current_status = "NG"
                overall_status = "NG"
                break

        if SAVE_VIS_IMAGE:
            vis_image = image.copy()

            for defect in defects:
                contour = defect["Contour"]
                cx, cy = defect["X Position"], defect["Y Position"]
                radius_px = defect["Radius (px)"]
                diameter_mm = defect["Diameter (mm)"]
                area_mm2 = defect["Area (mm²)"]
                defect_id = defect["Defect ID"]
                region_str = defect["Region Type"]

                cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)
                cv2.circle(vis_image, (cx, cy), radius_px, (0, 255, 255), 2)
                cv2.putText(vis_image, f"{defect_id}", (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(vis_image, f"Dia: {diameter_mm:.2f}mm", (cx, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(vis_image, f"Area: {area_mm2:.2f}mm²", (cx, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(vis_image, f"Region: {region_str}", (cx, cy + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            cv2.putText(
                vis_image,
                current_status,
                (vis_image.shape[1] - 300, vis_image.shape[0] - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                6,
                (0, 0, 255) if current_status == "NG" else (0, 255, 0),
                10,
            )

            out_img = os.path.join(out_root, f"{os.path.splitext(img_name)[0]}_{current_status}.png")
            cv2.imwrite(out_img, vis_image)
            print(f"保存结果图像: {out_img}")

        t5 = time.time()

        # 4.6 Excel
        if GENERATE_EXCEL:
            excel_path = os.path.join(data_root, f"{os.path.splitext(img_name)[0]}_report.xlsx")
            wb = Workbook()

            ws_summary = wb.active
            ws_summary.title = "Region Summary"
            ws_summary.append(["Region", "Type", "Defects", "Pores", "NG Count", "Status"])

            region_stats = {}
            for defect in defects:
                region_val = defect["Region Value"]
                region_str = defect["Region Type"]

                if region_val not in region_stats:
                    region_stats[region_val] = {
                        "type": region_str,
                        "defects": 0,
                        "pores": 0,
                        "ng": 0,
                    }

                if defect["IsPore"]:
                    region_stats[region_val]["pores"] += 1
                else:
                    region_stats[region_val]["defects"] += 1

                if defect["Status"] == "NG":
                    region_stats[region_val]["ng"] += 1

            for region, stats in region_stats.items():
                status = "NG" if stats["ng"] > 0 else "OK"
                ws_summary.append([
                    region,
                    stats["type"],
                    stats["defects"],
                    stats["pores"],
                    stats["ng"],
                    status
                ])

            ws_details = wb.create_sheet("Defect Details")
            ws_details.append(["ID", "Region", "Type", "DefectType", "Dia(mm)", "Area(mm²)", "X", "Y", "Status"])

            for defect in defects:
                ws_details.append([
                    defect["Defect ID"],
                    defect["Region Value"],
                    defect["Region Type"],
                    "Pore" if defect["IsPore"] else "Defect",
                    defect["Diameter (mm)"],
                    defect["Area (mm²)"],
                    defect["X Position"],
                    defect["Y Position"],
                    defect["Status"],
                ])

            ws_standards = wb.create_sheet("Standards")
            ws_standards.append([
                "Region Type", "Obs Dia(mm)", "Max Pores", "Max Pore Area(mm²)",
                "Max Defects", "Max Defect Area(mm²)", "Allow Pores", "Allow Defects"
            ])

            all_region_types = set(region_types.values())
            for region_type in all_region_types:
                std = standards.get(region_type, {})
                ws_standards.append([
                    region_type,
                    std.get("observation_diameter_mm", "N/A"),
                    std.get("max_pores_per_observation", "N/A"),
                    std.get("max_pore_area_mm2", "N/A"),
                    std.get("max_defects_per_observation", "N/A"),
                    std.get("max_defect_area_mm2", "N/A"),
                    "是" if std.get("allow_pores", False) else "否",
                    "是" if std.get("allow_defects", False) else "否",
                ])

            for sheet in wb:
                set_excel_style(sheet)

            wb.save(excel_path)
            print(f"生成Excel报告: {excel_path}")

        t6 = time.time()

        print(
            f"[耗时] 读取: {t1 - t0:.2f}s | "
            f"对齐: {t2 - t1:.2f}s | "
            f"检测: {t3 - t2:.2f}s | "
            f"后处理: {t4 - t3:.2f}s | "
            f"绘图: {t5 - t4:.2f}s | "
            f"Excel: {t6 - t5:.2f}s | "
            f"总计: {t6 - t0:.2f}s"
        )

    base_num = "".join(filter(str.isdigit, os.path.basename(IMG_FOLDER)))

    final_img_dir = os.path.join(OUTPUT_ROOT, f"{base_num}_{overall_status}")
    final_data_dir = os.path.join(OUTPUT_ROOT, f"{base_num}_{overall_status}_data")

    if os.path.exists(final_img_dir):
        final_img_dir = os.path.join(OUTPUT_ROOT, f"{base_num}_{overall_status}_{ts}")
    if os.path.exists(final_data_dir):
        final_data_dir = os.path.join(OUTPUT_ROOT, f"{base_num}_{overall_status}_data_{ts}")

    shutil.move(out_root, final_img_dir)

    if os.path.exists(data_root):
        shutil.move(data_root, final_data_dir)

    print(f"\n处理完成 → 图像目录: {final_img_dir}")
    if GENERATE_EXCEL:
        print(f"处理完成 → 数据目录: {final_data_dir}")


if __name__ == "__main__":
    main()