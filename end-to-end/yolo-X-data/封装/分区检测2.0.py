import cv2
import numpy as np
import os, time, json
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
from PIL import Image

# ============ 0. 初始化 SAHI 模型 ============
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov11',
    model_path='best_LR_75.pt',
    confidence_threshold=0.3,
    device='cuda:1',
    image_size=1280
)


# ============ 1. 工具函数 ============
def count_defects_in_circle(center, defects_list, radius_mm,
                            pixel_size_mm_per_pixel, max_area=None, is_pore=False):
    count = 0
    cx, cy = center
    radius_px = radius_mm / pixel_size_mm_per_pixel
    for d in defects_list:
        dist = np.sqrt((d["X Position"] - cx) ** 2 + (d["Y Position"] - cy) ** 2)
        if (dist <= radius_px and
                (max_area is None or d["Area (mm²)"] <= max_area) and
                d["IsPore"] == is_pore):
            count += 1
    return count


def set_excel_style(ws):
    thin = Side(style='thin')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_font = Font(bold=True, color='FFFFFF')
    header_fill = PatternFill(start_color='4F81BD', end_color='4F81BD', fill_type='solid')
    for row in ws.iter_rows():
        for c in row:
            c.border, c.alignment = border, Alignment(horizontal='center', vertical='center')
    for c in ws[1]:
        c.font, c.fill = header_font, header_fill
    for col in ws.columns:
        length = max(len(str(cell.value)) for cell in col) if col else 0
        ws.column_dimensions[get_column_letter(col[0].column)].width = (length + 2) * 1.2


def align_mask_to_image(img, template, mask):
    """
    将模板和掩膜对齐到输入图像
    返回对齐后的掩膜
    """
    # 转换为灰度图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 使用SIFT特征检测器
    sift = cv2.SIFT_create()

    # 检测关键点和描述子
    kp1, des1 = sift.detectAndCompute(img_gray, None)
    kp2, des2 = sift.detectAndCompute(template_gray, None)

    # 如果特征点不足，返回原始掩膜
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        print("特征点不足，使用原始掩膜")
        return cv2.resize(mask, (img.shape[1], img.shape[0]))

    # 创建FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用KNN匹配描述子
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选匹配点，使用Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 如果匹配点不足，返回原始掩膜
    if len(good_matches) < 4:
        print("匹配点不足，使用原始掩膜")
        return cv2.resize(mask, (img.shape[1], img.shape[0]))

    # 提取匹配的关键点坐标
    pts_img = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_template = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算透视变换矩阵
    M, mask_homography = cv2.findHomography(pts_template, pts_img, cv2.RANSAC, 5.0)

    if M is None:
        print("无法计算变换矩阵，使用原始掩膜")
        return cv2.resize(mask, (img.shape[1], img.shape[0]))

    # 对掩膜图像进行透视变换 - 使用最近邻插值保持原始值
    h, w = img_gray.shape
    aligned_mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST)

    return aligned_mask


# ============ 2. 主流程 ============
def main():
    # 1) 输出目录
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join('outputs', ts)
    data_root = os.path.join('outputs', f"{ts}_data")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(data_root, exist_ok=True)

    # 2) 校准 & 标准
    with open('calibration_result.json', 'r', encoding='utf-8') as f:
        pixel_size = json.load(f)["pixel_size_mm_per_pixel"]

    # 读取区域类型映射
    with open('region_types.json', 'r', encoding='utf-8') as f:
        region_types = json.load(f)

    # 读取检测标准
    with open('standard3.json', 'r', encoding='utf-8') as f:
        standards = json.load(f)

    # 添加背景区域标准
    standards["背景"] = {
        "allow_defects": False,
        "allow_pores": False,
        "observation_diameter_mm": 0,
        "max_pores_per_observation": 0,
        "max_pore_area_mm2": 0,
        "max_defects_per_observation": 0,
        "max_defect_area_mm2": 0
    }

    # 3) 文件路径设置
    img_folder = '38048 OK'  # 图像文件夹路径
    template_folder = '11193 NG'  # 模板文件夹路径
    mask_folder = 'VK3-mask'  # 掩膜文件夹路径

    # 获取文件夹下所有的图片文件
    img_files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg'))])
    template_files = sorted([f for f in os.listdir(template_folder) if f.lower().endswith(('.png', '.jpg'))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.lower().endswith('.png')])

    if len(img_files) != len(template_files) or len(img_files) != len(mask_files):
        raise ValueError("图像、模板与掩膜数量不一致！")

    overall_status = "OK"

    # 4) 逐图处理
    for img_name, template_name, mask_name in zip(img_files, template_files, mask_files):
        print(f"处理图像: {img_name}")

        # 读取图像、模板和掩膜
        image = cv2.imread(os.path.join(img_folder, img_name))
        template = cv2.imread(os.path.join(template_folder, template_name))
        mask = cv2.imread(os.path.join(mask_folder, mask_name), cv2.IMREAD_GRAYSCALE)

        if image is None or template is None or mask is None:
            print(f"无法读取图像、模板或掩膜文件: {img_name}, {template_name}, {mask_name}")
            continue

        # 使用特征匹配对齐掩膜
        mask_g = align_mask_to_image(image, template, mask)

        # ---- 4.1 SAHI 检测 ----
        result = get_sliced_prediction(
            image, detection_model,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.3, overlap_width_ratio=0.3
        )

        # ---- 4.2 缺陷转 dict ----
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
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            (x, y), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
            cx, cy = int(x), int(y)
            dia_mm = 2 * r * pixel_size

            # 获取中心点位置在掩膜中的值
            region_val = str(mask_g[cy, cx])
            region_str = region_types.get(region_val, f"未知区域({region_val})")
            is_pore = obj.category.name.lower() in ("pore", "porosity")

            # 标准判定
            status = "OK"
            std = standards.get(region_str, {})

            # 检查是否允许此类缺陷
            if is_pore and not std.get("allow_pores", True):
                status = "NG"
            elif not is_pore and not std.get("allow_defects", True):
                status = "NG"

            # 检查面积限制
            if status == "OK":
                max_area = std.get("max_pore_area_mm2", float('inf')) if is_pore else std.get("max_defect_area_mm2",
                                                                                              float('inf'))
                if area_mm2 > max_area:
                    status = "NG"

            # 检查数量限制
            if status == "OK":
                obs_r = std.get("observation_diameter_mm", 0) / 2
                max_num = std.get("max_pores_per_observation", float('inf')) if is_pore else std.get(
                    "max_defects_per_observation", float('inf'))
                if max_num > 0 and obs_r > 0:
                    near = count_defects_in_circle((cx, cy), defects, obs_r, pixel_size, max_area, is_pore)
                    if near >= max_num:
                        status = "NG"

            # 保存缺陷信息
            defect_info = {
                "Defect ID": defect_id,
                "Region Type": region_str,
                "Region Value": region_val,
                "Diameter (mm)": round(dia_mm, 2),
                "Area (mm²)": round(area_mm2, 2),
                "X Position": cx,
                "Y Position": cy,
                "IsPore": is_pore,
                "Status": status,
                "Contour": max(contours, key=cv2.contourArea)
            }
            defects.append(defect_info)
            defect_id += 1

        # ---- 4.3 整图状态 & 绘图 ----
        for defect in defects:
            contour = defect["Contour"]
            cx, cy = defect["X Position"], defect["Y Position"]
            diameter_mm = defect["Diameter (mm)"]
            area_mm2 = defect["Area (mm²)"]
            defect_id = defect["Defect ID"]
            region_str = defect["Region Type"]

            # 绘制轮廓
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            # 绘制最小外接圆
            cv2.circle(image, (cx, cy), int(diameter_mm / 2), (0, 255, 255), 2)
            # 标注缺陷编号
            cv2.putText(image, f"{defect_id}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 标注直径
            cv2.putText(image, f"Dia: {diameter_mm:.2f}mm", (cx, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            # 标注面积
            cv2.putText(image, f"Area: {area_mm2:.2f}mm²", (cx, cy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            # 标注区域类型
            cv2.putText(image, f"Region: {region_str}", (cx, cy + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 整体状态判定
        current_status = "OK"
        for defect in defects:
            if defect["Status"] == "NG":
                current_status = overall_status = "NG"
                break

        # 添加判定结果标注
        cv2.putText(image, current_status, (image.shape[1] - 300, image.shape[0] - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 6,
                    (0, 0, 255) if current_status == "NG" else (0, 255, 0), 10)

        # 保存结果图像
        out_img = os.path.join(out_root, f"{os.path.splitext(img_name)[0]}_{current_status}.png")
        cv2.imwrite(out_img, image)
        print(f"保存结果图像: {out_img}")

        # ---- 4.4 生成 Excel ----
        excel_path = os.path.join(data_root, f"{os.path.splitext(img_name)[0]}_report.xlsx")
        wb = Workbook()
        # 1. 区域汇总表
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
                    "ng": 0
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

        # 2. 缺陷详情表
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
                defect["Status"]
            ])

        # 3. 检测标准表
        ws_standards = wb.create_sheet("Standards")
        ws_standards.append(["Region Type", "Obs Dia(mm)", "Max Pores", "Max Pore Area(mm²)",
                             "Max Defects", "Max Defect Area(mm²)", "Allow Pores", "Allow Defects"])

        # 获取所有区域类型（包括JSON中定义的和实际检测到的）
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
                "是" if std.get("allow_defects", False) else "否"
            ])

        # 应用样式
        for sheet in wb:
            set_excel_style(sheet)
        wb.save(excel_path)
        print(f"生成Excel报告: {excel_path}")

    # 5) 批量完成后重命名输出文件夹
    base_num = ''.join(filter(str.isdigit, os.path.basename(img_folder)))
    final_dir = os.path.join('outputs', f"{base_num}_{overall_status}")
    os.rename(out_root, final_dir)
    print(f"\n处理完成 → {final_dir}")


if __name__ == "__main__":
    main()