import cv2
import numpy as np
import os
import time
import json
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

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
        ws.column_dimensions[get_column_letter(col[0].column)].width = min((length + 2) * 1.2, 40)



def preprocess_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)



def safe_imwrite(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)



def draw_text_block(img, lines, x=20, y=30, line_h=28, color=(255, 255, 255), bg=(0, 0, 0)):
    out = img.copy()
    pad = 8
    max_w = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(str(line), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        max_w = max(max_w, w)
    box_h = len(lines) * line_h + pad * 2
    box_w = max_w + pad * 2
    cv2.rectangle(out, (x - pad, y - 22 - pad), (x - pad + box_w, y - 22 - pad + box_h), bg, -1)
    alpha = 0.5
    img[:] = cv2.addWeighted(out, alpha, img, 1 - alpha, 0)
    yy = y
    for line in lines:
        cv2.putText(img, str(line), (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        yy += line_h
    return img



def colorize_mask(mask_gray):
    unique_vals = [v for v in np.unique(mask_gray) if v != 0]
    color = np.zeros((mask_gray.shape[0], mask_gray.shape[1], 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    lut = {0: (0, 0, 0)}
    for v in unique_vals:
        lut[int(v)] = tuple(int(x) for x in rng.integers(50, 255, size=3))
    for v, c in lut.items():
        color[mask_gray == v] = c
    return color



def overlay_mask_on_image(image_bgr, mask_gray, alpha=0.35):
    color_mask = colorize_mask(mask_gray)
    overlay = cv2.addWeighted(image_bgr, 1.0, color_mask, alpha, 0)
    contours, _ = cv2.findContours((mask_gray > 0).astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay



def make_side_by_side(images, target_height=720):
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_height / max(h, 1)
        new_w = max(1, int(w * scale))
        resized.append(cv2.resize(img, (new_w, target_height)))
    return cv2.hconcat(resized)



def build_alignment_debug_visuals(template, image, mask, aligned_mask, kp_template, kp_img,
                                  good_matches, inlier_matches, projected_corners, metrics):
    # 1) 关键点匹配图
    show_good = good_matches[:80]
    match_vis = cv2.drawMatches(
        template, kp_template, image, kp_img, show_good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0)
    )
    draw_text_block(match_vis, [
        f"raw_matches: {metrics['raw_matches']}",
        f"good_matches: {metrics['good_matches']}",
        f"inliers: {metrics['inliers']}",
        f"inlier_ratio: {metrics['inlier_ratio']:.3f}",
        f"reproj_error(px): {metrics['reprojection_error']:.3f}",
        f"status: {metrics['status']}"
    ])

    # 2) 模板投影图
    proj_vis = image.copy()
    if projected_corners is not None:
        cv2.polylines(proj_vis, [np.int32(projected_corners)], True, (0, 0, 255), 4)
        for i, p in enumerate(np.int32(projected_corners).reshape(-1, 2), start=1):
            cv2.circle(proj_vis, tuple(p), 8, (255, 255, 0), -1)
            cv2.putText(proj_vis, str(i), tuple(p + np.array([10, -10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    draw_text_block(proj_vis, [
        "Template projected to target image",
        f"status: {metrics['status']}"
    ])

    # 3) 掩膜叠加图
    overlay_vis = overlay_mask_on_image(image, aligned_mask)
    draw_text_block(overlay_vis, [
        "Aligned mask overlay",
        f"mask_nonzero: {int(np.count_nonzero(aligned_mask))}"
    ])

    # 4) 模板原图 + 原始掩膜叠加
    if len(mask.shape) == 2:
        template_overlay = overlay_mask_on_image(template, mask)
    else:
        template_overlay = template.copy()
    draw_text_block(template_overlay, ["Template + original mask"])

    return match_vis, proj_vis, overlay_vis, template_overlay



def save_alignment_debug(debug_dir, base_name, template, image, mask, aligned_mask, metrics,
                         kp_template, kp_img, good_matches, inlier_matches, projected_corners):
    ensure_dir(debug_dir)

    match_vis, proj_vis, overlay_vis, template_overlay = build_alignment_debug_visuals(
        template, image, mask, aligned_mask, kp_template, kp_img,
        good_matches, inlier_matches, projected_corners, metrics
    )

    safe_imwrite(os.path.join(debug_dir, f"{base_name}_01_template_mask.png"), template_overlay)
    safe_imwrite(os.path.join(debug_dir, f"{base_name}_02_matches.png"), match_vis)
    safe_imwrite(os.path.join(debug_dir, f"{base_name}_03_projection.png"), proj_vis)
    safe_imwrite(os.path.join(debug_dir, f"{base_name}_04_overlay.png"), overlay_vis)

    panel = make_side_by_side([template_overlay, match_vis, proj_vis, overlay_vis], target_height=700)
    safe_imwrite(os.path.join(debug_dir, f"{base_name}_DEBUG_ALL.png"), panel)

    with open(os.path.join(debug_dir, f"{base_name}_align_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)



def compute_reprojection_error(pts_template, pts_img, H, inlier_mask):
    if H is None or inlier_mask is None:
        return float('inf')
    inlier_mask = inlier_mask.ravel().astype(bool)
    if not np.any(inlier_mask):
        return float('inf')

    pts_t = pts_template[inlier_mask].reshape(-1, 1, 2)
    pts_i = pts_img[inlier_mask].reshape(-1, 1, 2)
    pts_proj = cv2.perspectiveTransform(pts_t, H)
    err = np.linalg.norm(pts_proj.reshape(-1, 2) - pts_i.reshape(-1, 2), axis=1)
    return float(np.mean(err)) if len(err) > 0 else float('inf')



def align_mask_to_image(img, template, mask, ratio_thresh=0.72, ransac_thresh=4.0):
    """
    将模板和掩膜对齐到输入图像。
    返回: aligned_mask, H, metrics, debug_pack
    """
    img_gray = preprocess_gray(img)
    template_gray = preprocess_gray(template)

    # SIFT 特征
    sift = cv2.SIFT_create(nfeatures=4000)
    kp_img, des_img = sift.detectAndCompute(img_gray, None)
    kp_template, des_template = sift.detectAndCompute(template_gray, None)

    metrics = {
        "kp_img": 0 if kp_img is None else len(kp_img),
        "kp_template": 0 if kp_template is None else len(kp_template),
        "raw_matches": 0,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "reprojection_error": float('inf'),
        "status": "INIT",
        "used_fallback_resize": False,
        "homography": None
    }

    fallback_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    if des_img is None or des_template is None or len(des_img) < 4 or len(des_template) < 4:
        metrics["status"] = "特征点不足"
        metrics["used_fallback_resize"] = True
        return fallback_mask, None, metrics, {
            "kp_template": kp_template or [],
            "kp_img": kp_img or [],
            "good_matches": [],
            "inlier_matches": [],
            "projected_corners": None
        }

    # FLANN + ratio test
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=80)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(des_template, des_img, k=2)
    metrics["raw_matches"] = len(knn_matches)

    good_matches = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    metrics["good_matches"] = len(good_matches)

    if len(good_matches) < 4:
        metrics["status"] = "有效匹配点不足"
        metrics["used_fallback_resize"] = True
        return fallback_mask, None, metrics, {
            "kp_template": kp_template,
            "kp_img": kp_img,
            "good_matches": good_matches,
            "inlier_matches": [],
            "projected_corners": None
        }

    pts_template = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_img = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, inlier_mask = cv2.findHomography(pts_template, pts_img, cv2.RANSAC, ransac_thresh)

    if H is None:
        metrics["status"] = "单应矩阵求解失败"
        metrics["used_fallback_resize"] = True
        return fallback_mask, None, metrics, {
            "kp_template": kp_template,
            "kp_img": kp_img,
            "good_matches": good_matches,
            "inlier_matches": [],
            "projected_corners": None
        }

    inlier_mask = inlier_mask.ravel().astype(bool)
    inlier_matches = [m for i, m in enumerate(good_matches) if inlier_mask[i]]
    metrics["inliers"] = len(inlier_matches)
    metrics["inlier_ratio"] = len(inlier_matches) / max(len(good_matches), 1)
    metrics["reprojection_error"] = compute_reprojection_error(pts_template, pts_img, H, inlier_mask)
    metrics["homography"] = H.tolist()

    h, w = img_gray.shape
    aligned_mask = cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST)

    # 投影模板四角，便于可视化模板是否落在正确位置
    th, tw = template.shape[:2]
    corners = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]]).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(corners, H)

    # 质量门限：避免“看起来算出来了，但其实已经漂了”
    if metrics["inliers"] < 8:
        metrics["status"] = "匹配不稳定(内点太少)"
    elif metrics["inlier_ratio"] < 0.35:
        metrics["status"] = "匹配不稳定(内点比例低)"
    elif metrics["reprojection_error"] > 6.0:
        metrics["status"] = "匹配不稳定(重投影误差大)"
    else:
        metrics["status"] = "OK"

    return aligned_mask, H, metrics, {
        "kp_template": kp_template,
        "kp_img": kp_img,
        "good_matches": good_matches,
        "inlier_matches": inlier_matches,
        "projected_corners": projected_corners
    }


# ============ 2. 主流程 ============
def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join('outputs', ts)
    data_root = os.path.join('outputs', f"{ts}_data")
    debug_root = os.path.join('outputs', f"{ts}_debug")
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(debug_root, exist_ok=True)

    with open('calibration_result.json', 'r', encoding='utf-8') as f:
        pixel_size = json.load(f)["pixel_size_mm_per_pixel"]

    with open('region_types.json', 'r', encoding='utf-8') as f:
        region_types = json.load(f)

    with open('standard31.json', 'r', encoding='utf-8') as f:
        standards = json.load(f)

    standards["背景"] = {
        "allow_defects": False,
        "allow_pores": False,
        "observation_diameter_mm": 0,
        "max_pores_per_observation": 0,
        "max_pore_area_mm2": 0,
        "max_defects_per_observation": 0,
        "max_defect_area_mm2": 0
    }

    img_folder = '38048 OK'
    template_folder = '11193 NG'
    mask_folder = 'VK3-mask'

    img_files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    template_files = sorted([f for f in os.listdir(template_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.lower().endswith('.png')])

    if len(img_files) != len(template_files) or len(img_files) != len(mask_files):
        raise ValueError("图像、模板与掩膜数量不一致！")

    overall_status = "OK"

    for img_name, template_name, mask_name in zip(img_files, template_files, mask_files):
        print(f"处理图像: {img_name}")
        base_name = os.path.splitext(img_name)[0]

        image = cv2.imread(os.path.join(img_folder, img_name))
        template = cv2.imread(os.path.join(template_folder, template_name))
        mask = cv2.imread(os.path.join(mask_folder, mask_name), cv2.IMREAD_GRAYSCALE)

        if image is None or template is None or mask is None:
            print(f"无法读取图像、模板或掩膜文件: {img_name}, {template_name}, {mask_name}")
            continue

        mask_g, H, align_metrics, dbg = align_mask_to_image(image, template, mask)

        save_alignment_debug(
            debug_dir=debug_root,
            base_name=base_name,
            template=template,
            image=image,
            mask=mask,
            aligned_mask=mask_g,
            metrics=align_metrics,
            kp_template=dbg["kp_template"],
            kp_img=dbg["kp_img"],
            good_matches=dbg["good_matches"],
            inlier_matches=dbg["inlier_matches"],
            projected_corners=dbg["projected_corners"]
        )

        # ---- 4.1 SAHI 检测 ----
        result = get_sliced_prediction(
            image, detection_model,
            slice_height=640, slice_width=640,
            overlap_height_ratio=0.3, overlap_width_ratio=0.3
        )

        # ---- 4.2 缺陷转 dict ----
        defects = []
        defect_id = 1
        image_h, image_w = image.shape[:2]

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

            contour = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(contour)
            cx, cy = int(round(x)), int(round(y))
            cx = int(np.clip(cx, 0, image_w - 1))
            cy = int(np.clip(cy, 0, image_h - 1))
            dia_mm = 2 * r * pixel_size

            region_val = str(mask_g[cy, cx])
            region_str = region_types.get(region_val, f"未知区域({region_val})")
            is_pore = obj.category.name.lower() in ("pore", "porosity")

            status = "OK"
            std = standards.get(region_str, {})

            if is_pore and not std.get("allow_pores", True):
                status = "NG"
            elif not is_pore and not std.get("allow_defects", True):
                status = "NG"

            if status == "OK":
                max_area = std.get("max_pore_area_mm2", float('inf')) if is_pore else std.get("max_defect_area_mm2", float('inf'))
                if area_mm2 > max_area:
                    status = "NG"

            if status == "OK":
                obs_r = std.get("observation_diameter_mm", 0) / 2
                max_num = std.get("max_pores_per_observation", float('inf')) if is_pore else std.get("max_defects_per_observation", float('inf'))
                if max_num > 0 and obs_r > 0:
                    near = count_defects_in_circle((cx, cy), defects, obs_r, pixel_size, max_area, is_pore)
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
                "Radius (px)": float(r),
                "IsPore": is_pore,
                "Status": status,
                "Contour": contour
            }
            defects.append(defect_info)
            defect_id += 1

        # ---- 4.3 整图状态 & 绘图 ----
        vis = image.copy()
        for defect in defects:
            contour = defect["Contour"]
            cx, cy = defect["X Position"], defect["Y Position"]
            diameter_mm = defect["Diameter (mm)"]
            area_mm2 = defect["Area (mm²)"]
            defect_id = defect["Defect ID"]
            region_str = defect["Region Type"]
            radius_px = int(round(defect["Radius (px)"]))  # 修复原代码 mm/px 单位混用问题

            color = (0, 0, 255) if defect["Status"] == "NG" else (0, 255, 0)
            cv2.drawContours(vis, [contour], -1, color, 2)
            cv2.circle(vis, (cx, cy), max(radius_px, 1), (0, 255, 255), 2)
            cv2.putText(vis, f"{defect_id}", (cx, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(vis, f"Dia: {diameter_mm:.2f}mm", (cx, cy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(vis, f"Area: {area_mm2:.2f}mm2", (cx, cy + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(vis, f"Region: {region_str}", (cx, cy + 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        current_status = "OK"
        for defect in defects:
            if defect["Status"] == "NG":
                current_status = overall_status = "NG"
                break

        overlay = overlay_mask_on_image(vis, mask_g, alpha=0.25)
        draw_text_block(overlay, [
            f"Image status: {current_status}",
            f"Align status: {align_metrics['status']}",
            f"good/inliers: {align_metrics['good_matches']}/{align_metrics['inliers']}",
            f"inlier_ratio: {align_metrics['inlier_ratio']:.3f}",
            f"reproj_err(px): {align_metrics['reprojection_error']:.3f}"
        ], x=25, y=40)

        cv2.putText(overlay, current_status, (max(30, overlay.shape[1] - 320), max(80, overlay.shape[0] - 80)),
                    cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255) if current_status == "NG" else (0, 255, 0), 6)

        out_img = os.path.join(out_root, f"{base_name}_{current_status}.png")
        cv2.imwrite(out_img, overlay)
        print(f"保存结果图像: {out_img}")

        # ---- 4.4 生成 Excel ----
        excel_path = os.path.join(data_root, f"{base_name}_report.xlsx")
        wb = Workbook()

        ws_summary = wb.active
        ws_summary.title = "Region Summary"
        ws_summary.append(["Region", "Type", "Defects", "Pores", "NG Count", "Status"])

        region_stats = {}
        for defect in defects:
            region_val = defect["Region Value"]
            region_str = defect["Region Type"]
            if region_val not in region_stats:
                region_stats[region_val] = {"type": region_str, "defects": 0, "pores": 0, "ng": 0}
            if defect["IsPore"]:
                region_stats[region_val]["pores"] += 1
            else:
                region_stats[region_val]["defects"] += 1
            if defect["Status"] == "NG":
                region_stats[region_val]["ng"] += 1

        for region, stats in region_stats.items():
            status = "NG" if stats["ng"] > 0 else "OK"
            ws_summary.append([region, stats["type"], stats["defects"], stats["pores"], stats["ng"], status])

        ws_details = wb.create_sheet("Defect Details")
        ws_details.append(["ID", "Region", "Type", "DefectType", "Dia(mm)", "Area(mm²)", "X", "Y", "Status"])
        for defect in defects:
            ws_details.append([
                defect["Defect ID"], defect["Region Value"], defect["Region Type"],
                "Pore" if defect["IsPore"] else "Defect",
                defect["Diameter (mm)"], defect["Area (mm²)"],
                defect["X Position"], defect["Y Position"], defect["Status"]
            ])

        ws_standards = wb.create_sheet("Standards")
        ws_standards.append(["Region Type", "Obs Dia(mm)", "Max Pores", "Max Pore Area(mm²)",
                             "Max Defects", "Max Defect Area(mm²)", "Allow Pores", "Allow Defects"])
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

        ws_align = wb.create_sheet("Alignment Metrics")
        ws_align.append(["Metric", "Value"])
        for k in ["kp_img", "kp_template", "raw_matches", "good_matches", "inliers",
                  "inlier_ratio", "reprojection_error", "status", "used_fallback_resize"]:
            ws_align.append([k, align_metrics[k]])

        for sheet in wb:
            set_excel_style(sheet)
        wb.save(excel_path)
        print(f"生成Excel报告: {excel_path}")

    base_num = ''.join(filter(str.isdigit, os.path.basename(img_folder)))
    final_dir = os.path.join('outputs', f"{base_num}_{overall_status}")
    final_debug_dir = os.path.join('outputs', f"{base_num}_{overall_status}_debug")

    if os.path.exists(final_dir):
        final_dir = os.path.join('outputs', f"{base_num}_{overall_status}_{ts}")
    if os.path.exists(final_debug_dir):
        final_debug_dir = os.path.join('outputs', f"{base_num}_{overall_status}_debug_{ts}")

    os.rename(out_root, final_dir)
    os.rename(debug_root, final_debug_dir)
    print(f"\n处理完成 → {final_dir}")
    print(f"模板匹配可视化 → {final_debug_dir}")


if __name__ == "__main__":
    main()
