import cv2
import numpy as np
import os
import json
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter

# 1. 初始化YOLO检测模型
# detection_model = AutoDetectionModel.from_pretrained(
#     model_type='yolov11',
#     model_path='D:\zxy\pythonProject\d100-3.pt',
#     confidence_threshold=0.3,
#     device='cuda:0',
#     image_size=1280
# )


def set_excel_style(worksheet):
    """设置Excel表格样式"""
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'),
                         top=Side(style='thin'), bottom=Side(style='thin'))
    header_font = Font(bold=True, color='FFFFFF')
    header_fill = PatternFill(start_color='4F81BD', end_color='4F81BD', fill_type='solid')

    for row in worksheet.iter_rows():
        for cell in row:
            cell.border = thin_border
            cell.alignment = Alignment(horizontal='center', vertical='center')

    for cell in worksheet[1]:
        cell.font = header_font
        cell.fill = header_fill

    for col in worksheet.columns:
        max_length = max(len(str(cell.value)) for cell in col) if any(cell.value for cell in col) else 0
        worksheet.column_dimensions[get_column_letter(col[0].column)].width = (max_length + 2) * 1.2



def detect_with_sahi(image_path, output_path, pixel_size_mm_per_pixel=1.0):
    """使用SAHI进行YOLOv11检测，并计算每个缺陷的面积"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return []

    # --- 切片推理 ---
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
        verbose=0
    )

    # --- 可视化检测结果 ---
    vis_result = visualize_object_predictions(
        image,
        object_prediction_list=result.object_prediction_list,
        output_dir=None,
        rect_th=0,
        text_th=0,
        text_size=0,
        hide_labels=True,
        hide_conf=False
    )

    # 处理可视化结果为 ndarray
    if isinstance(vis_result, dict):
        vis_img = vis_result['image']
    elif isinstance(vis_result, Image.Image):
        vis_img = np.array(vis_result)
    else:
        vis_img = vis_result
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    defects_list = []
    defect_counter = 1

    # --- 遍历检测结果，计算面积 ---
    for obj in result.object_prediction_list:
        if obj.mask is None:
            continue
        mask = obj.mask.bool_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        area_px = sum(cv2.contourArea(c) for c in contours)
        if area_px < 5:  # 忽略小面积噪声
            continue

        area_mm2 = area_px * (pixel_size_mm_per_pixel ** 2)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        center_x, center_y = x + w // 2, y + h // 2
        equivalent_diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)

        defects_list.append({
            "Defect ID": defect_counter,
            "Area (mm²)": round(area_mm2, 4),
            "Equivalent Diameter (mm)": round(equivalent_diameter_mm, 4),
            "X Position": center_x,
            "Y Position": center_y,
            "Width (pixels)": w,
            "Height (pixels)": h,
            "Category": obj.category.name,
            "Confidence": round(obj.score.value, 3)
        })

        # --- 绘制标注信息 ---
        cv2.drawContours(vis_img, [max(contours, key=cv2.contourArea)], -1, (0, 255, 0), 2)
        cv2.putText(vis_img, f"#{defect_counter} {area_mm2:.1f}mm2",
                    (x, max(y - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        defect_counter += 1

    # --- 保存结果图像 ---
    cv2.imwrite(output_path, vis_img)
    print(f"检测完成: {os.path.basename(image_path)} -> {output_path}")
    return defects_list

def generate_excel_report(defects_list, output_excel_path, image_name):
    """为每张图生成Excel报告"""
    wb = Workbook()

    # 汇总表
    ws_sum = wb.active
    ws_sum.title = "Defect Summary"
    ws_sum.append(["Image Name", "Total Defects", "Total Area (mm²)", "Max Area (mm²)", "Min Area (mm²)"])

    total_area = sum(d["Area (mm²)"] for d in defects_list) if defects_list else 0
    max_area = max(d["Area (mm²)"] for d in defects_list) if defects_list else 0
    min_area = min(d["Area (mm²)"] for d in defects_list) if defects_list else 0

    ws_sum.append([image_name, len(defects_list), total_area, max_area, min_area])

    # 详情表
    ws_detail = wb.create_sheet("Defect Details")
    ws_detail.append(["Defect ID", "Area (mm²)", "Equivalent Diameter (mm)", "X", "Y", "Width", "Height", "Category", "Confidence"])
    for d in defects_list:
        ws_detail.append([
            d["Defect ID"], d["Area (mm²)"], d["Equivalent Diameter (mm)"],
            d["X Position"], d["Y Position"], d["Width (pixels)"], d["Height (pixels)"],
            d["Category"], d["Confidence"]
        ])

    for sheet in wb:
        set_excel_style(sheet)

    wb.save(output_excel_path)
    print(f"Excel报告已保存: {output_excel_path}")

def generate_summary_report(all_results, output_folder):
    """报告匯總"""
    wb = Workbook()
    ws = wb.active
    ws.title = "Overall Summary"
    ws.append(["Image Name", "Total Defects", "Total Area (mm²)", "Max Area (mm²)", "Min Area (mm²)", "Avg Area (mm²)"])

    for r in all_results:
        defects = r["defects"]
        image_name = os.path.basename(r["image_path"])
        total_area = sum(d["Area (mm²)"] for d in defects) if defects else 0
        max_area = max(d["Area (mm²)"] for d in defects) if defects else 0
        min_area = min(d["Area (mm²)"] for d in defects) if defects else 0
        avg_area = total_area / len(defects) if defects else 0

        ws.append([image_name, len(defects), total_area, max_area, min_area, round(avg_area, 3)])

    set_excel_style(ws)
    summary_path = os.path.join(output_folder, "00_Overall_Summary.xlsx")
    wb.save(summary_path)
    print(f"\n汇总报告已生成: {summary_path}")

if __name__ == "__main__":
    # 加载标定文件（像素→mm换算）
    if os.path.exists("calibration_result.json"):
        with open("calibration_result.json", "r", encoding="utf-8") as f:
            pixel_size_mm_per_pixel = json.load(f)["pixel_size_mm_per_pixel"]
    else:
        pixel_size_mm_per_pixel = 1.0  # 默认值（无标定文件）

    # 初始化YOLOv11模型
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov11',
        model_path=r'D:\zxy\pythonProject\best_LR_75.pt',
        confidence_threshold=0.3,
        device='cuda:0',
        image_size=1280
    )

    input_folder = r'D:\zxy\pythonProject\predictIMG'
    output_folder = r'D:\zxy\pythonProject\predictIMG\outputd100-3'
    os.makedirs(output_folder, exist_ok=True)

    all_results = []

    for img_file in os.listdir(input_folder):
        if not img_file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            continue
        img_path = os.path.join(input_folder, img_file)
        out_img_path = os.path.join(output_folder, f"result_{img_file}")
        out_excel_path = os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_defect.xlsx")

        defects = detect_with_sahi(img_path, out_img_path, pixel_size_mm_per_pixel)
        if defects:
            generate_excel_report(defects, out_excel_path, img_file)
        all_results.append({"image_path": img_path, "defects": defects})

    generate_summary_report(all_results, output_folder)
    print("\n所有图片检测与报表生成完成！")