import cv2
import numpy as np
import os
import json
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
import pandas as pd
from datetime import datetime


# 从JSON文件加载校准参数
def load_calibration_config(config_path='calibration_config.json'):
    """
    从JSON文件加载校准参数

    参数:
        config_path: JSON配置文件路径

    返回:
        calibration_data: 校准参数字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            calibration_data = json.load(f)

        # 验证必要的参数是否存在
        required_keys = ['image_height_pixels', 'calibration_length_mm', 'pixel_size_mm_per_pixel']
        for key in required_keys:
            if key not in calibration_data:
                raise ValueError(f"缺少必要的校准参数: {key}")

        print(f"成功从 {config_path} 加载校准配置文件")
        return calibration_data

    except FileNotFoundError:
        print(f"警告: 未找到校准配置文件 {config_path}，使用默认参数")
        return {
            "image_height_pixels": 255,
            "calibration_length_mm": 45.0,
            "pixel_size_mm_per_pixel": 0.046778
        }
    except json.JSONDecodeError:
        print(f"错误: 校准配置文件 {config_path} 格式不正确")
        raise
    except Exception as e:
        print(f"加载校准配置时出错: {e}")
        raise


# 加载校准参数
CALIBRATION_DATA = load_calibration_config()

# 外观评级标准（根据腐蚀点总面积）
APPEARANCE_GRADE_STANDARD = {
    'A': (0, 0),  # 无缺陷
    'B': (0, 0.1),  # 0.5×0.1
    'C': (0.1, 0.25),  # 0.1×0.2/3
    'D': (0.25, 0.5),  # 0.25×0.4/3
    'E': (0.5, 1.0),  # 0.50×0.1
    'F': (1.0, 2.5),  # 1.0×0.2/3
    'G': (2.5, 5.0),  # 2.5×0.5/0
    'H': (5.0, 10.0),  # 3.0×0.6/0
    'I': (10.0, 25.0),  # 10×0.6/3
    'J': (25.0, 50.0),  # 25×0.6/0
    'K': (50.0, float('inf'))  # 30×A
}

# 1. 初始化YOLO检测模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov11',
    model_path='best.pt',
    confidence_threshold=0.3,
    device='cuda:0',
)


def calculate_appearance_grade(total_area_mm2):
    """
    根据腐蚀点总面积计算外观评级A-K

    参数:
        total_area_mm2: 腐蚀点总面积(mm²)

    返回:
        grade: 评级字母(A-K)
    """
    for grade, (min_area, max_area) in APPEARANCE_GRADE_STANDARD.items():
        if min_area <= total_area_mm2 < max_area:
            return grade
    return 'K'  # 如果超过所有范围，返回K级


def calculate_corrosion_area(bbox, pixel_size_mm):
    """
    计算腐蚀区域的面积
    """
    # 计算像素尺寸
    width_pixels = bbox[2] - bbox[0]
    height_pixels = bbox[3] - bbox[1]

    # 转换为实际尺寸
    width_mm = width_pixels * pixel_size_mm
    height_mm = height_pixels * pixel_size_mm

    # 计算面积（假设为矩形）
    area_mm2 = width_mm * height_mm

    return area_mm2, width_mm, height_mm


def detect_with_sahi(image_path, output_path):
    """
    使用SAHI进行切片推理并计算腐蚀面积
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None, None

    # 使用SAHI的切片推理
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
        verbose=1
    )

    # 计算每个检测到的腐蚀点的面积
    area_info = []
    for i, pred in enumerate(result.object_prediction_list):
        bbox = pred.bbox.to_xyxy()
        area_mm2, width_mm, height_mm = calculate_corrosion_area(
            bbox, CALIBRATION_DATA["pixel_size_mm_per_pixel"]
        )

        area_info.append({
            'id': i + 1,
            'category': pred.category.name,
            'confidence': float(pred.score.value),
            'bbox_pixels': bbox,
            'width_pixels': bbox[2] - bbox[0],
            'height_pixels': bbox[3] - bbox[1],
            'width_mm': width_mm,
            'height_mm': height_mm,
            'area_mm2': area_mm2
        })

    # 在图像上绘制面积信息
    output_image = image.copy()
    for info in area_info:
        xmin, ymin, xmax, ymax = map(int, info['bbox_pixels'])

        # 绘制边界框
        cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 添加面积文本
        text = f"{info['area_mm2']:.2f}mm2"
        cv2.putText(output_image, text, (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存结果图像
    cv2.imwrite(output_path, output_image)
    print(f"结果已保存到: {output_path}")

    return result, area_info


def check_excel_engines():
    """
    检查可用的Excel引擎
    """
    available_engines = []

    # 检查xlsxwriter
    try:
        import xlsxwriter
        available_engines.append('xlsxwriter')
    except ImportError:
        pass

    # 检查openpyxl
    try:
        import openpyxl
        available_engines.append('openpyxl')
    except ImportError:
        pass

    print(f"可用的Excel引擎: {available_engines}")
    return available_engines


def auto_adjust_column_widths(excel_file):
    """
    自动调整Excel文件的列宽
    """
    try:
        from openpyxl import load_workbook
        wb = load_workbook(excel_file)

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if cell.value and len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 30)
                ws.column_dimensions[column_letter].width = adjusted_width

        wb.save(excel_file)
        print(f"已自动调整 {excel_file} 的列宽")

    except ImportError:
        print("openpyxl未安装，跳过自动调整列宽")
    except Exception as e:
        print(f"调整列宽时出错: {e}")


def create_excel_report(all_reports, output_folder):
    """
    创建Excel格式的报告，强制使用.xlsx格式
    """
    # 创建详细数据表格
    detailed_data = []
    summary_data = []

    # 计算所有图片的总腐蚀面积
    total_all_area = sum([report['total_area_mm2'] for report in all_reports])

    # 计算外观评级
    appearance_grade = calculate_appearance_grade(total_all_area)
    print(f"总腐蚀面积: {total_all_area:.2f}mm², 外观评级: {appearance_grade}")

    for report in all_reports:
        image_name = report['image_name']
        total_points = report['total_corrosion_points']
        total_area = report['total_area_mm2']

        # 汇总数据
        summary_data.append({
            '图片名称': image_name,
            '腐蚀点数量': total_points,
            '总面积(mm²)': total_area,
            '平均面积(mm²)': total_area / total_points if total_points > 0 else 0,
            '最大面积(mm²)': max([d['area_mm2'] for d in report['corrosion_details']]) if total_points > 0 else 0,
            '最小面积(mm²)': min([d['area_mm2'] for d in report['corrosion_details']]) if total_points > 0 else 0
        })

        # 详细数据
        for detail in report['corrosion_details']:
            detailed_data.append({
                '图片名称': image_name,
                '腐蚀点ID': detail['id'],
                '类别': detail['category'],
                '置信度': detail['confidence'],
                'X_min(像素)': detail['bbox_pixels'][0],
                'Y_min(像素)': detail['bbox_pixels'][1],
                'X_max(像素)': detail['bbox_pixels'][2],
                'Y_max(像素)': detail['bbox_pixels'][3],
                '宽度(像素)': detail['width_pixels'],
                '高度(像素)': detail['height_pixels'],
                '宽度(mm)': detail['width_mm'],
                '高度(mm)': detail['height_mm'],
                '面积(mm²)': detail['area_mm2']
            })

    # 创建Excel文件，强制使用.xlsx后缀
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = os.path.join(output_folder, f'corrosion_report_{timestamp}.xlsx')

    # 检查可用引擎
    available_engines = check_excel_engines()

    # 优先使用xlsxwriter，其次openpyxl
    if 'xlsxwriter' in available_engines:
        engine = 'xlsxwriter'
        print("使用xlsxwriter引擎生成.xlsx文件")
    elif 'openpyxl' in available_engines:
        engine = 'openpyxl'
        print("使用openpyxl引擎生成.xlsx文件")
    else:
        print("没有找到.xlsx引擎，将生成CSV文件")
        return create_csv_report(all_reports, output_folder)

    try:
        with pd.ExcelWriter(excel_file, engine=engine) as writer:
            # 总体统计表（添加外观评级）
            stats_data = {
                '统计项目': ['总图片数', '总腐蚀点数', '平均每图腐蚀点数', '总面积(mm²)',
                             '平均腐蚀点面积(mm²)', '外观评级'],
                '数值': [
                    len(all_reports),
                    len(detailed_data),
                    len(detailed_data) / len(all_reports) if all_reports else 0,
                    total_all_area,
                    total_all_area / len(detailed_data) if detailed_data else 0,
                    appearance_grade
                ]
            }
            df_stats = pd.DataFrame(stats_data)
            df_stats.to_excel(writer, sheet_name='总体统计', index=False)

            # 汇总表
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='汇总统计', index=False)

            # 详细数据表
            df_detailed = pd.DataFrame(detailed_data)
            df_detailed.to_excel(writer, sheet_name='详细数据', index=False)

            # 校准信息表
            calibration_df = pd.DataFrame([CALIBRATION_DATA])
            calibration_df.to_excel(writer, sheet_name='校准信息', index=False)

            # 评级标准表
            grade_standard_data = []
            for grade, (min_area, max_area) in APPEARANCE_GRADE_STANDARD.items():
                if max_area == float('inf'):
                    area_range = f"≥{min_area}mm²"
                else:
                    area_range = f"{min_area}-{max_area}mm²"

                grade_description = {
                    'A': '无缺陷',
                    'B': '软氧管常见，正常不及时阻塞及周围环路的残留',
                    'C': '软氧管中氧化氮或氧气缺乏',
                    'D': '阳极性氧基酸的腐蚀产物',
                    'E': '阳极性氧基酸的腐蚀产物',
                    'F': '表面活性',
                    'G': '树脂、黏度、粘度',
                    'H': '颜色',
                    'I': '干燥',
                    'J': '龟裂',
                    'K': '高浓度碳化物'
                }

                grade_standard_data.append({
                    '评级': grade,
                    '面积范围(mm²)': area_range,
                    '描述': grade_description.get(grade, '')
                })

            df_grade_standard = pd.DataFrame(grade_standard_data)
            df_grade_standard.to_excel(writer, sheet_name='评级标准', index=False)

        # 如果使用openpyxl引擎，可以调整列宽
        if engine == 'openpyxl':
            auto_adjust_column_widths(excel_file)

        print(f"Excel报告已保存到: {excel_file}")
        return excel_file

    except Exception as e:
        print(f"生成Excel文件时出错: {e}")
        print("将生成CSV文件作为备用")
        return create_csv_report(all_reports, output_folder)


def create_csv_report(all_reports, output_folder):
    """
    创建CSV格式的报告（备用方案）
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 生成汇总数据
    summary_data = []
    detailed_data = []

    for report in all_reports:
        image_name = report['image_name']
        total_points = report['total_corrosion_points']
        total_area = report['total_area_mm2']

        summary_data.append({
            '图片名称': image_name,
            '腐蚀点数量': total_points,
            '总面积(mm²)': total_area,
            '平均面积(mm²)': total_area / total_points if total_points > 0 else 0
        })

        for detail in report['corrosion_details']:
            detailed_data.append({
                '图片名称': image_name,
                '腐蚀点ID': detail['id'],
                '类别': detail['category'],
                '置信度': detail['confidence'],
                '宽度(mm)': detail['width_mm'],
                '高度(mm)': detail['height_mm'],
                '面积(mm²)': detail['area_mm2']
            })

    # 保存CSV文件
    df_summary = pd.DataFrame(summary_data)
    df_detailed = pd.DataFrame(detailed_data)

    summary_csv = os.path.join(output_folder, f'summary_report_{timestamp}.csv')
    detailed_csv = os.path.join(output_folder, f'detailed_report_{timestamp}.csv')

    df_summary.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    df_detailed.to_csv(detailed_csv, index=False, encoding='utf-8-sig')

    print(f"汇总报告已保存到: {summary_csv}")
    print(f"详细报告已保存到: {detailed_csv}")
    return summary_csv, detailed_csv


def process_single_folder(input_folder, output_folder):
    """
    处理单个文件夹中的所有图片并生成报告
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中所有图片文件（包括子文件夹）
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []

    # 递归查找所有图片文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(supported_formats):
                # 保持相对路径结构
                rel_path = os.path.relpath(os.path.join(root, file), input_folder)
                image_files.append(rel_path)

    if not image_files:
        print(f"在文件夹 {input_folder} 中没有找到支持的图片文件")
        return

    all_reports = []

    # 处理每张图片
    for img_file in image_files:
        input_path = os.path.join(input_folder, img_file)

        # 在输出文件夹中保持相同的目录结构
        output_subdir = os.path.dirname(img_file)
        output_dir_path = os.path.join(output_folder, output_subdir)
        os.makedirs(output_dir_path, exist_ok=True)

        output_filename = f"result_{os.path.basename(img_file)}"
        output_path = os.path.join(output_dir_path, output_filename)

        print(f"\n正在处理: {img_file}")
        detection_result, area_info = detect_with_sahi(input_path, output_path)

        if detection_result and area_info:
            # 生成单张图片的报告
            image_report = {
                'image_name': img_file,
                'total_corrosion_points': len(area_info),
                'total_area_mm2': sum(info['area_mm2'] for info in area_info),
                'corrosion_details': area_info
            }

            all_reports.append(image_report)

            # 打印结果
            print(f"检测到 {len(area_info)} 个腐蚀点")
            print("腐蚀点详情:")
            for info in area_info:
                print(f"  点{info['id']}: {info['area_mm2']:.2f}mm² "
                      f"(置信度: {info['confidence']:.2f})")
            print(f"总面积: {image_report['total_area_mm2']:.2f}mm²")
        else:
            print("未检测到腐蚀点")

    # 生成报告
    if all_reports:
        result = create_excel_report(all_reports, output_folder)
        if isinstance(result, tuple):
            print(f"\nCSV报告已生成: {result[0]} 和 {result[1]}")
        else:
            print(f"\n完整Excel报告已生成: {result}")
    else:
        print("没有检测到任何腐蚀点，未生成报告")


def process_all_subfolders(base_input_folder, base_output_folder):
    """
    处理基础文件夹中的所有子文件夹
    """
    # 确保基础输出文件夹存在
    os.makedirs(base_output_folder, exist_ok=True)

    # 获取所有子文件夹（包括基础文件夹本身）
    all_folders = [base_input_folder]

    # 递归查找所有子文件夹
    for root, dirs, files in os.walk(base_input_folder):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            all_folders.append(full_path)

    print(f"找到 {len(all_folders)} 个文件夹需要处理")

    # 处理每个文件夹
    for input_folder in all_folders:
        # 计算对应的输出文件夹路径
        if input_folder == base_input_folder:
            output_folder = base_output_folder
        else:
            rel_path = os.path.relpath(input_folder, base_input_folder)
            output_folder = os.path.join(base_output_folder, rel_path)

        print(f"\n{'=' * 50}")
        print(f"处理文件夹: {input_folder}")
        print(f"输出到: {output_folder}")
        print(f"{'=' * 50}")

        process_single_folder(input_folder, output_folder)


# 主程序
if __name__ == '__main__':
    # 输入和输出基础文件夹路径
    base_input_folder = 'corrosion'
    base_output_folder = 'output'

    # 显示校准信息
    print("校准参数:")
    print(f"  图像高度: {CALIBRATION_DATA['image_height_pixels']} 像素")
    print(f"  标定长度: {CALIBRATION_DATA['calibration_length_mm']} mm")
    print(f"  像素尺寸: {CALIBRATION_DATA['pixel_size_mm_per_pixel']:.6f} mm/像素")
    print(f"  换算比例: 1 像素 = {CALIBRATION_DATA['pixel_size_mm_per_pixel']:.6f} mm")

    # 显示评级标准
    print("\n外观评级标准:")
    for grade, (min_area, max_area) in APPEARANCE_GRADE_STANDARD.items():
        if max_area == float('inf'):
            print(f"  {grade}: ≥{min_area}mm²")
        else:
            print(f"  {grade}: {min_area}-{max_area}mm²")

    # 检查并显示可用引擎
    check_excel_engines()

    # 批量处理所有文件夹
    process_all_subfolders(base_input_folder, base_output_folder)
    print("\n所有文件夹处理完成!")