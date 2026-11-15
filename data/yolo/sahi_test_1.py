import cv2
import numpy as np
import os
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions

# 1. 初始化YOLO检测模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov11',
    model_path='best.pt',
    confidence_threshold=0.3,
    device='cuda:0',
    # image_size=640
)


# 2. 定义检测函数
def detect_with_sahi(image_path, output_path):
    """
    使用SAHI进行切片推理

    参数:
        image_path: 输入图像路径
        output_path: 结果保存路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

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
    # print(result)
    # 可视化结果
    visualization_result = visualize_object_predictions(
        image,
        object_prediction_list=result.object_prediction_list,
        output_dir=None,
        rect_th=0,
        text_th=0,
        text_size=0,
        hide_labels=True,    # 显式隐藏标签（如果SAHI支持此参数）
        hide_conf=False
    )

    # 保存结果
    cv2.imwrite(output_path, visualization_result['image'])
    print(f"结果已保存到: {output_path}")
    return result


# 3. 批量处理文件夹中的图片
def process_folder(input_folder, output_folder):
    """
    处理文件夹中的所有图片

    参数:
        input_folder: 输入图片文件夹路径
        output_folder: 输出结果文件夹路径
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中所有图片文件
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"在文件夹 {input_folder} 中没有找到支持的图片文件")
        return

    # 处理每张图片
    for img_file in image_files:
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, f"result_{img_file}")

        print(f"\n正在处理: {img_file}")
        detection_result = detect_with_sahi(input_path, output_path)

        if detection_result:
            print("检测结果:")
            for pred in detection_result.object_prediction_list:
                print(
                    f"类别: {pred.category.name}, "
                    f"置信度: {pred.score.value:.2f}, "
                    f"位置: {pred.bbox.to_xyxy()}"
                )


# 4. 主程序
if __name__ == '__main__':
    # 输入和输出文件夹路径
    input_folder = 'images'
    output_folder = 'output'

    # 批量处理
    process_folder(input_folder, output_folder)
    print("\n所有图片处理完成!")