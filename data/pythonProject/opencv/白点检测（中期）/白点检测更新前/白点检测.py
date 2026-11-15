import cv2
import numpy as np
import os
import glob
from datetime import datetime

def detect_white_points(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image {image_path} not found.")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用阈值
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 10:  # 过滤小轮廓
            # 获取轮廓的边界框和最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            diameter = radius * 2  # 直径为半径的两倍

            # 在图像上绘制白点的圆和直径信息
            center = (int(x), int(y))
            cv2.circle(image, center, int(radius), (0, 255, 0), 2)  # 绘制圆
            cv2.putText(image, f'Diameter: {diameter:.2f} px',
                        (center[0] - 50, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 保存处理后的图像
    cv2.imwrite(output_path, image)

def process_images_in_folder(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中所有图像文件
    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))  # 可以修改为支持其他格式

    for image_file in image_files:
        output_file = os.path.join(output_folder, os.path.basename(image_file))
        detect_white_points(image_file, output_file)

# 主程序
if __name__ == "__main__":
    # 设置输入和输出文件夹
    input_folder = 'image'
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join('output', current_time)

    # 处理所有图像
    process_images_in_folder(input_folder, output_folder)
