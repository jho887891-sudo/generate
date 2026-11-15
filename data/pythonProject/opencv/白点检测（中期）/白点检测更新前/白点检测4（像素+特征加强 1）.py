import cv2
import numpy as np
import os
import glob
from datetime import datetime
import json

# 创建或更新配置文件
def create_or_update_config(config_path):
    pixel_size_mm = float(input("请输入每个像素对应的实际尺寸（mm）："))
    config = {'pixel_size_mm': pixel_size_mm}
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"配置文件已创建/更新：{config_path}")

# 读取像素尺寸配置
def load_pixel_size(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['pixel_size_mm']

def enhance_image(image):
    # 灰度处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    equ = cv2.equalizeHist(gray)
    # 中值滤波去除散粒噪声
    median_filtered = cv2.medianBlur(blurred, 5)  # 添加中值滤波步骤
    equ = cv2.equalizeHist(median_filtered)  # 使用中值滤波后的图像进行直方图均衡化
    # 使用CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(equ)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(cl1, -1, kernel)

    # 使用Otsu的阈值化
    _, thresh = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 反向阈值化处理
    # thresh = cv2.bitwise_not(thresh)

    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    enhanced_image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    enhanced_image = cv2.morphologyEx(enhanced_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    return enhanced

def detect_white_points(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image {image_path} not found.")
        return

    enhanced_image = enhance_image(image)
    _, thresh = cv2.threshold(enhanced_image, 150, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # 过滤小面积的轮廓
            continue

        diameter_mm = 2 * np.sqrt(area / np.pi) * PIXEL_SIZE_MM
        if MIN_DIAMETER_MM <= diameter_mm <= MAX_DIAMETER_MM:
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (area / (perimeter ** 2))

            if circularity > 0.7:  # 过滤不规则形状
                # 用绿色绘制轮廓
                cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
                # 计算轮廓中心点
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # 在轮廓中心标记直径
                cv2.putText(image, f'Diameter: {diameter_mm:.2f} mm',
                            (cX - 50, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imwrite(output_path, image)

def process_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    for image_file in image_files:
        output_file = os.path.join(output_folder, os.path.basename(image_file))
        detect_white_points(image_file, output_file)

if __name__ == "__main__":
    config_path = '白点检测更新前/config.json'  # 配置文件路径

    # 检查配置文件是否存在，若不存在则创建
    if not os.path.exists(config_path):
        create_or_update_config(config_path)

    # 读取配置文件获取像素尺寸
    PIXEL_SIZE_MM = load_pixel_size(config_path)

    # 设定直径范围
    MIN_DIAMETER_MM = 0.1  # 最小直径
    MAX_DIAMETER_MM = 5.0  # 最大直径

    input_folder = 'images/image12'
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join('output', current_time)
    process_images_in_folder(input_folder, output_folder)
