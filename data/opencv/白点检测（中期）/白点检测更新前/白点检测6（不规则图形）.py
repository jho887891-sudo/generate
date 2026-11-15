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
    # 中值滤波去除散粒噪声
    median_filtered = cv2.medianBlur(blurred, 5)
    equ = cv2.equalizeHist(median_filtered)
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
    thresh = cv2.bitwise_not(thresh)

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

    # 识别
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 调整
    lower_white = np.array([0, 0, 194])  # 低阈值
    upper_white = np.array([180, 15, 255])  # 高阈值
    mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # 形态学操作以去除小噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50:  # 过滤小面积的轮廓
            continue

        # 计算外接圆和近似多边形
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter_mm = 2 * radius * PIXEL_SIZE_MM

        # 计算轮廓的近似多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)  # 近似精度
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 计算轮廓的周长和圆度
        perimeter = cv2.arcLength(approx, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

        if MIN_DIAMETER_MM <= diameter_mm <= MAX_DIAMETER_MM:
            # 使用质心来标记形状
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # 过滤不规则形状，设置一个宽松的圆度条件
            if circularity > 0.5:  # 过滤不规则形状
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
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
    config_path = 'config.json'  # 配置文件路径

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
