import cv2
import numpy as np
import os
import glob
from datetime import datetime

# 每个像素对应的尺寸（mm）
PIXEL_SIZE_MM = 0.1
MAX_DIAMETER_MM = 5.0  # 最大直径设置为5 mm
MIN_DIAMETER_MM = 0.4  # 最小直径设置为0.4 mm
MIN_AREA_MM2 = np.pi * (MAX_DIAMETER_MM / 2) ** 2  # 最小面积

def detect_white_points(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image {image_path} not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 可以尝试增大这个值
    _, thresh = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)  # 调整阈值

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_diameter_pixels = MAX_DIAMETER_MM / PIXEL_SIZE_MM  # 最大直径像素值
    min_area_pixels = MIN_AREA_MM2 / (PIXEL_SIZE_MM ** 2)  # 最小面积像素值

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area_pixels:  # 更新面积过滤条件
            (x, y), radius = cv2.minEnclosingCircle(contour)
            diameter_pixels = radius * 2
            if diameter_pixels <= max_diameter_pixels:  # 直径过滤
                diameter_mm = diameter_pixels * PIXEL_SIZE_MM  # 转换为毫米

                center = (int(x), int(y))
                cv2.circle(image, center, int(radius), (0, 255, 0), 2)
                cv2.putText(image, f'Diameter: {diameter_mm:.2f} mm',
                            (center[0] - 50, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imwrite(output_path, image)

def process_images_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = glob.glob(os.path.join(input_folder, '*.jpg'))

    for image_file in image_files:
        output_file = os.path.join(output_folder, os.path.basename(image_file))
        detect_white_points(image_file, output_file)

if __name__ == "__main__":
    input_folder = 'image'
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join('output', current_time)
    process_images_in_folder(input_folder, output_folder)
