import cv2
import numpy as np
import os
import glob
from datetime import datetime

# 每个像素对应的尺寸（mm）
PIXEL_SIZE_MM = 0.1
MIN_DIAMETER_MM = 0.4  # 最小直径
MAX_DIAMETER_MM = 5.0  # 最大直径
MIN_AREA_MM2 = np.pi * (MIN_DIAMETER_MM / 2) ** 2  # 最小面积
MAX_AREA_MM2 = np.pi * (MAX_DIAMETER_MM / 2) ** 2  # 最大面积

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(equ)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    enhanced = cv2.filter2D(cl1, -1, kernel)
    return enhanced

def detect_white_points(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Image {image_path} not found.")
        return

    enhanced_image = enhance_image(image)
    _, thresh = cv2.threshold(enhanced_image, 230, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        diameter_mm = 2 * np.sqrt(area / np.pi) * PIXEL_SIZE_MM  # 计算直径

        if MIN_DIAMETER_MM <= diameter_mm <= MAX_DIAMETER_MM:  # 过滤条件
            (x, y), radius = cv2.minEnclosingCircle(contour)
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
    input_folder = 'images/image3'
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join('output', current_time)
    process_images_in_folder(input_folder, output_folder)
