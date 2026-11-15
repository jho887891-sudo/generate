import cv2
import numpy as np
import os
from datetime import datetime

def enhance_contrast(image):
    # 使用CLAHE增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def remove_white_lines(original_image):
    # 加载原图
    original = cv2.imread(original_image)
    if original is None:
        print(f"Error: Unable to read original image {original_image}. Skipping.")
        return None

    # 转换为 HSV 色彩空间
    hsv_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    # 定义白色的范围
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])

    # 创建白色掩膜
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # 反转掩模以去除白色干扰
    mask_inv = cv2.bitwise_not(white_mask)

    # 转换为灰度图
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 增强对比度
    enhanced_gray = enhance_contrast(original_gray)

    # 高斯模糊和去噪声
    blurred = cv2.GaussianBlur(enhanced_gray, (7, 7), 0)
    median_filtered = cv2.medianBlur(blurred, 5)

    # 使用Canny边缘检测来识别白线
    edges = cv2.Canny(median_filtered, 40, 150)

    # 形态学操作
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    edges_eroded = cv2.erode(edges_dilated, kernel, iterations=1)

    # 创建一个全零的黑色图像，与原图大小相同
    black_image = np.zeros_like(original_gray)

    # 使用掩模将白线变为黑色
    black_image[edges_eroded > 0] = 0

    # 创建一个掩膜，检测并保留小白点
    _, white_spots_mask = cv2.threshold(original_gray, 150, 255, cv2.THRESH_BINARY)

    # 找到白点的轮廓
    contours, _ = cv2.findContours(white_spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 保留小白点
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80:  # 根据需要调整小白点的面积阈值
            cv2.drawContours(black_image, [contour], -1, (255), thickness=cv2.FILLED)

    # 将原图中非白线的部分复制到黑色图像中
    black_image[original_gray < 200] = original_gray[original_gray < 200]

    # 使用反转掩模去除白色干扰
    result = cv2.bitwise_and(original, original, mask=mask_inv)

    # 将处理结果与去除白色干扰的结果合并
    final_result = cv2.bitwise_or(result, cv2.cvtColor(black_image, cv2.COLOR_GRAY2BGR))

    # 后处理步骤：恢复被错误去除的白点
    # 重新检测白点
    _, restored_white_spots_mask = cv2.threshold(original_gray, 200, 255, cv2.THRESH_BINARY)
    contours_restored, _ = cv2.findContours(restored_white_spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 恢复白点
    for contour in contours_restored:
        area = cv2.contourArea(contour)
        if area < 100:  # 根据需要调整小白点的面积阈值
            cv2.drawContours(final_result, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    return final_result

def process_images_in_folder(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    # 获取输入文件夹中的所有图像文件
    original_images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('jpg', 'jpeg', 'png'))]

    # 对每张原图应用白线去除处理
    for original_image in original_images:
        processed_image = remove_white_lines(original_image)
        if processed_image is not None:
            # 创建输出文件名
            output_image_name = os.path.basename(original_image)
            output_image_path = os.path.join(output_folder, output_image_name)
            # 保存处理后的图像
            cv2.imwrite(output_image_path, processed_image)
            print(f"Processed image saved as {output_image_path}")

# 示例使用
input_folder = 'images/image3'  # 输入文件夹
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join('output', current_time)  # 输出文件夹
process_images_in_folder(input_folder, output_folder)
