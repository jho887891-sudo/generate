import cv2
import numpy as np
import os
from datetime import datetime


def remove_white_lines(original_image):
    # 加载原图
    original = cv2.imread(original_image)
    if original is None:
        print(f"Error: Unable to read original image {original_image}. Skipping.")
        return None

    # 转换为灰度图
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(original_gray, (9, 9), 0)  # 增加核大小
    # 去噪声
    median_filtered = cv2.medianBlur(blurred, 5)
    equ = cv2.equalizeHist(median_filtered)

    # 使用Canny边缘检测来识别白线
    edges = cv2.Canny(median_filtered, 30, 150)

    # 创建一个全零的黑色图像，与原图大小相同
    black_image = np.zeros_like(original_gray)

    # 使用掩模将白线变为黑色
    black_image[edges > 0] = 0  # 将检测到的白线部分设置为黑色

    # 创建一个掩膜，检测并保留小白点
    _, white_spots_mask = cv2.threshold(original_gray, 200, 255, cv2.THRESH_BINARY)

    # 找到白点的轮廓
    contours, _ = cv2.findContours(white_spots_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 保留小白点
    for contour in contours:
        # 计算白点的面积
        area = cv2.contourArea(contour)
        if area < 100:  # 根据需求调整白点的最大面积
            cv2.drawContours(black_image, [contour], -1, (255), thickness=cv2.FILLED)

    # 将原图中非白线的部分复制到黑色图像中
    black_image[original_gray < 200] = original_gray[original_gray < 200]  # 保留灰色和其他部分

    return black_image

def process_images_in_folder(input_folder, output_folder):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有图像文件
    original_images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                       f.endswith(('jpg', 'jpeg', 'png'))]

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
input_folder = 'images/image4'  # 输入文件夹
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = os.path.join('output', current_time)  # 输出文件夹

process_images_in_folder(input_folder, output_folder)
