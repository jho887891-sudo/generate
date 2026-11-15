import cv2
import numpy as np
import os
from datetime import datetime


def process_image(img):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用阈值将白色分离出来
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 进行边缘检测
    edges = cv2.Canny(thresholded, 50, 150)

    # 过滤小的白点（使用形态学开运算）
    kernel = np.ones((5, 5), np.uint8)  # 可以根据需要调整核的大小
    filtered = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    # 将白线设置为黑色（填充），保留白点
    img[edges > 0] = [0, 0, 0]  # 将边缘处的像素设置为黑色

    # 将白点（小于一定大小）填充为黑色
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # 可以根据需要调整面积阈值
            cv2.drawContours(img, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

    return img  # 返回处理后的原图


def process_images_in_folder(input_folder):
    # 创建主输出文件夹
    main_output_folder = 'output'
    if not os.path.exists(main_output_folder):
        os.makedirs(main_output_folder)

    # 生成基于当前日期和时间的子输出文件夹名称
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = os.path.join(main_output_folder, f'output_{timestamp}')  # 生成新的子文件夹名称

    # 确保子输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 遍历所有图像文件
    for idx, image_file in enumerate(image_files):
        img_path = os.path.join(input_folder, image_file)

        # 读取图像
        img = cv2.imread(img_path)

        # 检查图像是否读取成功
        if img is None:
            print(f"Error: Unable to load image {image_file}.")
            continue

        # 处理图像
        processed_image = process_image(img)

        # 输出文件名，改为 jpg 格式
        output_file_name = f"image{idx + 1}.jpg"  # 使用 .jpg 格式
        output_path = os.path.join(output_folder, output_file_name)

        # 保存处理后的图像
        cv2.imwrite(output_path, processed_image)
        print(f"Processed image saved as {output_path}")


# 示例用法
input_folder = 'images/image1'  # 输入文件夹路径
process_images_in_folder(input_folder)
