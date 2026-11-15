import cv2
import numpy as np
import os
from datetime import datetime

def process_image(img):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行阈值处理，找到黑线
    _, black_region = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    # 选择水平方向的结构元素（1x20），用于横向膨胀
    horizontal_kernel = np.ones((1, 20), np.uint8)  # 20个像素的宽度

    # 水平膨胀，朝向白色膨胀
    dilated_black_region_horizontal = cv2.dilate(black_region, horizontal_kernel, iterations=1)

    # 选择垂直方向的结构元素（20x1），用于纵向膨胀
    vertical_kernel = np.ones((20, 1), np.uint8)  # 20个像素的高度

    # 垂直膨胀，朝向白色膨胀
    dilated_black_region_vertical = cv2.dilate(black_region, vertical_kernel, iterations=1)

    # 将膨胀后的块填充为黑色
    img[dilated_black_region_horizontal > 0] = [0, 0, 0]  # 水平膨胀后的填充
    img[dilated_black_region_vertical > 0] = [0, 0, 0]    # 垂直膨胀后的填充

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
