import cv2
import os
import numpy as np
from tqdm import tqdm  # 用于显示进度条


def convert_to_grayscale(input_dir, output_dir):
    """
    将输入目录中的所有彩色图像转换为黑白图像并保存到输出目录

    参数:
        input_dir (str): 包含彩色图像的输入目录路径
        output_dir (str): 保存黑白图像的输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取输入目录中的所有图像文件
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    print(f"正在处理 {len(image_files)} 张图像...")

    for img_file in tqdm(image_files):
        try:
            # 读取图像
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)

            # 转换为灰度图
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 保存灰度图
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, gray_img)

        except Exception as e:
            print(f"处理 {img_file} 时出错: {e}")


# 使用示例
if __name__ == "__main__":
    input_directory = "images"  # 替换为你的彩色图像目录
    output_directory = "output"  # 替换为你想保存灰度图的目录

    convert_to_grayscale(input_directory, output_directory)
    print("转换完成！")