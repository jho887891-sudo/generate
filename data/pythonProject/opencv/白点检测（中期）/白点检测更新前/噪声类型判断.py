import cv2
import numpy as np


def calculate_noise(image):
    # 转为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算均值和标准差
    mean, stddev = cv2.meanStdDev(gray_image)

    # 打印均值和标准差
    print(f"Mean: {mean[0][0]}, Stddev: {stddev[0][0]}")

    # 使用标准差作为噪声的衡量
    noise_level = stddev[0][0]

    # 根据噪声水平判断
    if noise_level < 25:
        return "Low Noise"
    elif noise_level < 50:
        return "Medium Noise"
    else:
        return "High Noise"


# 读取图片
image_path = 'images/image5.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)

# 判断噪声
result = calculate_noise(image)
print("Noise Level:", result)
