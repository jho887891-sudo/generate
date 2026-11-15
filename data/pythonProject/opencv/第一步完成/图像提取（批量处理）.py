import cv2
import os
import numpy as np
from datetime import datetime

# 创建结果文件夹
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
parent_folder = 'detection_results'
folder_name = os.path.join(parent_folder, timestamp)
os.makedirs(folder_name, exist_ok=True)

# 创建子文件夹
subfolder_name = os.path.join(folder_name, 'output')
os.makedirs(subfolder_name, exist_ok=True)

# 定义原图和模板的路径
image_folder = 'images/detect22'
template_folder = 'templates'

# 获取所有原图和模板
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
template_files = [f for f in os.listdir(template_folder) if f.endswith('.jpg')]

# 定义模板与原图的对应关系
correspondences = {
    'template1.jpg': ['image1.jpg'],
    'template2.jpg': ['image1.jpg'],
    'template3.jpg': ['image2.jpg'],
    'template4.jpg': ['image1.jpg'],
    'template5.jpg': ['image2.jpg'],
    'template6.jpg': ['image3.jpg'],
    'template7.jpg': ['image5.jpg'],
    'template8.jpg': ['image8.jpg'],
    'template9.jpg': ['image10.jpg'],
    'template10.jpg': ['image10.jpg'],
    # 'template11.jpg': ['image11.jpg'],
    'template12.jpg': ['image4.jpg']
}

# 定义一个函数来旋转图像
def rotate_image(image, angle):
    """ 旋转图像函数 """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 旋转矩阵
    rotated_image = cv2.warpAffine(image, matrix, (width, height))  # 进行旋转
    return rotated_image

# 初始化计数器
counter = 1

# 遍历每个模板
for template_file in sorted(correspondences.keys()):
    images = correspondences[template_file]
    template_path = os.path.join(template_folder, template_file)
    template = cv2.imread(template_path, 0)
    if template is None:
        print(f"Template {template_file} not found. Skipping...")
        continue

    # 初始阈值和参数
    threshold = 0.8
    max_attempts = 6  # 最大尝试次数
    rotation_angles = [0, 1, -1, 2, -2, 3, -3]  # 定义旋转角度

    # 遍历所有与模板匹配的图片
    for image_file in images:
        image_path = os.path.join(image_folder, image_file)
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 进行图像预处理
            gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊

            # 尝试不同的旋转角度和阈值
            found_match = False
            for attempt in range(max_attempts):
                for angle in rotation_angles:
                    # 根据当前尝试的角度旋转原图
                    gray_rotated = rotate_image(gray, angle)

                    # 进行匹配
                    result = cv2.matchTemplate(gray_rotated, template, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(result >= threshold)

                    if len(loc[0]) > 0:
                        for match_index, pt in enumerate(zip(*loc[::-1]), start=1):
                            w, h = template.shape[::-1]
                            top_left = pt
                            bottom_right = (top_left[0] + w, top_left[1] + h)

                            # 只保存截取的图像，修改保存名称为'image'加上计数器
                            cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                            cv2.imwrite(os.path.join(subfolder_name, f'image{counter}.jpg'), cropped_image)

                            print(f"Saved cropped image for {image_file}, match {match_index} at angle {angle}")
                            found_match = True
                            counter += 1  # 生成一个新的截图后，计数器加1
                            break
                        if found_match:
                            break
                if found_match:
                    break
                else:
                    # 如果没有找到匹配，尝试降低阈值
                    print(f"No matches found for {image_file} with threshold {threshold}. Trying lower threshold...")
                    threshold -= 0.1  # 每次尝试降低阈值
                    if threshold < 0:
                        print(f"Max attempts reached for {image_file}. Skipping...")
                        break

        else:
            print(f"Image file {image_file} not found.")

print("图像处理完成！")
