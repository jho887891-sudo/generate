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
image_folder = 'images/detect1'
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
    'template7.jpg': ['image4.jpg', 'image5.jpg'],
    'template8.jpg': ['image8.jpg'],
    'template9.jpg': ['image10.jpg'],
    'template10.jpg': ['image10.jpg', 'image11.jpg'],
    'template11.jpg': ['image11.jpg'],
}

# 在处理模板时，仅处理一次
for template_file in sorted(correspondences.keys()):
    images = correspondences[template_file]
    template_path = os.path.join(template_folder, template_file)
    template = cv2.imread(template_path, 0)

    if template_file in ['template4.jpg', 'template5.jpg', 'template6.jpg', 'template7.jpg', 'template10.jpg', 'template11.jpg']:
        threshold = 0.4  # 对于相似度低的模板使用较低的阈值
    else:
        threshold = 0.8  # 默认阈值

    for image_file in images:
        image_path = os.path.join(image_folder, image_file)
        if os.path.isfile(image_path):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= threshold)

            if len(loc[0]) > 0:
                for match_index, pt in enumerate(zip(*loc[::-1]), start=1):
                    w, h = template.shape[::-1]
                    top_left = pt
                    bottom_right = (top_left[0] + w, top_left[1] + h)

                    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
                    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                    cv2.imwrite(os.path.join(subfolder_name, f'detected_{image_file}_{template_file}.jpg'), image)
                    cv2.imwrite(os.path.join(subfolder_name, f'cropped_{image_file}_{template_file}.jpg'),
                                cropped_image)

                    print(f"Saved results for {image_file}, match {match_index}")
                    break
            else:
                print(f"No matches found for {image_file}. Skipping...")
        else:
            print(f"Image file {image_file} not found.")

    print("所有图像处理完成！")