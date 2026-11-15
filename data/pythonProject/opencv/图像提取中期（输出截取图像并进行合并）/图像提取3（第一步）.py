import cv2
import os
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
    'template1.jpg': ['image1.jpg'],        # 模板1对应原图1
    'template2.jpg': ['image1.jpg', 'image2.jpg'],  # 模板2对应原图1和原图2
    'template3.jpg': ['image2.jpg'],
    'template4.jpg': ['image1.jpg'],
    'template5.jpg': ['image2.jpg'],
    'template6.jpg': ['image3.jpg'],
    'template7.jpg': ['image4.jpg', 'image5.jpg'],
    'template8.jpg': ['image8.jpg'],
    'template9.jpg': ['image10.jpg'],
    'template10.jpg': ['image10.jpg', 'image11.jpg'],
    'template11.jpg': ['image11.jpg'],
    # 继续添加对应关系...
}

# 循环处理每个模板及其对应的原图
for template_file, images in correspondences.items():
    template_path = os.path.join(template_folder, template_file)
    template = cv2.imread(template_path, 0)

    for image_file in images:
        if image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 使用模板匹配
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 计算模板区域的位置
            w, h = template.shape[::-1]
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # 绘制矩形（可选）
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            # 截取区域
            cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # 保存结果
            cv2.imwrite(os.path.join(subfolder_name, f'detected_{image_file}'), image)
            cv2.imwrite(os.path.join(subfolder_name, f'cropped_{image_file}'), cropped_image)

# 输出处理完成的消息
print("所有图像处理完成！")
