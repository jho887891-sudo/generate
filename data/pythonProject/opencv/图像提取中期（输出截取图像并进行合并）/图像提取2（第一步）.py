import cv2
import os
from datetime import datetime

# 创建一个以当前时间戳命名的子文件夹路径
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
parent_folder = 'detection_results'  # 父文件夹名
folder_name = os.path.join(parent_folder, timestamp)  # 结合父文件夹和时间戳
os.makedirs(folder_name, exist_ok=True)  # 创建文件夹，如果已经存在则不会报错

# 创建一个子文件夹，比如 'output'
subfolder_name = os.path.join(folder_name, 'output')
os.makedirs(subfolder_name, exist_ok=True)

# 设置原始图像的路径
image_path = os.path.join('images', 'detect1', 'image11.jpg')

# 读取原始图像
image = cv2.imread(image_path)

# 设置模板图像的路径
template_path = os.path.join('templates', 'template11.jpg')

# 读取模板图像
template = cv2.imread(template_path, 0)

# 将原始图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用模板匹配
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 计算模板区域的位置
w, h = template.shape[::-1]
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# 调整矩形的位置和大小
padding_x = 100
padding_y = 100
bottom_right = (bottom_right[0] + padding_x, bottom_right[1] + padding_y)

# 确保截取区域在图像范围内
if top_left[0] < 0:
    top_left = (0, top_left[1])
if top_left[1] < 0:
    top_left = (top_left[0], 0)
if bottom_right[0] > image.shape[1]:
    bottom_right = (image.shape[1], bottom_right[1])
if bottom_right[1] > image.shape[0]:
    bottom_right = (bottom_right[0], image.shape[0])

# 绘制矩形（可选）
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# 截取区域
cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# 保存图像到新创建的子文件夹
cv2.imwrite(os.path.join(subfolder_name, 'detected_image.jpg'), image)
cv2.imwrite(os.path.join(subfolder_name, 'cropped_image.jpg'), cropped_image)

# 调整图像大小（可选）
scaled_image = cv2.resize(image, (800, 600))
scaled_cropped_image = cv2.resize(cropped_image, (800, 600))

# 显示调整后的图像（可选）
cv2.imshow('Detected', scaled_image)
cv2.imshow('Cropped', scaled_cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
