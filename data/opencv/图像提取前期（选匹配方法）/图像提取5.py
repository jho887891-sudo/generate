import cv2
import numpy as np

# 读取图像
image = cv2.imread('image3.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 读取模板图像
template = cv2.imread('muban9.jpg', 0)
w, h = template.shape[::-1]

# 使用模板匹配
result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 最佳位置
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# 调整矩形位置和大小
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

# 显示结果
cv2.imshow('Detected', image)
cv2.imshow('Cropped', cropped_image)
# 保存图像
cv2.imwrite('detected_image.jpg', image)
cv2.imwrite('cropped_image.jpg', cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
