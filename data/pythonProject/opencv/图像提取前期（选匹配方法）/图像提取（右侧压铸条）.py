import cv2

# 读取图像
image = cv2.imread('image2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 读取模板图像
template = cv2.imread('muban.jpg', 0)
w, h = template.shape[::-1]

# 使用模板匹配
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# 定义最左侧部分的宽度
w_left = w // 3  # 提取模板的最左侧三分之一部分

# 确定最左侧部分的区域
top_left = max_loc
left_top_left = (top_left[0], top_left[1])  # 最左侧部分的左上角
left_bottom_right = (top_left[0] + w_left, top_left[1] + h)  # 最左侧部分的右下角

# 确保截取区域在图像范围内
if left_top_left[0] < 0:
    left_top_left = (0, left_top_left[1])
if left_top_left[1] < 0:
    left_top_left = (left_top_left[0], 0)
if left_bottom_right[0] > image.shape[1]:
    left_bottom_right = (image.shape[1], left_bottom_right[1])
if left_bottom_right[1] > image.shape[0]:
    left_bottom_right = (left_bottom_right[0], image.shape[0])

# 绘制矩形（可选）
cv2.rectangle(image, left_top_left, left_bottom_right, (0, 255, 0), 2)

# 截取最左侧部分区域
cropped_image = image[left_top_left[1]:left_bottom_right[1], left_top_left[0]:left_bottom_right[0]]

# 显示结果
cv2.imshow('Detected', image)
cv2.imshow('Cropped Left Side', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
