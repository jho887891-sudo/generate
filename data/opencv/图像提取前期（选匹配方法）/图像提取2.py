import cv2

# 读取图像
image = cv2.imread('image2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 读取模板图像
template = cv2.imread('muban7.jpg', 0)
w, h = template.shape[::-1]

# 设定阶梯区域在模板中的位置和尺寸
x_step, y_step = 70, 60  # 阶梯区域左上角在模板中的相对坐标
w_step, h_step = 120, 90  # 阶梯区域的宽度和高度

# 使用模板匹配
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# 调整矩形位置和大小
padding_x = 100
padding_y = 100
top_left = max_loc
bottom_right = (top_left[00] + w + padding_x, top_left[1] + h + padding_y)

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
cv2.waitKey(0)
cv2.destroyAllWindows()
