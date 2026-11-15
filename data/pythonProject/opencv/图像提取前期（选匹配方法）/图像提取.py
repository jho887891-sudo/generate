import cv2

# 读取图像
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 读取模板图像
template = cv2.imread('muban.jpg', 0)
w, h = template.shape[::-1]

# 使用模板匹配
res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# 确定位置
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# 绘制矩形（可选）
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

# 截取区域
cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

# 显示结果
cv2.imshow('Detected', image)
cv2.imshow('Cropped', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
