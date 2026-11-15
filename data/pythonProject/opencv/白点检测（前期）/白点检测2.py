import cv2
import numpy as np

# 加载图像
image = cv2.imread('test.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用阈值
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# 检测轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 筛选和绘制白点
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 10:  # 根据需要调整最小面积
        # 获取轮廓的边界框
        (x, y, w, h) = cv2.boundingRect(cnt)
        # 在边界框上绘制矩形
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 保存结果
cv2.imwrite('detected_points.png', image)

# 显示结果
cv2.imshow('Detected Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
