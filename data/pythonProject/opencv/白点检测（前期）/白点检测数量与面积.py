
import cv2
import numpy as np


def detect_white_points(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用阈值
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化计数器
    white_point_count = 0
    areas = []  # 用于存储白点的面积

    for contour in contours:
        if cv2.contourArea(contour) > 10:  # 过滤小轮廓
            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制矩形框

            # 计算白点尺寸（面积）
            pixel_area = cv2.contourArea(contour)
            actual_area_mm2 = pixel_area * (0.001 * 0.001)  # 转换为平方毫米 设置每个像素点为0.01mm

            areas.append(actual_area_mm2)  # 存储每个白点的面积

            white_point_count += 1  # 增加计数

            # 在图像上显示白点的尺寸
            cv2.putText(image, f'Area: {actual_area_mm2:.7f} mm²',
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 输出总白点数量和每个白点的面积
    total_text = f"Total White Points: {white_point_count}"
    cv2.putText(image, total_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for i, area in enumerate(areas):
        print(f"White Point {i + 1}: Area = {area:.7f} mm²")

    # 显示和保存结果
    cv2.imshow('Detected White Points', image)
    cv2.imwrite(output_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用示例
detect_white_points('test.jpg', 'output_image-3.jpg')