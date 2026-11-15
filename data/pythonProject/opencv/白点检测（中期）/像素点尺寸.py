import cv2
import numpy as np
import json

# 读取图像
image_path = 'template8.jpg'  # 替换为你的图像路径
image = cv2.imread(image_path)

# 获取图像的高度和宽度
height, width = image.shape[:2]

# 输入已知的标定物体的实际高度
calibration_length_mm = float(input("请输入标定物体的实际高度（mm）："))

# 计算每个像素对应的实际尺寸（单位：mm/pixel）
pixel_size = calibration_length_mm / height if height > 0 else 0

# 输出每个像素对应的实际尺寸
print(f"图像的高度：{height} 像素, 每个像素对应的实际尺寸：{pixel_size:.4f} mm/pixel")

# 将结果保存到 JSON 文件
result = {
    "image_height_pixels": height,
    "calibration_length_mm": calibration_length_mm,
    "pixel_size_mm_per_pixel": pixel_size
}

# 保存为 JSON 文件
json_filename = "../../x光图预测/X-data/区域检测面积/calibration_result.json"
with open(json_filename, 'w') as json_file:
    json.dump(result, json_file, indent=4)

# 显示图像（可选）
cv2.line(image, (width // 2, 0), (width // 2, height), (0, 0, 255), 2)  # 在图像上画一条线
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
