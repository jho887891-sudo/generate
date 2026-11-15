import cv2
import numpy as np

# 定义函数detect_white_points 接受两个参数
# image_path: 输入图像的文件路径。
# output_path: 处理后图像的输出路径。
def detect_white_points(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 检查图像是否成功读取。如果未找到图像，打印错误信息并返回，避免后续调用因空图像而出错。
    if image is None:
        print("Error: Image not found.")
        return

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用阈值
    # 使用 cv2.threshold() 对灰度图进行二值化处理。
    # 第一个参数是输入图像（灰度图）。
    # 第二个参数为阈值（240）：大于该值的像素将被设为白色（255），小于或等于该值的像素将被设为黑色（0）。
    # THRESH_BINARY 指定了二值化方法，生成的二值图像将存储在 thresh 中
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    # 使用cv2.findContours() 查找二值图像中的轮廓。
    # RETR_EXTERNAL 表示只提取外部轮廓。
    # CHAIN_APPROX_SIMPLE 表示使用简单的轮廓近似法，减少存储空间
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓
    for contour in contours:
        # 遍历所有找到的轮廓，使用cv2.contourArea() 计算每个轮廓的面积。
        # 仅处理面积大于10的轮廓，以过滤掉小的噪声和不相关的区域。
        if cv2.contourArea(contour) > 10:  # 过滤小轮廓
            # 获取轮廓的边界框
            # 使用 cv2.boundingRect() 计算每个有效轮廓的最小外接矩形，返回矩形的左上角坐标(x, y)和宽高(w, h)。
            # 使用 cv2.rectangle()在原图像上绘制绿色矩形框，标记检测到的白点区域。
            # 颜色设为绿色(0, 255, 0)，线宽为2。
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制矩形框

    # 显示和保存结果
    # 使用cv2.imshow() 显示带有检测结果的图像窗口，标题为 "Detected White Points"。
    cv2.imshow('Detected White Points', image)
    cv2.imwrite(output_path, image)
    # cv2.waitKey(0) 等待用户按下任意键，防止窗口立即关闭。
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
detect_white_points('test.jpg', 'output_image-1.jpg')