import cv2
import numpy as np

# 定义颜色映射 (BGR格式)
COLOR_MAP = {
    1: (0, 0, 255),  # 红色 - 阈值1
    2: (0, 255, 0),  # 绿色 - 阈值2
    3: (255, 0, 0),  # 蓝色 - 阈值3
    4: (0, 255, 255),  # 黄色 - 阈值4
    5: (255, 0, 255),  # 粉色 - 阈值5
    6: (255, 255, 0)  # 青色 - 阈值6
}


def visualize_mask_thresholds(mask_path):
    # 读取mask图像 (灰度模式)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("错误：无法读取图像，请检查文件路径")
        return

    # 创建彩色可视化图像
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # 应用颜色映射
    for value, color in COLOR_MAP.items():
        colored_mask[mask == value] = color

    # 创建图例
    legend = np.zeros((100, 300, 3), dtype=np.uint8)
    for i, (value, color) in enumerate(COLOR_MAP.items()):
        cv2.rectangle(legend, (i * 50, 0), ((i + 1) * 50, 50), color, -1)
        cv2.putText(legend, str(value), (i * 50 + 15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 显示结果
    cv2.imshow("Threshold Visualization", colored_mask)
    cv2.imshow("Legend", legend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果（可选）
    cv2.imwrite("threshold_visualization.png", colored_mask)
    cv2.imwrite("color_legend.png", legend)


if __name__ == "__main__":
    mask_path = "VK3-mask/image7_mask.png"  # 替换为你的mask图像路径
    visualize_mask_thresholds(mask_path)