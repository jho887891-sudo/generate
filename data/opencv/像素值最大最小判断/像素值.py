from PIL import Image
import numpy as np

def get_max_pixel_values(image_path, threshold=128, limit=230, count=5):
    # 读取图片
    image = Image.open(image_path)

    # 转换为灰度图
    gray_image = image.convert("L")

    # 将灰度图转换为numpy数组
    gray_array = np.array(gray_image)

    # 查找小于limit的原始灰度值
    filtered_array = gray_array[gray_array < limit]

    # 检查是否有符合条件的元素
    if filtered_array.size > 0:
        # 获取最大值并去重
        unique_values = np.unique(filtered_array)
        # 从大到小排序
        sorted_values = np.sort(unique_values)[::-1]
        # 取前count个最大值
        max_values = sorted_values[:count]
    else:
        max_values = None  # 或者设置为空数组，表示没有符合条件的像素值

    return max_values

# 使用示例
image_path = 'images/image7.jpg'  # 替换为你的图片路径
max_values = get_max_pixel_values(image_path)

# 正确引用 limit
if max_values is not None:
    print(f"图片中小于{230}的最大像素点值依次为: {max_values}")
else:
    print(f"没有找到小于{230}的像素点值。")
