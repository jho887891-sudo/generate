import json
import numpy as np
import cv2
from PIL import Image
import os


def json_to_mask(json_file, output_dir):
    # 读取labelme生成的json文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 获取图片的尺寸
    img_width = data['imageWidth']
    img_height = data['imageHeight']

    # 创建一个全为6的mask图像作为背景，大小和原图一致
    mask = np.full((img_height, img_width), 6, dtype=np.uint8)

    # 处理每个标注
    for shape in data['shapes']:
        label = shape['label']  # 获取当前标注的标签

        # 检查标签是否是1到19的数字
        try:
            label_value = int(label)
            if label_value < 1 or label_value > 19:
                print(f"Warning: Label {label} has value out of range (1-19). Skipping.")
                continue
        except ValueError:
            print(f"Warning: Label {label} is not a number between 1-19. Skipping.")
            continue

        points = shape['points']  # 获取当前标注的轮廓点
        points = np.array(points, dtype=np.int32)

        # 填充多边形区域
        cv2.fillPoly(mask, [points], label_value)

    # 保存mask图像
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_mask_path = os.path.join(output_dir, os.path.basename(json_file).replace('.json', '_mask.png'))
    Image.fromarray(mask).save(output_mask_path)

    print(f'Mask image saved to: {output_mask_path}')
    return output_mask_path


def batch_process_json_files(json_dir, output_dir):
    # 获取所有json文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    # 处理每个json文件
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        json_to_mask(json_path, output_dir)


if __name__ == '__main__':
    json_dir = '11193 NG3'  # 存放json文件的文件夹路径
    output_dir = 'jpg'  # 存放生成的mask图片的目录

    batch_process_json_files(json_dir, output_dir)