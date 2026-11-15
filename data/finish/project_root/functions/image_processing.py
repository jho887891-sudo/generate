import cv2
import numpy as np
import os

def process_images(input_folder, output_folder, border_size=76):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"读取图像失败: {image_path}")
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, mask1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
                _, mask2 = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)
                mask = cv2.bitwise_and(mask1, mask2)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    x = max(0, x - border_size)
                    y = max(0, y - border_size)
                    w = min(image.shape[1] - x, w + 2 * border_size)
                    h = min(image.shape[0] - y, h + 2 * border_size)

                    cropped_image = image[y:y+h, x:x+w]
                    white_background = np.ones((h, w, 3), dtype=np.uint8) * 255
                    white_background[0:h, 0:w] = cropped_image

                    output_path = os.path.join(output_folder, filename)
                    success = cv2.imwrite(output_path, white_background)
                    if success:
                        print(f"处理后的图像已保存到: {output_path}")
                    else:
                        print(f"图像保存失败: {output_path}")
                else:
                    print(f"{filename} 没有找到轮廓。")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

def find_and_process_ok_folders(base_folder, output_base_folder):
    for root, dirs, files in os.walk(base_folder):
        for dir_name in dirs:
            if "OK" in dir_name and "NG" not in dir_name:
                input_folder = os.path.join(root, dir_name)
                output_folder = os.path.join(output_base_folder, dir_name)
                process_images(input_folder, output_folder)
