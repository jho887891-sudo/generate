import os
from PIL import Image
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as ExcelImage


def generate_report(base_image_folder, template_path, output_folder, aspect_ratios):
    # 获取所有包含"OK"的子文件夹路径，并排除包含"NG"的文件夹
    image_folders = [
        os.path.join(base_image_folder, folder)
        for folder in os.listdir(base_image_folder)
        if os.path.isdir(os.path.join(base_image_folder, folder)) and "OK" in folder and "NG" not in folder
    ]

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 定义每个文件夹生成的报告
    for image_folder in image_folders:
        folder_name = os.path.basename(image_folder)
        cleaned_folder_name = folder_name.replace("OK", "").replace(" ", "")

        # 获取模板文件名称（不包含扩展名）
        template_file_name = os.path.splitext(os.path.basename(template_path))[0]

        # 生成新的文件名
        output_file_name = f"{template_file_name}-{folder_name}.xlsx"
        output_path = os.path.join(output_folder, output_file_name)

        # 获取文件夹中的图片文件
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        # 创建新的工作簿或加载模板
        if os.path.exists(template_path):
            workbook = load_workbook(template_path)
        else:
            workbook = Workbook()

        sheet = workbook.active
        sheet['E4'] = cleaned_folder_name

        row_insert_positions = [56, 67, 79]  # 第一、第二、第三行的起始位置
        max_images_per_row = 4  # 每行最多4张图片

        # 根据图片数量来确定排版
        if len(image_files) == 9:
            max_images_per_row = 3  # 每行放3张
        elif len(image_files) == 10:
            max_images_per_row = 4  # 第一行4张，后两行3张
        elif len(image_files) == 11:
            max_images_per_row = 4  # 前两行4张，最后一行3张
        elif len(image_files) == 12:
            max_images_per_row = 4  # 每行4张

        # 插入图片
        for i, image_file in enumerate(image_files):
            if len(image_files) == 9:
                # 9张图片：每行3张，行数为3
                if i < 3:
                    current_row = row_insert_positions[0]
                elif i < 6:
                    current_row = row_insert_positions[1]
                else:
                    current_row = row_insert_positions[2]

                col = 2 + (i % 3)  # 每行3列
            elif len(image_files) == 10:
                # 10张图片：第一行4张，后两行3张
                if i < 4:
                    current_row = row_insert_positions[0]
                    col = 2 + (i % 4)  # 第一行4列
                elif i < 7:
                    current_row = row_insert_positions[1]
                    col = 2 + (i - 4)  # 第二行3列
                else:
                    current_row = row_insert_positions[2]
                    col = 2 + (i - 7)  # 第三行3列
            elif len(image_files) == 11:
                # 11张图片：前两行4张，第三行3张
                if i < 4:
                    current_row = row_insert_positions[0]
                    col = 2 + (i % 4)  # 第一行4列
                elif i < 8:
                    current_row = row_insert_positions[1]
                    col = 2 + (i - 4)  # 第二行4列
                else:
                    current_row = row_insert_positions[2]
                    col = 2 + (i - 8)  # 第三行3列
            elif len(image_files) == 12:
                # 12张图片：每行4张
                if i < 4:
                    current_row = row_insert_positions[0]
                    col = 2 + (i % 4)  # 第一行4列
                elif i < 8:
                    current_row = row_insert_positions[1]
                    col = 2 + (i - 4)  # 第二行4列
                else:
                    current_row = row_insert_positions[2]
                    col = 2 + (i - 8)  # 第三行4列

            # 图片路径
            image_path = os.path.join(image_folder, image_file)
            try:
                with Image.open(image_path) as img:
                    max_width, max_height = aspect_ratios[i]
                    img.thumbnail((max_width, max_height), Image.LANCZOS)

                    temp_image_path = os.path.join(output_folder, f'temp_{image_file}')
                    img.save(temp_image_path)

                    excel_image = ExcelImage(temp_image_path)

                    excel_image.anchor = sheet.cell(row=current_row, column=col).coordinate
                    sheet.add_image(excel_image)

            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")

        workbook.save(output_path)

        # 删除临时图像文件
        for image_file in os.listdir(output_folder):
            if image_file.startswith('temp_'):
                os.remove(os.path.join(output_folder, image_file))

        print(f'报告已保存至 {output_path}')
