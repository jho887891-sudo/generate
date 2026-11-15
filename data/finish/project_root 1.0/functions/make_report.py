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
        output_file_name = f"PIV-QL_X-Ray测试报告-{folder_name}.xlsx"
        output_path = os.path.join(output_folder, output_file_name)

        # 创建新的工作簿或加载模板
        if os.path.exists(template_path):
            workbook = load_workbook(template_path)
        else:
            workbook = Workbook()

        sheet = workbook.active
        sheet['E4'] = cleaned_folder_name

        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        row_insert_positions = [56, 77, 79]
        max_images_per_row = 4

        # aspect_ratios = [
        #     (300, 240), (300, 240), (300, 240), (300, 240),
        #     (300, 240), (300, 240), (300, 240), (300, 240),
        #     (300, 220), (300, 240), (300, 240), (300, 240)
        # ]

        for i, image_file in enumerate(image_files):
            if i < max_images_per_row:
                current_row = row_insert_positions[0]
            elif i < max_images_per_row * 2:
                current_row = row_insert_positions[1]
            elif i < max_images_per_row * 3:
                current_row = row_insert_positions[2]
            else:
                current_row = row_insert_positions[2] + 2

            image_path = os.path.join(image_folder, image_file)
            with Image.open(image_path) as img:
                max_width, max_height = aspect_ratios[i]
                img.thumbnail((max_width, max_height), Image.LANCZOS)

                temp_image_path = os.path.join(output_folder, f'temp_{image_file}')
                img.save(temp_image_path)

                excel_image = ExcelImage(temp_image_path)

                if i < 4:
                    col = 2 + (i % max_images_per_row)
                    excel_image.anchor = sheet.cell(row=current_row, column=col).coordinate
                elif 4 <= i < 8:
                    col = 2 + (i - 4)
                    excel_image.anchor = sheet.cell(row=68, column=col).coordinate
                else:
                    col = 2 + (i - 8)
                    excel_image.anchor = sheet.cell(row=80, column=col).coordinate

                sheet.add_image(excel_image)

        workbook.save(output_path)

        for image_file in os.listdir(output_folder):
            if image_file.startswith('temp_'):
                os.remove(os.path.join(output_folder, image_file))

    print(f'报告已保存至 {output_path}')
