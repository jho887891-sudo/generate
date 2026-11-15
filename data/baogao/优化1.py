import os
import datetime
from PIL import Image
from openpyxl import Workbook
from openpyxl import load_workbook
from openpyxl.drawing.image import Image as ExcelImage

# 设置图片根文件夹路径
base_image_folder = '10.10VK3'  # 替换为你的图片根文件夹路径

# 获取所有包含"OK"的子文件夹路径，并排除包含"NG"的文件夹
image_folders = [
    os.path.join(base_image_folder, folder)
    for folder in os.listdir(base_image_folder)
    if os.path.isdir(os.path.join(base_image_folder, folder)) and "OK" in folder and "NG" not in folder
]

# Excel模板路径
template_path = 'doocx/image4.xlsx'  # 替换为你的Excel模板文件路径

# 创建输出文件夹
output_folder = 'output'  # 输出文件夹名称
if not os.path.exists(output_folder):  # 检查输出文件夹是否存在
    os.makedirs(output_folder)  # 如果不存在，则创建输出文件夹

# 定义每个文件夹生成的报告
for image_folder in image_folders:
    folder_name = os.path.basename(image_folder)  # 获取文件夹名称作为报告名的一部分
    cleaned_folder_name = folder_name.replace("OK", "").replace(" ", "")  # 去掉 "OK" 和空格
    output_file_name = f"PIV-QL_X-Ray测试报告-{folder_name}.xlsx"  # 修改输出文件名
    output_path = os.path.join(output_folder, output_file_name)  # 拼接输出文件的完整路径

    # 创建新的工作簿或加载模板
    if os.path.exists(template_path):  # 检查模板文件是否存在
        workbook = load_workbook(template_path)  # 从模板加载工作簿
    else:
        workbook = Workbook()  # 如果模板不存在，创建一个新的工作簿

    # 选择默认工作表
    sheet = workbook.active  # 选择活动工作表

    # 在 E4 单元格中插入处理后的 folder_name
    sheet['E4'] = cleaned_folder_name  # 设置 E4 单元格的值

    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]  # 获取所有图片文件

    # 假设每张图片插入到文档中的固定位置，这里可以自定义位置
    row_insert_positions = [56, 77, 79]  # 定义行位置
    max_images_per_row = 4  # 每行最多放置4张图片

    # 按位置插入图片
    for i, image_file in enumerate(image_files):
        # 确定当前行
        if i < max_images_per_row:  # 前四张图片到第一行
            current_row = row_insert_positions[0]
        elif i < max_images_per_row * 2:  # 第二组四张图片到第二行
            current_row = row_insert_positions[1]
        elif i < max_images_per_row * 3:  # 第五到第八张图片到第三行
            current_row = row_insert_positions[2]
        elif i < max_images_per_row * 4:  # 第九到第十二张图片到第四行
            current_row = row_insert_positions[2] + 2  # 设置为79行 (79 + 2 = 81行)，这里可以直接使用79
        else:
            break  # 超过已有图片数量时，退出循环

        # 加载和缩放图片
        image_path = os.path.join(image_folder, image_file)  # 获取图片的完整路径
        with Image.open(image_path) as img:  # 打开图片文件
            # 如果是第九张图片，使用特殊的缩放比例
            if i == 0:  # 第九张图片（索引为8）
                max_width = 300  # 最大宽度
                max_height = 220  # 最大高度
            else:
                max_width = 300  # 其他图片的最大宽度
                max_height = 240  # 其他图片的最大高度

            # 计算缩放比例
            img.thumbnail((max_width, max_height), Image.LANCZOS)  # 使用Image.LANCZOS进行缩放

            # 将缩放后的图片保存为临时文件
            temp_image_path = os.path.join(output_folder, f'temp_{image_file}')  # 临时文件路径
            img.save(temp_image_path)  # 保存临时文件

            # 创建ExcelImage对象
            excel_image = ExcelImage(temp_image_path)  # 创建ExcelImage对象

            # 根据图片序号设置列位置
            if i < 4:  # 前四张图片到第一行
                col = 2 + (i % max_images_per_row)  # 列数从A列开始
                excel_image.anchor = sheet.cell(row=current_row, column=col).coordinate  # 将图片锚定到指定单元格
            elif 4 <= i < 8:  # 第五到第八张图片插入到67行
                col = 2 + (i - 4)  # 列数从A列开始
                excel_image.anchor = sheet.cell(row=68, column=col).coordinate  # 锚定到67行
            elif 8 <= i < 12:  # 第九到第十二张图片插入到79行
                col = 2 + (i - 8)  # 列数从A列开始
                excel_image.anchor = sheet.cell(row=80, column=col).coordinate  # 锚定到79行

            # 添加图片到Excel表格
            sheet.add_image(excel_image)  # 将图片添加到工作表

    # 保存新的Excel文档
    workbook.save(output_path)  # 保存工作簿到输出路径

    # 可选：删除临时图片文件
    for image_file in os.listdir(output_folder):  # 遍历输出文件夹
        if image_file.startswith('temp_'):  # 检查文件名是否以'temp_'开头
            os.remove(os.path.join(output_folder, image_file))  # 删除临时文件

    print(f'报告已保存至 {output_path}')  # 打印保存路径
