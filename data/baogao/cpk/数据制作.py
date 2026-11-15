import os
import pandas as pd
from openpyxl import load_workbook

def read_database_excel(file_path):
    """
    读取数据库 Excel 文件，从 F 到 DZ 列获取数据，同时也读取 C 和 D 列
    """
    # 读取数据库 Excel（从 C 到 D 列以及从 F 到 DZ 列）
    df = pd.read_excel(file_path, usecols="C,D,F:DZ", header=None)  # 读取 C, D 列和 F 到 DZ 列
    return df

def paste_to_template_excel(df, template_file, output_file, row_index):
    """
    将拆分后的数据粘贴到模板 Excel 中的指定位置
    """
    # 加载已有的模板 Excel 文件
    wb = load_workbook(template_file)
    ws = wb.active

    # 确定粘贴起始行和列（从 C31 开始）
    start_row = 31
    start_col = 3  # C 列是第 3 列

    # 选择需要的数据库行（注意行索引从 0 开始）
    row = df.iloc[row_index]

    # 获取 C 列和 D 列的值
    c_value = row[2]  # C 列的数据
    d_value = row[3]  # D 列的数据

    # 将 C 列和 D 列的数据粘贴到模板中的 M2 和 M3
    ws['M2'] = c_value
    ws['M3'] = d_value

    # 遍历拆分后的数据
    for j in range(0, 125, 25):  # 每行 25 个数据，共 5 行
        data_row = row[2 + j:2 + j + 25].tolist()  # 获取从 F 列开始的 25 个数据

        # 将拆分后的数据粘贴到模板 Excel 的对应位置
        current_row = start_row + (j // 25)  # 每行数据的位置
        for col_idx, value in enumerate(data_row, start=start_col):
            ws.cell(row=current_row, column=col_idx, value=value)

    # 保存修改后的 Excel 文件
    wb.save(output_file)
    print(f"数据已成功粘贴到模板 Excel: {output_file}")

def main():
    # 输入和输出路径
    database_file = 'data/database.xlsx'  # 数据库 Excel 文件路径
    template_file = 'templates/template1.xlsx'  # 模板 Excel 文件路径
    output_dir = 'outputs'  # 输出文件夹

    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取数据库 Excel 文件
    df = read_database_excel(database_file)

    # 询问用户选择处理的行范围
    user_input = input(f"请输入要处理的行号范围 (1 到 {len(df)}), 例如 '9-25': ")

    try:
        # 解析用户输入的行范围
        start_row, end_row = map(int, user_input.split('-'))

        # 检查行号范围的有效性
        if start_row < 1 or end_row > len(df) or start_row > end_row:
            print("无效的行号范围，请检查输入。")
            return

        # 遍历指定范围内的每一行，生成对应的输出文件
        for row_index in range(start_row - 1, end_row):
            # 生成对应的输出文件名
            output_file = os.path.join(output_dir, f"output_template_{row_index + 1}.xlsx")

            # 将数据粘贴到模板 Excel 文件
            paste_to_template_excel(df, template_file, output_file, row_index)

    except ValueError:
        print("输入格式错误，请使用 '起始行-结束行' 的格式，例如 '9-25'。")
        return

if __name__ == "__main__":
    main()
