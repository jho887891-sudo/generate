import pandas as pd
import os


def negative_quantity(input_path, output_dir):
    """
    将Excel文件中的"数量"列的值取负（乘以-1）

    参数:
        input_path (str): 输入的Excel文件路径
        output_dir (str): 输出目录路径

    返回:
        str: 处理后的文件路径
    """
    # 读取Excel文件
    all_sheets = pd.read_excel(input_path, sheet_name=None)

    # 处理每个工作表
    for sheet_name, df in all_sheets.items():
        # 获取"数量"列的位置
        if '数量' in df.columns:
            # 处理每一行（跳过非数字值）
            for idx in range(len(df)):
                # 尝试将值转为数字
                try:
                    num = float(df.at[idx, '数量'])
                    df.at[idx, '数量'] = -num
                except (ValueError, TypeError):
                    # 转换失败则保持原值
                    continue

    # 构建新文件名（添加Negative_number后缀）
    original_filename = os.path.basename(input_path)
    filename, ext = os.path.splitext(original_filename)
    new_filename = f"{filename}_Negative_number{ext}"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, new_filename)

    # 保存修改后的文件
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return output_file