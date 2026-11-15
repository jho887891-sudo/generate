import pandas as pd
import os
import re


def negative_quantity(input_path, output_dir):
    """
    将Excel文件中的"数量"列的值取负（乘以-1），并增强数据清洗功能

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
        # 检查是否存在"数量"列
        if '数量' in df.columns:
            # 处理每一行
            for idx in range(len(df)):
                value = df.at[idx, '数量']

                # 跳过空值
                if pd.isna(value):
                    continue

                # 尝试转换为数值类型
                try:
                    # 如果是数值类型直接取负
                    num = float(value)
                    df.at[idx, '数量'] = -num
                except (ValueError, TypeError):
                    # 如果是字符串类型，尝试提取数字
                    if isinstance(value, str):
                        # 清理字符串 - 移除非数字字符（保留小数点和负号）
                        cleaned = re.sub(r'[^\d.-]', '', value)

                        # 处理多余的小数点
                        if cleaned.count('.') > 1:
                            parts = cleaned.split('.')
                            cleaned = parts[0] + '.' + ''.join(parts[1:])

                        # 尝试转换为浮点数
                        try:
                            num = float(cleaned)
                            df.at[idx, '数量'] = -num
                        except (ValueError, TypeError):
                            # 如果转换失败，保留原值
                            continue
                    else:
                        # 其他类型无法处理，保留原值
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