import pandas as pd
import os

# 读取Excel文件
excel_file = 'output-Excel/delivery_summary_6.xlsx'
all_sheets = pd.read_excel(excel_file, sheet_name=None)

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
original_filename = os.path.basename(excel_file)
filename, ext = os.path.splitext(original_filename)
new_filename = f"{filename}_Negative_number{ext}"

# 保持输出路径不变
output_path = 'output -/'  # 注意：这个路径有空格，确保目录存在
output_file = os.path.join(output_path, new_filename)

# 保存修改后的文件
with pd.ExcelWriter(output_file) as writer:
    for sheet_name, df in all_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"处理完成，已保存为: {output_file}")