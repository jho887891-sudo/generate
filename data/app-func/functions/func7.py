import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill
import re
from difflib import SequenceMatcher
import os
import datetime
import logging

# 设置日志
logger = logging.getLogger(__name__)


def similarity(a, b):
    """计算两个字符串的相似度"""
    if pd.isna(a) or pd.isna(b):
        return 0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def normalize_string(s):
    """标准化字符串，处理0和o的混淆"""
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    # 将O替换为0，处理可能的混淆
    s = s.replace('O', '0')
    return s


def create_excel_log(output_dir, timestamp, matched_records, mismatched_records, not_found_records):
    """创建Excel格式的日志文件"""
    log_excel_file = os.path.join(output_dir, f"采购收据对账日志_{timestamp}.xlsx")

    # 创建工作簿
    wb = openpyxl.Workbook()

    # 定义颜色填充
    red_fill = PatternFill(start_color='FFFF9999', end_color='FFFF9999', fill_type='solid')  # 浅红色
    blue_fill = PatternFill(start_color='FF99CCFF', end_color='FF99CCFF', fill_type='solid')  # 浅蓝色

    # 第一部分：数量不同的信息
    ws1 = wb.active
    ws1.title = "数量不同记录"

    # 添加标题行
    headers = ["页码", "行号", "采购单号", "物料编码", "原数量", "新数量", "采购单行号"]
    for col, header in enumerate(headers, 1):
        ws1.cell(row=1, column=col, value=header)

    # 添加数据行
    row_num = 2
    for record in mismatched_records:
        ws1.cell(row=row_num, column=1, value=record['sheet'])
        ws1.cell(row=row_num, column=2, value=record['row'])
        ws1.cell(row=row_num, column=3, value=record['po_number'])
        ws1.cell(row=row_num, column=4, value=record['material_code'])
        ws1.cell(row=row_num, column=5, value=record['old_quantity'])
        ws1.cell(row=row_num, column=6, value=record['new_quantity'])
        ws1.cell(row=row_num, column=7, value=record['delivery_row'])

        # 整行填充红色
        for col in range(1, 8):
            ws1.cell(row=row_num, column=col).fill = red_fill

        row_num += 1

    # 第二部分：已核对记录
    ws2 = wb.create_sheet("已核对记录")

    # 添加标题行
    for col, header in enumerate(headers[:6], 1):  # 只取前6列
        ws2.cell(row=1, column=col, value=header)

    # 添加数据行
    row_num = 2
    for record in matched_records:
        ws2.cell(row=row_num, column=1, value=record['sheet'])
        ws2.cell(row=row_num, column=2, value=record['row'])
        ws2.cell(row=row_num, column=3, value=record['po_number'])
        ws2.cell(row=row_num, column=4, value=record['material_code'])
        ws2.cell(row=row_num, column=5, value=record['quantity'])
        ws2.cell(row=row_num, column=6, value=record['quantity'])  # 新数量与原数量相同

        # 不填充颜色（默认）
        row_num += 1

    # 第三部分：未核对记录
    ws3 = wb.create_sheet("未核对记录")

    # 添加标题行
    for col, header in enumerate(headers[:5], 1):  # 只取前5列
        ws3.cell(row=1, column=col, value=header)

    # 添加数据行
    row_num = 2
    for record in not_found_records:
        ws3.cell(row=row_num, column=1, value=record['sheet'])
        ws3.cell(row=row_num, column=2, value=record['row'])
        ws3.cell(row=row_num, column=3, value=record['po_number'])
        ws3.cell(row=row_num, column=4, value=record['material_code'])
        ws3.cell(row=row_num, column=5, value=record['quantity'])

        # 整行填充蓝色
        for col in range(1, 6):
            ws3.cell(row=row_num, column=col).fill = blue_fill

        row_num += 1

    # 调整列宽
    for sheet in wb.worksheets:
        for column in sheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2) * 1.2
            sheet.column_dimensions[column_letter].width = min(adjusted_width, 50)

    # 保存Excel日志文件
    wb.save(log_excel_file)
    return log_excel_file


def reconcile_purchase_receipts(delivery_file, inventory_file, output_dir):
    """
    对采购单和入库单进行对账

    Args:
        delivery_file: 采购单文件路径
        inventory_file: 入库单文件路径
        output_dir: 输出目录

    Returns:
        tuple: (result_file, log_file) 结果文件和日志文件路径
    """
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"创建输出目录: {output_dir}")

    # 设置输出文件路径
    result_file = os.path.join(output_dir, f"采购收据对账结果_{timestamp}.xlsx")
    log_file = None

    # 检查输入文件是否存在
    if not os.path.exists(delivery_file):
        error_msg = f"错误: 采购单文件不存在: {delivery_file}"
        logger.error(error_msg)
        return None, None, error_msg

    if not os.path.exists(inventory_file):
        error_msg = f"错误: 入库单文件不存在: {inventory_file}"
        logger.error(error_msg)
        return None, None, error_msg

    try:
        # 读取两个Excel文件的所有sheet
        delivery_sheets = pd.read_excel(delivery_file, sheet_name=None)
        inventory_sheets = pd.read_excel(inventory_file, sheet_name=None)

        # 使用openpyxl加载入库单文件，用于修改样式
        inventory_wb = openpyxl.load_workbook(inventory_file)

        # 定义颜色填充 - 使用更浅的颜色
        red_fill = PatternFill(start_color='FFFF9999', end_color='FFFF9999', fill_type='solid')  # 浅红色
        blue_fill = PatternFill(start_color='FF99CCFF', end_color='FF99CCFF', fill_type='solid')  # 浅蓝色

        # 统计信息记录
        matched_records = []
        mismatched_records = []
        not_found_records = []
        error_records = []

        logger.info("开始采购收据对账处理...")
        logger.info(f"采购单文件: {delivery_file}")
        logger.info(f"入库单文件: {inventory_file}")
        logger.info(f"输出文件: {result_file}")

        # 处理每个页码对应的sheet
        for sheet_name in inventory_sheets.keys():
            if sheet_name not in delivery_sheets:
                logger.warning(f"警告: 采购单中没有找到对应的 {sheet_name}")
                continue

            logger.info(f"处理 {sheet_name}")

            # 获取当前sheet的数据
            delivery_df = delivery_sheets[sheet_name]
            inventory_df = inventory_sheets[sheet_name]
            inventory_ws = inventory_wb[sheet_name]

            # 处理采购单数据
            delivery_data = []
            for idx, row in delivery_df.iterrows():
                try:
                    # 采购单：E列采购单号，F列物料编号，I列数量
                    po_number = row.iloc[4] if len(row) > 4 else None  # E列
                    material_code = row.iloc[5] if len(row) > 5 else None  # F列
                    quantity = row.iloc[8] if len(row) > 8 else None  # I列

                    if pd.notna(po_number) and pd.notna(material_code) and pd.notna(quantity):
                        delivery_data.append({
                            'po_number': normalize_string(po_number),
                            'material_code': normalize_string(material_code),
                            'quantity': quantity,
                            'row_index': idx + 2  # 记录行号，用于日志
                        })
                    else:
                        logger.warning(f"警告: 采购单 {sheet_name} 第 {idx + 2} 行数据不完整")
                except (IndexError, ValueError) as e:
                    error_msg = f"采购单 {sheet_name} 第 {idx + 2} 行读取错误: {str(e)}"
                    logger.error(f"错误: {error_msg}")
                    error_records.append(error_msg)
                    continue

            # 处理入库单数据
            for idx, row in inventory_df.iterrows():
                try:
                    # 入库单：K列采购单号，E列料品编码，J列数量
                    po_number_inv = row.iloc[10] if len(row) > 10 else None  # K列
                    material_code_inv = row.iloc[4] if len(row) > 4 else None  # E列
                    quantity_inv = row.iloc[9] if len(row) > 9 else None  # J列

                    if pd.isna(po_number_inv) or pd.isna(material_code_inv) or pd.isna(quantity_inv):
                        logger.warning(f"警告: 入库单 {sheet_name} 第 {idx + 2} 行数据不完整")
                        continue

                    po_norm_inv = normalize_string(po_number_inv)
                    material_norm_inv = normalize_string(material_code_inv)

                    # 在采购单中查找匹配项
                    matched = False
                    for delivery_item in delivery_data:
                        # 使用相似度匹配，处理可能的拼写错误
                        po_similarity = similarity(po_norm_inv, delivery_item['po_number'])
                        material_similarity = similarity(material_norm_inv, delivery_item['material_code'])

                        # 如果相似度超过阈值，认为匹配
                        if po_similarity > 0.8 and material_similarity > 0.8:
                            matched = True

                            # 计算数量所在的单元格位置
                            quantity_cell_col = 10  # J列
                            quantity_cell_row = idx + 2  # 行号（openpyxl从1开始，加上标题行）

                            # 获取数量单元格
                            quantity_cell = inventory_ws.cell(row=quantity_cell_row, column=quantity_cell_col)

                            if delivery_item['quantity'] != quantity_inv:
                                # 数量不符，用采购单数量覆盖并标记红色
                                old_value = quantity_inv
                                new_value = delivery_item['quantity']
                                quantity_cell.value = new_value
                                quantity_cell.fill = red_fill

                                # 记录不匹配信息
                                mismatch_info = {
                                    'sheet': sheet_name,
                                    'row': idx + 2,
                                    'po_number': po_number_inv,
                                    'material_code': material_code_inv,
                                    'old_quantity': old_value,
                                    'new_quantity': new_value,
                                    'delivery_row': delivery_item['row_index']
                                }
                                mismatched_records.append(mismatch_info)

                                logger.info(f"数量不符 - {sheet_name} 第 {idx + 2} 行: "
                                            f"PO={po_number_inv}, 物料={material_code_inv}, "
                                            f"入库数量: {old_value} → 送货数量: {new_value}")
                            else:
                                # 数量相同，标记浅蓝色
                                quantity_cell.fill = blue_fill

                                # 记录匹配信息
                                match_info = {
                                    'sheet': sheet_name,
                                    'row': idx + 2,
                                    'po_number': po_number_inv,
                                    'material_code': material_code_inv,
                                    'quantity': quantity_inv
                                }
                                matched_records.append(match_info)

                                logger.info(f"数量一致 - {sheet_name} 第 {idx + 2} 行: "
                                            f"PO={po_number_inv}, 物料={material_code_inv}, 数量: {quantity_inv}")

                            break

                    if not matched:
                        # 记录未找到匹配的信息
                        not_found_info = {
                            'sheet': sheet_name,
                            'row': idx + 2,
                            'po_number': po_number_inv,
                            'material_code': material_code_inv,
                            'quantity': quantity_inv
                        }
                        not_found_records.append(not_found_info)

                        logger.info(f"未找到匹配 - {sheet_name} 第 {idx + 2} 行: "
                                    f"PO={po_number_inv}, 物料={material_code_inv}")

                except (IndexError, ValueError) as e:
                    error_msg = f"入库单 {sheet_name} 第 {idx + 2} 行处理错误: {str(e)}"
                    logger.error(f"错误: {error_msg}")
                    error_records.append(error_msg)
                    continue

        # 保存修改后的入库单
        inventory_wb.save(result_file)

        # 创建Excel日志文件
        log_file = create_excel_log(output_dir, timestamp, matched_records, mismatched_records, not_found_records)

        # 生成统计报告
        logger.info("\n" + "=" * 50)
        logger.info("采购收据对账完成!")
        logger.info("=" * 50)

        # 统计报告
        logger.info(f"统计结果:")
        logger.info(f"  - 数量一致: {len(matched_records)} 条")
        logger.info(f"  - 数量不符: {len(mismatched_records)} 条")
        logger.info(f"  - 未找到匹配: {len(not_found_records)} 条")
        logger.info(f"  - 处理错误: {len(error_records)} 条")

        logger.info(f"\n结果文件: {result_file}")
        logger.info(f"Excel日志文件: {log_file}")

        return result_file, log_file, None

    except Exception as e:
        error_msg = f"处理过程中出现错误: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return None, None, error_msg