import openpyxl
from openpyxl.styles import PatternFill
import pandas as pd
import os
import csv
from datetime import datetime
import logging
import re

# 蓝色填充 - 表示已核对
highlight = PatternFill(start_color="ADD8E6", end_color="ADD8E6", fill_type="solid")
# 红色填充 - 表示数量不匹配
red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")

logger = logging.getLogger(__name__)


def is_alphanumeric(s):
    """检查字符串是否只包含字母和数字"""
    return all(c.isalnum() for c in s)


def clean_quantity(value):
    """清理数量值，只保留数字和小数点"""
    if value is None:
        return None

    # 如果是数值类型直接返回
    if isinstance(value, (int, float)):
        return value

    # 处理字符串类型
    s = str(value).strip()

    # 如果包含单位（如"270 PCS"），先提取数字部分
    if ' ' in s:
        parts = s.split(' ')
        if parts and parts[0]:
            s = parts[0]  # 取空格前的部分作为数量

    # 移除非数字字符（保留小数点和负号）
    cleaned = re.sub(r'[^\d.-]', '', s)

    # 如果清理后为空，返回None
    if not cleaned:
        return None

    # 处理多余的小数点
    if cleaned.count('.') > 1:
        parts = cleaned.split('.')
        cleaned = parts[0] + '.' + ''.join(parts[1:])

    try:
        # 转换为浮点数
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def extract_document_number(text):
    """从文本中提取单据号"""
    if text is None:
        return ""

    text = str(text).strip()

    # 尝试匹配常见的单据号格式
    patterns = [
        r'[A-Za-z]{2,}\d{5,}[A-Za-z]?-\d{3}',  # 类似 SZJGD2025102A-003
        r'[A-Za-z]{2,}\d{5,}[A-Za-z]?',  # 类似 SZJGD202514583067
        r'[A-Za-z]{2,}\d{5,}',  # 类似 SZ202508250070
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]

    # 如果没有匹配到特定格式，返回原始文本
    return text


def clean_document_number(doc_no):
    """清理单据号，移除空格和特殊字符，保留字母和数字"""
    if doc_no is None:
        return ""

    # 先尝试提取单据号
    extracted = extract_document_number(doc_no)
    if extracted:
        doc_no = extracted

    # 转换为字符串并移除空格
    cleaned = str(doc_no).strip()

    # 修正正则表达式错误：移除\a
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', cleaned)

    return cleaned


def preprocess_receipt(ws, log_list, receipt_file):
    """预处理入库单：合并行并清理数量列"""
    row = 2
    merge_count = 0

    while row <= ws.max_row:
        # 清理L列数量值
        l_val = ws[f"L{row}"].value
        cleaned_val = clean_quantity(l_val)
        if l_val != cleaned_val and cleaned_val is not None:
            log_list.append({
                "类型": "数据清理",
                "入库单文件": receipt_file,
                "入库单工作表": ws.title,
                "入库单行号": row,
                "详情": f"第{row}行L列: 从'{l_val}'清理为'{cleaned_val}'"
            })
            ws[f"L{row}"].value = cleaned_val

        # 检查当前行F列的值
        f_val = str(ws[f"F{row}"].value).strip() if ws[f"F{row}"].value else ""

        # 如果当前行为空，跳过
        if not f_val:
            row += 1
            continue

        # 检查是否只包含字母和数字
        if not is_alphanumeric(f_val):
            row += 1
            continue

        # 检查长度是否小于17且大于0
        if 0 < len(f_val) < 17 and row + 1 <= ws.max_row:
            # 检查下一行
            next_f_val = str(ws[f"F{row + 1}"].value).strip() if ws[f"F{row + 1}"].value else ""

            # 检查下一行是否只包含字母和数字
            if next_f_val and is_alphanumeric(next_f_val):
                # 检查合并后长度是否为17
                if len(f_val) + len(next_f_val) == 17:
                    # 合并F列的值
                    merged_value = f_val + next_f_val
                    ws[f"F{row}"] = merged_value

                    # 清除下一行的F列数据
                    ws[f"F{row + 1}"] = None

                    merge_count += 1

                    log_list.append({
                        "类型": "数据合并",
                        "入库单文件": receipt_file,
                        "入库单工作表": ws.title,
                        "入库单行号": row,
                        "详情": f"第{row}行和第{row + 1}行F列合并，合并后值: {merged_value}"
                    })

                    # 向上移动后续的F列数据填补空缺
                    move_f_column_up(ws, row + 1, log_list, receipt_file)

                    # 继续检查当前行，因为可能有连续的合并
                    continue

        row += 1

    return merge_count


def move_f_column_up(ws, empty_row, log_list, receipt_file):
    """向上移动F列数据填补空缺"""
    for row in range(empty_row, ws.max_row):
        next_f_val = ws[f"F{row + 1}"].value
        ws[f"F{row}"] = next_f_val
        ws[f"F{row + 1}"] = None

        if next_f_val:
            log_list.append({
                "类型": "数据移动",
                "入库单文件": receipt_file,
                "入库单工作表": ws.title,
                "入库单行号": row,
                "详情": f"第{row + 1}行F列数据移动到第{row}行F列"
            })


def reconcile_all_sheets(receipt_wb, receipt_file, total_wb, total_file, log_list):
    """对账逻辑：核对入库单和送货单的所有工作表"""
    global_delivery_index = {}

    for total_sheet_name in total_wb.sheetnames:
        total_ws = total_wb[total_sheet_name]
        for t in range(2, total_ws.max_row + 1):
            # 单据号在F列
            total_f = str(total_ws[f"F{t}"].value).strip() if total_ws[f"F{t}"].value else ""

            # 料品编码在I列
            total_g = str(total_ws[f"I{t}"].value).strip() if total_ws[f"I{t}"].value else ""

            total_l = total_ws[f"L{t}"].value if total_ws[f"L{t}"].value is not None else ""

            if total_f and total_g:
                cleaned_total_f = clean_document_number(total_f)
                if len(cleaned_total_f) >= 4:
                    last_four = cleaned_total_f[-4:]
                    cleaned_total_g = clean_document_number(total_g)
                    if cleaned_total_g:
                        key = (last_four, cleaned_total_g)
                        if key not in global_delivery_index:
                            global_delivery_index[key] = []
                        global_delivery_index[key].append({
                            "sheet": total_sheet_name,
                            "row": t,
                            "doc_no": total_f,
                            "cleaned_doc_no": cleaned_total_f,
                            "material_code": total_g,
                            "cleaned_material_code": cleaned_total_g,
                            "quantity": total_l
                        })

    # 记录送货单索引大小用于调试
    log_list.append({
        "类型": "调试信息",
        "入库单文件": receipt_file,
        "送货单文件": total_file,
        "详情": f"送货单全局索引构建完成，共{len(global_delivery_index)}个键，{sum(len(v) for v in global_delivery_index.values())}条记录"
    })

    return global_delivery_index


def reconcile_single_sheet(receipt_ws, receipt_sheet_name, receipt_file,
                           global_delivery_index, total_file, log_list):
    """对账逻辑：核对单个工作表的入库单和送货单（使用全局送货单索引）"""
    matched_count = 0
    unmatched_count = 0

    for r in range(2, receipt_ws.max_row + 1):
        receipt_f = str(receipt_ws[f"F{r}"].value).strip() if receipt_ws[f"F{r}"].value else ""
        receipt_g = str(receipt_ws[f"G{r}"].value).strip() if receipt_ws[f"G{r}"].value else ""  # 料品编码
        receipt_l = receipt_ws[f"L{r}"].value

        # 检查必要的数据是否存在
        if not receipt_f:
            log_list.append({
                "类型": "未对账",
                "入库单文件": receipt_file,
                "入库单工作表": receipt_sheet_name,
                "入库单行号": r,
                "详情": f"第{r}行: 单据号为空，无法对账"
            })
            unmatched_count += 1
            continue

        if not receipt_g:
            log_list.append({
                "类型": "未对账",
                "入库单文件": receipt_file,
                "入库单工作表": receipt_sheet_name,
                "入库单行号": r,
                "详情": f"第{r}行: 料品编码为空，无法对账 (单据号: {receipt_f})"
            })
            unmatched_count += 1
            continue

        if receipt_l is None:
            log_list.append({
                "类型": "未对账",
                "入库单文件": receipt_file,
                "入库单工作表": receipt_sheet_name,
                "入库单行号": r,
                "详情": f"第{r}行: 数量为空，无法对账 (单据号: {receipt_f}, 料品编码: {receipt_g})"
            })
            unmatched_count += 1
            continue

        # 清理入库单单据号
        cleaned_receipt_f = clean_document_number(receipt_f)
        if len(cleaned_receipt_f) < 4:
            log_list.append({
                "类型": "未对账",
                "入库单文件": receipt_file,
                "入库单工作表": receipt_sheet_name,
                "入库单行号": r,
                "详情": f"第{r}行: 单据号'{receipt_f}'清理后为'{cleaned_receipt_f}'，长度不足4位，无法对账"
            })
            unmatched_count += 1
            continue

        # 清理料品编码
        cleaned_receipt_g = clean_document_number(receipt_g)
        if not cleaned_receipt_g:
            log_list.append({
                "类型": "未对账",
                "入库单文件": receipt_file,
                "入库单工作表": receipt_sheet_name,
                "入库单行号": r,
                "详情": f"第{r}行: 料品编码'{receipt_g}'清理后为空，无法对账"
            })
            unmatched_count += 1
            continue

        receipt_key = (cleaned_receipt_f[-4:], cleaned_receipt_g)
        found_match = False

        # 在全局送货单索引中查找
        if receipt_key in global_delivery_index:
            # 可能有多个匹配项
            for delivery_item in global_delivery_index[receipt_key]:
                t = delivery_item["row"]
                total_sheet_name = delivery_item["sheet"]
                total_f = delivery_item["doc_no"]
                total_g = delivery_item["material_code"]
                total_l = delivery_item["quantity"]
                cleaned_total_f = delivery_item["cleaned_doc_no"]

                found_match = True

                # 清理数量值
                receipt_l_cleaned = clean_quantity(receipt_l)
                total_l_cleaned = clean_quantity(total_l)

                # 比较数量
                try:
                    receipt_num = float(receipt_l_cleaned) if receipt_l_cleaned is not None else None
                    total_num = float(total_l_cleaned) if total_l_cleaned is not None else None

                    if receipt_num is not None and total_num is not None:
                        if abs(receipt_num - total_num) > 0.001:
                            receipt_ws[f"L{r}"].fill = red_fill
                            log_list.append({
                                "类型": "数量不匹配",
                                "入库单文件": receipt_file,
                                "入库单工作表": receipt_sheet_name,
                                "入库单行号": r,
                                "送货单文件": total_file,
                                "送货单工作表": total_sheet_name,
                                "送货单行号": t,
                                "入库单数量": receipt_l,
                                "送货单数量": total_l,
                                "详情": f"入库单{receipt_sheet_name}第{r}行与送货单{total_sheet_name}第{t}行数量不匹配: {receipt_l_cleaned} vs {total_l_cleaned}"
                            })
                        else:
                            receipt_ws[f"L{r}"].fill = highlight
                            log_list.append({
                                "类型": "对账成功",
                                "入库单文件": receipt_file,
                                "入库单工作表": receipt_sheet_name,
                                "入库单行号": r,
                                "送货单文件": total_file,
                                "送货单工作表": total_sheet_name,
                                "送货单行号": t,
                                "入库单数量": receipt_l,
                                "送货单数量": total_l,
                                "详情": f"单据号{cleaned_receipt_f}和料品编码{cleaned_receipt_g}核对成功"
                            })
                    else:
                        receipt_ws[f"L{r}"].fill = red_fill
                        log_list.append({
                            "类型": "数量错误",
                            "入库单文件": receipt_file,
                            "入库单工作表": receipt_sheet_name,
                            "入库单行号": r,
                            "送货单文件": total_file,
                            "送货单工作表": total_sheet_name,
                            "送货单行号": t,
                            "入库单数量": receipt_l,
                            "送货单数量": total_l,
                            "详情": f"无法比较数量值: {receipt_l_cleaned} vs {total_l_cleaned}"
                        })
                except (ValueError, TypeError):
                    receipt_ws[f"L{r}"].fill = red_fill
                    log_list.append({
                        "类型": "数量错误",
                        "入库单文件": receipt_file,
                        "入库单工作表": receipt_sheet_name,
                        "入库单行号": r,
                        "送货单文件": total_file,
                        "送货单工作表": total_sheet_name,
                        "送货单行号": t,
                        "入库单数量": receipt_l,
                        "送货单数量": total_l,
                        "详情": f"数量值格式错误: {receipt_l_cleaned} vs {total_l_cleaned}"
                    })

                matched_count += 1
                break

        if not found_match:
            # 详细记录未匹配信息，便于调试
            log_list.append({
                "类型": "未对账-详细",
                "入库单文件": receipt_file,
                "入库单工作表": receipt_sheet_name,
                "入库单行号": r,
                "入库单单据号": receipt_f,
                "清理后单据号": cleaned_receipt_f,
                "入库单料品编码": receipt_g,
                "清理后料品编码": cleaned_receipt_g,
                "搜索键": f"({cleaned_receipt_f[-4:]}, {cleaned_receipt_g})",
                "详情": f"第{r}行: 在送货单中未找到匹配项。原始单据号: {receipt_f}, 料品编码: {receipt_g}"
            })
            unmatched_count += 1

    return matched_count, unmatched_count


def create_excel_log(output_dir, timestamp, matched_records, mismatched_records, not_found_records, log_list,
                     input_receipt_path, input_total_path):
    """创建Excel格式的日志文件"""
    log_excel_file = os.path.join(output_dir, f"送货单对账日志_{timestamp}.xlsx")

    # 创建工作簿
    wb = openpyxl.Workbook()

    # 定义颜色填充
    red_fill = PatternFill(start_color='FFFF9999', end_color='FFFF9999', fill_type='solid')  # 浅红色
    blue_fill = PatternFill(start_color='FF99CCFF', end_color='FF99CCFF', fill_type='solid')  # 浅蓝色
    yellow_fill = PatternFill(start_color='FFFFFF99', end_color='FFFFFF99', fill_type='solid')  # 浅黄色

    # 第一部分：数量不同的信息
    ws1 = wb.active
    ws1.title = "数量不同记录"

    # 添加标题行
    headers = ["入库单工作表", "入库单行号", "送货单工作表", "送货单行号", "单据号", "料品编码", "入库单数量",
               "送货单数量", "原因"]
    for col, header in enumerate(headers, 1):
        ws1.cell(row=1, column=col, value=header)

    # 添加数据行
    row_num = 2
    for record in mismatched_records:
        ws1.cell(row=row_num, column=1, value=record.get('receipt_sheet', ''))
        ws1.cell(row=row_num, column=2, value=record.get('receipt_row', ''))
        ws1.cell(row=row_num, column=3, value=record.get('delivery_sheet', ''))
        ws1.cell(row=row_num, column=4, value=record.get('delivery_row', ''))
        ws1.cell(row=row_num, column=5, value=record.get('doc_no', ''))
        ws1.cell(row=row_num, column=6, value=record.get('material_code', ''))
        ws1.cell(row=row_num, column=7, value=record.get('receipt_quantity', ''))
        ws1.cell(row=row_num, column=8, value=record.get('delivery_quantity', ''))
        ws1.cell(row=row_num, column=9, value=record.get('reason', '数量不匹配'))

        # 整行填充红色
        for col in range(1, 10):
            ws1.cell(row=row_num, column=col).fill = red_fill

        row_num += 1

    # 第二部分：已核对记录
    ws2 = wb.create_sheet("已核对记录")

    # 添加标题行
    headers = ["入库单工作表", "入库单行号", "送货单工作表", "送货单行号", "单据号", "料品编码", "数量", "状态"]
    for col, header in enumerate(headers, 1):
        ws2.cell(row=1, column=col, value=header)

    # 添加数据行
    row_num = 2
    for record in matched_records:
        ws2.cell(row=row_num, column=1, value=record.get('receipt_sheet', ''))
        ws2.cell(row=row_num, column=2, value=record.get('receipt_row', ''))
        ws2.cell(row=row_num, column=3, value=record.get('delivery_sheet', ''))
        ws2.cell(row=row_num, column=4, value=record.get('delivery_row', ''))
        ws2.cell(row=row_num, column=5, value=record.get('doc_no', ''))
        ws2.cell(row=row_num, column=6, value=record.get('material_code', ''))
        ws2.cell(row=row_num, column=7, value=record.get('quantity', ''))
        ws2.cell(row=row_num, column=8, value="已核对")

        # 不填充颜色（默认）
        row_num += 1

    # 第三部分：未核对记录
    ws3 = wb.create_sheet("未核对记录")

    # 添加标题行
    headers = ["入库单工作表", "入库单行号", "单据号", "料品编码", "数量", "原因"]
    for col, header in enumerate(headers, 1):
        ws3.cell(row=1, column=col, value=header)

    # 添加数据行
    row_num = 2
    for record in not_found_records:
        ws3.cell(row=row_num, column=1, value=record.get('receipt_sheet', ''))
        ws3.cell(row=row_num, column=2, value=record.get('receipt_row', ''))
        ws3.cell(row=row_num, column=3, value=record.get('doc_no', ''))
        ws3.cell(row=row_num, column=4, value=record.get('material_code', ''))
        ws3.cell(row=row_num, column=5, value=record.get('quantity', ''))
        ws3.cell(row=row_num, column=6, value=record.get('reason', '未找到匹配'))

        # 整行填充蓝色
        for col in range(1, 7):
            ws3.cell(row=row_num, column=col).fill = blue_fill

        row_num += 1

    # 第四部分：数据修改记录
    ws4 = wb.create_sheet("数据修改记录")

    # 添加标题行
    headers = ["工作表", "行号", "列", "原值", "修改后值", "修改类型"]
    for col, header in enumerate(headers, 1):
        ws4.cell(row=1, column=col, value=header)

    # 从全局log_list中提取数据修改记录
    data_modifications = [log for log in log_list if log.get("类型") in ["数据清理", "数据合并", "数据移动"]]
    row_num = 2
    for record in data_modifications:
        ws4.cell(row=row_num, column=1, value=record.get('入库单工作表', ''))
        ws4.cell(row=row_num, column=2, value=record.get('入库单行号', ''))

        # 判断是哪个列被修改
        detail = record.get('详情', '')
        col_name = '未知'
        if 'L列' in detail:
            col_name = 'L列'
        elif 'F列' in detail:
            col_name = 'F列'

        ws4.cell(row=row_num, column=3, value=col_name)

        # 提取原值和修改后值
        old_value = ''
        new_value = ''
        if "从'" in detail and "清理为'" in detail:
            old_value = detail.split("从'")[1].split("'")[0] if "从'" in detail else ''
            new_value = detail.split("清理为'")[1].split("'")[0] if "清理为'" in detail else ''
        elif "合并后值" in detail:
            old_value = "合并前数据"
            new_value = detail.split("合并后值: ")[1] if "合并后值: " in detail else ''
        elif "移动到" in detail:
            old_value = "移动前数据"
            new_value = "移动后数据"

        ws4.cell(row=row_num, column=4, value=old_value)
        ws4.cell(row=row_num, column=5, value=new_value)
        ws4.cell(row=row_num, column=6, value=record.get('类型', ''))

        # 整行填充黄色
        for col in range(1, 7):
            ws4.cell(row=row_num, column=col).fill = yellow_fill

        row_num += 1

    # 第五部分：对账摘要
    ws5 = wb.create_sheet("对账摘要")

    # 添加摘要信息
    total_records = len(matched_records) + len(mismatched_records) + len(not_found_records)
    success_rate = (len(matched_records) / total_records * 100) if total_records > 0 else 0

    summary_data = [
        ["对账摘要", ""],
        ["入库单文件", os.path.basename(input_receipt_path)],
        ["送货单文件", os.path.basename(input_total_path)],
        ["对账时间", timestamp],
        ["", ""],
        ["对账结果统计", ""],
        ["已核对记录", len(matched_records)],
        ["数量不匹配记录", len(mismatched_records)],
        ["未核对记录", len(not_found_records)],
        ["数据修改记录", len(data_modifications)],
        ["", ""],
        ["总记录数", total_records],
        ["对账成功率", f"{success_rate:.2f}%"]
    ]

    for row, data in enumerate(summary_data, 1):
        ws5.cell(row=row, column=1, value=data[0])
        ws5.cell(row=row, column=2, value=data[1])

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


def reconcile_delivery_files(receipt_path, total_path, output_dir):
    """
    执行送货单对账功能 - 适配网页接口

    参数:
        receipt_path: 入库单文件路径
        total_path: 送货单文件路径
        output_dir: 输出目录

    返回:
        result_file: 对账结果文件路径（入库单）
        log_file: 日志文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建处理后的入库单目录
    processed_receipt_dir = os.path.join(output_dir, "processed_receipts")
    os.makedirs(processed_receipt_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 对账结果文件路径（入库单）
    result_file = os.path.join(output_dir, f"送货单对账结果_{timestamp}.xlsx")

    log_list = []

    # 处理入库单
    try:
        receipt_wb = openpyxl.load_workbook(receipt_path)
        total_merge_count = 0

        for sheet in receipt_wb.sheetnames:
            merge_count = preprocess_receipt(receipt_wb[sheet], log_list, os.path.basename(receipt_path))
            total_merge_count += merge_count

        # 保存处理后的入库单
        base_name = os.path.splitext(os.path.basename(receipt_path))[0]
        ext = os.path.splitext(os.path.basename(receipt_path))[1]
        processed_filename = f"processed_{base_name}_{timestamp}{ext}"
        processed_path = os.path.join(processed_receipt_dir, processed_filename)
        receipt_wb.save(processed_path)
    except Exception as e:
        log_list.append({
            "类型": "错误",
            "入库单文件": os.path.basename(receipt_path),
            "入库单工作表": "所有工作表",
            "详情": f"处理入库单失败: {str(e)}"
        })
        return None, None

    total_matched = 0
    total_unmatched = 0

    # 处理送货单
    try:
        # 转换xls为xlsx
        if total_path.lower().endswith(".xls"):
            tmp_file = total_path + "x"
            df = pd.read_excel(total_path)
            df.to_excel(tmp_file, index=False)
            total_path = tmp_file

        total_wb = openpyxl.load_workbook(total_path)

        # 对账所有工作表 - 现在返回索引
        global_delivery_index = reconcile_all_sheets(
            receipt_wb, os.path.basename(receipt_path),
            total_wb, os.path.basename(total_path),
            log_list
        )

        # 然后逐个工作表对账
        for receipt_sheet_name in receipt_wb.sheetnames:
            receipt_ws = receipt_wb[receipt_sheet_name]
            matched, unmatched = reconcile_single_sheet(
                receipt_ws, receipt_sheet_name, os.path.basename(receipt_path),
                global_delivery_index, os.path.basename(total_path),
                log_list
            )
            total_matched += matched
            total_unmatched += unmatched

        # 保存对账结果（入库单）
        receipt_wb.save(result_file)
    except Exception as e:
        log_list.append({
            "类型": "错误",
            "送货单文件": os.path.basename(total_path),
            "送货单工作表": "所有工作表",
            "详情": f"处理送货单失败: {str(e)}"
        })
        return None, None

    # 从log_list中提取三类记录
    matched_records = []
    mismatched_records = []
    not_found_records = []

    for log in log_list:
        if log.get("类型") == "对账成功":
            matched_records.append({
                "receipt_sheet": log.get("入库单工作表", ""),
                "receipt_row": log.get("入库单行号", ""),
                "delivery_sheet": log.get("送货单工作表", ""),
                "delivery_row": log.get("送货单行号", ""),
                "doc_no": log.get("详情", "").split("单据号")[1].split("和")[0] if "单据号" in log.get("详情", "") else "",
                "material_code": log.get("详情", "").split("料品编码")[1].split("核对成功")[0] if "料品编码" in log.get("详情", "") else "",
                "quantity": log.get("入库单数量", "")
            })
        elif log.get("类型") in ["数量不匹配", "数量错误"]:
            mismatched_records.append({
                "receipt_sheet": log.get("入库单工作表", ""),
                "receipt_row": log.get("入库单行号", ""),
                "delivery_sheet": log.get("送货单工作表", ""),
                "delivery_row": log.get("送货单行号", ""),
                "doc_no": log.get("详情", "").split("单据号")[1].split("和")[0] if "单据号" in log.get("详情", "") else "",
                "material_code": log.get("详情", "").split("料品编码")[1].split("核对成功")[0] if "料品编码" in log.get("详情", "") else "",
                "receipt_quantity": log.get("入库单数量", ""),
                "delivery_quantity": log.get("送货单数量", ""),
                "reason": log.get("类型", "")
            })
        elif log.get("类型") in ["未对账", "未对账-详细"]:
            reason = "未找到匹配"
            detail = log.get("详情", "")
            if "单据号为空" in detail:
                reason = "单据号为空"
            elif "数量为空" in detail:
                reason = "数量为空"
            elif "长度不足4位" in detail:
                reason = "单据号格式错误"
            elif "料品编码为空" in detail:
                reason = "料品编码为空"
            elif "未找到匹配项" in detail or "未找到匹配" in detail:
                reason = "未找到匹配"

            not_found_records.append({
                "receipt_sheet": log.get("入库单工作表", ""),
                "receipt_row": log.get("入库单行号", ""),
                "doc_no": log.get("入库单单据号", log.get("详情", "").split("单据号: ")[1].split(",")[0] if "单据号: " in log.get("详情", "") else ""),
                "material_code": log.get("入库单料品编码", log.get("详情", "").split("料品编码: ")[1].split(")")[0] if "料品编码: " in log.get("详情", "") else ""),
                "quantity": log.get("入库单数量", ""),
                "reason": reason
            })

    # 创建Excel日志
    log_excel_file = create_excel_log(output_dir, timestamp, matched_records, mismatched_records, not_found_records,
                                      log_list, receipt_path, total_path)

    return result_file, log_excel_file


