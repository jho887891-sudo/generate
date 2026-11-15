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

    # 移除非字母数字字符（保留字母和数字）
    cleaned = re.sub(r'[^\a-zA-Z0-9]', '', cleaned)

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
    total_matched = 0
    total_unmatched = 0

    # 处理每个入库单工作表
    for receipt_sheet_name in receipt_wb.sheetnames:
        receipt_ws = receipt_wb[receipt_sheet_name]

        # 处理每个送货单工作表
        for total_sheet_name in total_wb.sheetnames:
            total_ws = total_wb[total_sheet_name]

            matched, unmatched = reconcile_single_sheet(
                receipt_ws, receipt_sheet_name, receipt_file,
                total_ws, total_sheet_name, total_file,
                log_list
            )
            total_matched += matched
            total_unmatched += unmatched

    return total_matched, total_unmatched


def reconcile_single_sheet(receipt_ws, receipt_sheet_name, receipt_file,
                           total_ws, total_sheet_name, total_file, log_list):
    """对账逻辑：核对单个工作表的入库单和送货单"""
    matched_count = 0
    unmatched_count = 0

    # 创建送货单单据号索引，提高匹配效率
    delivery_doc_index = {}
    for t in range(2, total_ws.max_row + 1):
        total_f = str(total_ws[f"F{t}"].value).strip() if total_ws[f"F{t}"].value else ""
        total_l = total_ws[f"L{t}"].value if total_ws[f"L{t}"].value is not None else ""

        if total_f:
            # 清理单据号
            cleaned_total_f = clean_document_number(total_f)
            if len(cleaned_total_f) >= 4:
                last_four = cleaned_total_f[-4:]
                if last_four not in delivery_doc_index:
                    delivery_doc_index[last_four] = []
                delivery_doc_index[last_four].append({
                    "row": t,
                    "doc_no": total_f,
                    "cleaned_doc_no": cleaned_total_f,
                    "quantity": total_l
                })

    # 记录送货单中存在的单据号用于调试
    delivery_doc_nums = list(delivery_doc_index.keys())
    log_list.append({
        "类型": "调试信息",
        "入库单文件": receipt_file,
        "入库单工作表": receipt_sheet_name,
        "送货单文件": total_file,
        "送货单工作表": total_sheet_name,
        "详情": f"送货单中存在{len(delivery_doc_nums)}个不同的单据号后四位: {delivery_doc_nums[:20]}"
    })

    for r in range(2, receipt_ws.max_row + 1):
        receipt_f = str(receipt_ws[f"F{r}"].value).strip() if receipt_ws[f"F{r}"].value else ""
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

        if receipt_l is None:
            log_list.append({
                "类型": "未对账",
                "入库单文件": receipt_file,
                "入库单工作表": receipt_sheet_name,
                "入库单行号": r,
                "详情": f"第{r}行: 数量为空，无法对账 (单据号: {receipt_f})"
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

        receipt_key = cleaned_receipt_f[-4:]
        found_match = False

        # 使用索引查找匹配的单据
        if receipt_key in delivery_doc_index:
            for delivery_item in delivery_doc_index[receipt_key]:
                t = delivery_item["row"]
                total_f = delivery_item["doc_no"]
                total_l = delivery_item["quantity"]
                cleaned_total_f = delivery_item["cleaned_doc_no"]

                # 再次确认匹配
                if len(cleaned_total_f) >= 4 and cleaned_total_f[-4:] == receipt_key:
                    found_match = True

                    # 清理数量值（处理可能包含单位的情况）
                    receipt_l_cleaned = clean_quantity(receipt_l)
                    total_l_cleaned = clean_quantity(total_l)

                    # 比较数量
                    try:
                        receipt_num = float(receipt_l_cleaned) if receipt_l_cleaned is not None else None
                        total_num = float(total_l_cleaned) if total_l_cleaned is not None else None

                        # 允许很小的浮点数误差
                        if receipt_num is not None and total_num is not None:
                            if abs(receipt_num - total_num) > 0.001:
                                # 标红入库单的L列单元格
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
                                # 数量匹配 - 标蓝入库单的L列单元格
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
                                    "详情": f"单据号{receipt_key}核对成功"
                                })
                        else:
                            # 无法比较数量 - 标红入库单的L列单元格
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
                        # 数量比较出错 - 标红入库单的L列单元格
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
            # 记录详细信息以便调试
            log_list.append({
                "类型": "未对账",
                "入库单文件": receipt_file,
                "入库单工作表": receipt_sheet_name,
                "入库单行号": r,
                "入库单单据号": receipt_f,
                "清理后单据号": cleaned_receipt_f,
                "详情": f"第{r}行: 单据号{receipt_key}在送货单中未找到匹配项。原始单据号: {receipt_f}"
            })
            unmatched_count += 1

    return matched_count, unmatched_count


def reconcile_delivery_files(receipt_path, total_path, output_dir):
    """
    执行送货单对账功能

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
    # 日志文件路径
    log_file = os.path.join(output_dir, f"送货单对账日志_{timestamp}.csv")
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

        # 对账所有工作表
        matched, unmatched = reconcile_all_sheets(
            receipt_wb, os.path.basename(receipt_path),
            total_wb, os.path.basename(total_path),
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

    # 重新组织日志内容，只保留重要信息
    simplified_log_list = []

    # 1. 首先添加数量有问题的页面信息（重要提示）
    quantity_issues = [log for log in log_list if log.get("类型") in ["数量不匹配", "数量错误"]]
    if quantity_issues:
        # 按工作表分组
        issue_by_sheet = {}
        for issue in quantity_issues:
            sheet_name = issue.get("入库单工作表", "未知工作表")
            if sheet_name not in issue_by_sheet:
                issue_by_sheet[sheet_name] = []
            issue_by_sheet[sheet_name].append(issue)

        # 添加汇总信息
        for sheet_name, issues in issue_by_sheet.items():
            simplified_log_list.append({
                "类型": "重要提示",
                "入库单文件": os.path.basename(receipt_path),
                "入库单工作表": sheet_name,
                "详情": f"工作表'{sheet_name}'中有{len(issues)}行数量不匹配，已用红色标记"
            })

            # 添加具体问题行
            for issue in issues:
                simplified_log_list.append({
                    "类型": issue["类型"],
                    "入库单文件": issue.get("入库单文件", ""),
                    "入库单工作表": issue.get("入库单工作表", ""),
                    "入库单行号": issue.get("入库单行号", ""),
                    "送货单文件": issue.get("送货单文件", ""),
                    "送货单工作表": issue.get("送货单工作表", ""),
                    "送货单行号": issue.get("送货单行号", ""),
                    "入库单数量": issue.get("入库单数量", ""),
                    "送货单数量": issue.get("送货单数量", ""),
                    "详情": issue.get("详情", "")
                })

    # 2. 添加简洁的未对账信息
    unmatched_logs = [log for log in log_list if log.get("类型") == "未对账"]
    if unmatched_logs:
        # 分析无法核对的原因
        reasons = {}
        for log in unmatched_logs:
            detail = log.get("详情", "")
            if "单据号为空" in detail:
                reason = "单据号为空"
            elif "数量为空" in detail:
                reason = "数量为空"
            elif "长度不足4位" in detail:
                reason = "单据号格式错误"
            elif "未找到匹配项" in detail:
                reason = "单据号不匹配"
            else:
                reason = "其他原因"

            if reason not in reasons:
                reasons[reason] = 0
            reasons[reason] += 1

        # 添加汇总信息（简洁版）
        simplified_log_list.append({
            "类型": "未对账汇总",
            "入库单文件": os.path.basename(receipt_path),
            "入库单工作表": "所有工作表",
            "详情": f"共有{len(unmatched_logs)}行数据无法对账，原因: {', '.join([f'{k}({v}行)' for k, v in reasons.items()])}"
        })

    # 3. 添加对账成功的信息
    success_logs = [log for log in log_list if log.get("类型") == "对账成功"]
    if success_logs:
        simplified_log_list.append({
            "类型": "对账成功",
            "入库单文件": os.path.basename(receipt_path),
            "入库单工作表": "所有工作表",
            "详情": f"共有{len(success_logs)}行数据对账成功，已用蓝色标记"
        })

    # 4. 添加对账汇总
    simplified_log_list.append({
        "类型": "对账汇总",
        "入库单文件": os.path.basename(receipt_path),
        "送货单文件": os.path.basename(total_path),
        "详情": f"成功对账: {total_matched}行, 未对账: {total_unmatched}行"
    })

    # 5. 添加错误信息（如果有）
    error_logs = [log for log in log_list if log.get("类型") == "错误"]
    if error_logs:
        simplified_log_list.extend(error_logs)

    # 保存简化的日志
    try:
        if simplified_log_list:
            with open(log_file, "w", newline="", encoding="utf-8-sig") as f:
                fieldnames = [
                    "类型",
                    "入库单文件", "入库单工作表", "入库单行号",
                    "送货单文件", "送货单工作表", "送货单行号",
                    "入库单数量", "送货单数量",
                    "详情"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                # 只写入存在的字段
                for log_entry in simplified_log_list:
                    filtered_entry = {k: v for k, v in log_entry.items() if k in fieldnames}
                    writer.writerow(filtered_entry)
    except Exception as e:
        logger.error(f"保存日志文件失败: {str(e)}")
        return None, None

    return result_file, log_file