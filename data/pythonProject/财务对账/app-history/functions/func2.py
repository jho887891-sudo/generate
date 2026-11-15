import openpyxl
from openpyxl.styles import PatternFill
import pandas as pd
import os
import csv
from datetime import datetime
import logging

# 蓝色填充
highlight = PatternFill(start_color="ADD8E6", end_color="ADD8E6", fill_type="solid")
# 红色填充
red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")

# 在函数开头添加日志记录器
logger = logging.getLogger(__name__)
# 检查字符串是否只包含字母和数字
def is_alphanumeric(s):
    return all(c.isalnum() for c in s)


# 处理收据表格，合并行逻辑 - 每次合并后都向上移动F列数据
def preprocess_receipt(ws, log_list, receipt_file):
    row = 2  # 从第二行开始（第一行一般是表头）
    merge_count = 0

    while row <= ws.max_row:
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
                        "文件": receipt_file,
                        "工作表": ws.title,
                        "详情": f"第{row}行和第{row + 1}行F列合并，合并后值: {merged_value}"
                    })

                    # 向上移动后续的F列数据填补空缺
                    move_f_column_up(ws, row + 1, log_list, receipt_file)

                    # 继续检查当前行，因为可能有连续的合并
                    continue

        row += 1

    return merge_count


# 向上移动F列数据填补空缺
def move_f_column_up(ws, empty_row, log_list, receipt_file):
    # 从空行开始，向上移动后续的F列数据
    for row in range(empty_row, ws.max_row):
        # 将下一行的F列数据移动到当前行
        next_f_val = ws[f"F{row + 1}"].value
        ws[f"F{row}"] = next_f_val
        ws[f"F{row + 1}"] = None

        if next_f_val:
            log_list.append({
                "类型": "数据移动",
                "文件": receipt_file,
                "工作表": ws.title,
                "详情": f"第{row + 1}行F列数据移动到第{row}行F列"
            })


# 对账逻辑，增加日志记录
def reconcile(receipt_ws, receipt_file, total_ws, total_file, log_list):
    matched_count = 0
    unmatched_count = 0

    for r in range(2, receipt_ws.max_row + 1):
        receipt_f = str(receipt_ws[f"F{r}"].value).strip() if receipt_ws[f"F{r}"].value else ""
        receipt_d = str(receipt_ws[f"D{r}"].value).strip() if receipt_ws[f"D{r}"].value else ""
        receipt_l = receipt_ws[f"L{r}"].value

        # 检查必要的数据是否存在
        if not receipt_f:
            log_list.append({
                "类型": "未对账",
                "文件": receipt_file,
                "工作表": receipt_ws.title,
                "详情": f"第{r}行: 单据号为空，无法对账"
            })
            unmatched_count += 1
            continue

        if receipt_l is None:
            log_list.append({
                "类型": "未对账",
                "文件": receipt_file,
                "工作表": receipt_ws.title,
                "详情": f"第{r}行: 数量为空，无法对账 (单据号: {receipt_f[-4:]})"
            })
            unmatched_count += 1
            continue

        receipt_key = receipt_f[-4:]
        found_match = False

        for t in range(2, total_ws.max_row + 1):
            total_b = str(total_ws[f"B{t}"].value).strip() if total_ws[f"B{t}"].value else ""
            total_d = str(total_ws[f"D{t}"].value).strip() if total_ws[f"D{t}"].value else ""
            total_i = total_ws[f"I{t}"].value

            if not total_b:
                continue

            # 匹配逻辑：单据号后4位必须匹配，单号如果有的话也要匹配
            if (total_b[-4:] == receipt_key and
                    (not receipt_d or not total_d or receipt_d == total_d)):
                found_match = True
                # 整行标黄
                for cell in total_ws[t]:
                    cell.fill = highlight

                # 改进的数量比较逻辑
                need_update = False
                status = "已核对"

                # 处理数值比较（考虑浮点数精度和类型转换）
                try:
                    receipt_num = float(receipt_l) if receipt_l is not None else None
                    total_num = float(total_i) if total_i is not None else None

                    if receipt_num is not None and total_num is not None:
                        # 允许很小的浮点数误差
                        if abs(receipt_num - total_num) > 0.001:
                            need_update = True
                            status = f"已修改 (第{t}行数量从{total_i}改为{receipt_l})"
                    elif receipt_l != total_i:  # 处理None值情况
                        need_update = True
                        status = f"已修改 (第{t}行数量从{total_i}改为{receipt_l})"
                except (ValueError, TypeError):
                    # 如果无法转换为数字，直接比较原始值
                    if str(receipt_l) != str(total_i):
                        need_update = True
                        status = f"已修改 (第{t}行数量从{total_i}改为{receipt_l})"

                log_entry = {
                    "类型": "对账结果",
                    "文件": receipt_file,
                    "工作表": receipt_ws.title,
                    "详情": f"第{r}行单据号{receipt_key}: {status}",
                    "单据号后4位": receipt_key,
                    "单号": receipt_d if receipt_d else "无",
                    "收据数量": receipt_l,
                    "总表原数量": total_i,
                    "状态": status
                }

                if need_update:
                    total_ws[f"I{t}"] = receipt_l
                    total_ws[f"I{t}"].fill = red_fill

                log_list.append(log_entry)
                matched_count += 1
                break

        if not found_match:
            log_list.append({
                "类型": "未对账",
                "文件": receipt_file,
                "工作表": receipt_ws.title,
                "详情": f"第{r}行: 单据号{receipt_key}在总表中未找到匹配项"
            })
            unmatched_count += 1

    return matched_count, unmatched_count


# 适配Flask应用的主函数
def reconcile_excel_files(receipt_path, total_path, output_dir):
    """
    执行对账功能

    参数:
        receipt_path: 收据文件路径
        total_path: 总表文件路径
        output_dir: 输出目录

    返回:
        result_file: 对账结果文件路径
        log_file: 日志文件路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建处理后的收据表目录
    processed_receipt_dir = os.path.join(output_dir, "processed_receipts")
    os.makedirs(processed_receipt_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 日志文件直接放在输出目录下
    log_file = os.path.join(output_dir, f"对账日志_{timestamp}.csv")
    result_file = os.path.join(output_dir, f"对账结果_{timestamp}.xlsx")

    log_list = []

    # 打开收据表并预处理
    try:
        receipt_wb = openpyxl.load_workbook(receipt_path)
        total_merge_count = 0

        for sheet in receipt_wb.sheetnames:
            merge_count = preprocess_receipt(receipt_wb[sheet], log_list, os.path.basename(receipt_path))
            total_merge_count += merge_count

        if total_merge_count > 0:
            log_list.append({
                "类型": "预处理汇总",
                "文件": os.path.basename(receipt_path),
                "工作表": "所有工作表",
                "详情": f"共合并了{total_merge_count}行数据"
            })

        # 保存处理后的收据表 - 添加时间戳后缀
        base_name = os.path.splitext(os.path.basename(receipt_path))[0]
        ext = os.path.splitext(os.path.basename(receipt_path))[1]
        processed_filename = f"processed_{base_name}_{timestamp}{ext}"
        processed_path = os.path.join(processed_receipt_dir, processed_filename)
        receipt_wb.save(processed_path)
    except Exception as e:
        log_list.append({
            "类型": "错误",
            "文件": os.path.basename(receipt_path),
            "工作表": "所有工作表",
            "详情": f"处理失败: {str(e)}"
        })
        return None, None

    total_matched = 0
    total_unmatched = 0

    # 处理总表文件
    try:
        # 如果是 .xls 就转成 .xlsx
        if total_path.lower().endswith(".xls"):
            tmp_file = total_path + "x"
            df = pd.read_excel(total_path)
            df.to_excel(tmp_file, index=False)
            total_path = tmp_file

        total_wb = openpyxl.load_workbook(total_path)
        total_ws = total_wb.active

        # 对账
        for sheet in receipt_wb.sheetnames:
            matched, unmatched = reconcile(receipt_wb[sheet], os.path.basename(receipt_path), total_ws,
                                           os.path.basename(total_path), log_list)
            total_matched += matched
            total_unmatched += unmatched

        # 保存对账结果
        total_wb.save(result_file)
    except Exception as e:
        log_list.append({
            "类型": "错误",
            "文件": os.path.basename(total_path),
            "工作表": "所有工作表",
            "详情": f"处理失败: {str(e)}"
        })
        return None, None

    # 添加汇总信息
    log_list.append({
        "类型": "对账汇总",
        "文件": "所有文件",
        "工作表": "所有工作表",
        "详情": f"成功对账: {total_matched}行, 未对账: {total_unmatched}行"
    })

    # 保存日志
    try:
        if log_list:
            with open(log_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f,
                                        fieldnames=["类型", "文件", "工作表", "详情", "单据号后4位", "单号", "收据数量",
                                                    "总表原数量", "状态"])
                writer.writeheader()
                writer.writerows(log_list)
    except Exception as e:
        # 修正这里：使用 logger 而不是 log
        logger.error(f"保存日志文件失败: {str(e)}")
        return None, None

    return result_file, log_file