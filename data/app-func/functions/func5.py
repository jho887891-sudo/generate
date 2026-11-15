import openpyxl
from openpyxl.styles import PatternFill
import pandas as pd
import os
import csv
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


def clean_quantity(value):
    """清理数量值，只保留数字和小数点"""
    if value is None:
        return None

    # 如果是数值类型直接返回
    if isinstance(value, (int, float)):
        return value

    # 处理字符串类型
    s = str(value).strip()

    # 移除非数字字符（保留小数点和负号）
    cleaned = re.sub(r'[^\d.-]', '', s)

    # 如果清理后为空，返回None
    if not cleaned:
        return None

    try:
        # 转换为浮点数
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def reconcile_purchase_files(purchase_receipt_path, purchase_statement_path, output_dir):
    """
    执行采购对账功能

    参数:
        purchase_receipt_path: 采购收据单文件路径（第一张图格式）
        purchase_statement_path: 采购对账单文件路径（第二张图格式）
        output_dir: 输出目录

    返回:
        result_file: 对账结果文件路径（采购对账单）
        log_file: 日志文件路径
    """
    # 蓝色填充 - 表示已核对
    highlight = PatternFill(start_color="ADD8E6", end_color="ADD8E6", fill_type="solid")
    # 红色填充 - 表示数量不匹配
    red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 日志文件路径
    log_file = os.path.join(output_dir, f"采购对账日志_{timestamp}.csv")
    # 对账结果文件路径（采购对账单）
    result_file = os.path.join(output_dir, f"采购对账结果_{timestamp}.xlsx")

    log_list = []

    try:
        # 记录输入路径信息
        log_list.append({
            "类型": "文件路径信息",
            "详情": f"采购收据单文件路径: {purchase_receipt_path}"
        })
        log_list.append({
            "类型": "文件路径信息",
            "详情": f"采购对账单文件路径: {purchase_statement_path}"
        })
        log_list.append({
            "类型": "文件路径信息",
            "详情": f"输出目录路径: {output_dir}"
        })

        # 检查文件是否存在
        if not os.path.exists(purchase_receipt_path):
            log_list.append({
                "类型": "错误",
                "详情": f"采购收据单文件不存在: {purchase_receipt_path}"
            })
            return None, None

        if not os.path.exists(purchase_statement_path):
            log_list.append({
                "类型": "错误",
                "详情": f"采购对账单文件不存在: {purchase_statement_path}"
            })
            return None, None

        # 加载采购收据单（第一张图格式）
        receipt_wb = openpyxl.load_workbook(purchase_receipt_path)
        # 加载采购对账单（第二张图格式）
        statement_wb = openpyxl.load_workbook(purchase_statement_path)

        # 记录文件基本信息
        receipt_sheet_count = len(receipt_wb.sheetnames)
        statement_sheet_count = len(statement_wb.sheetnames)
        log_list.append({
            "类型": "文件信息",
            "详情": f"采购收据单包含 {receipt_sheet_count} 个工作表: {', '.join(receipt_wb.sheetnames)}"
        })
        log_list.append({
            "类型": "文件信息",
            "详情": f"采购对账单包含 {statement_sheet_count} 个工作表: {', '.join(statement_wb.sheetnames)}"
        })

        # 处理每个采购收据单工作表
        for receipt_sheet_name in receipt_wb.sheetnames:
            receipt_ws = receipt_wb[receipt_sheet_name]

            # 处理每个采购对账单工作表
            for statement_sheet_name in statement_wb.sheetnames:
                statement_ws = statement_wb[statement_sheet_name]

                # 创建采购对账单索引 (采购单号 + 物料编码)
                statement_index = {}
                for t in range(2, statement_ws.max_row + 1):
                    po_number = str(statement_ws[f"B{t}"].value).strip() if statement_ws[f"B{t}"].value else ""
                    material_code = str(statement_ws[f"C{t}"].value).strip() if statement_ws[f"C{t}"].value else ""
                    quantity = statement_ws[f"F{t}"].value

                    if po_number and material_code:
                        key = f"{po_number}_{material_code}"
                        if key not in statement_index:
                            statement_index[key] = []
                        statement_index[key].append({
                            "row": t,
                            "quantity": quantity
                        })

                # 记录采购对账单索引信息用于调试
                log_list.append({
                    "类型": "调试信息",
                    "采购收据单文件": os.path.basename(purchase_receipt_path),
                    "采购收据单工作表": receipt_sheet_name,
                    "采购对账单文件": os.path.basename(purchase_statement_path),
                    "采购对账单工作表": statement_sheet_name,
                    "详情": f"采购对账单中存在{len(statement_index)}个不同的采购单号+物料编码组合"
                })

                # 遍历采购收据单进行对账
                for r in range(2, receipt_ws.max_row + 1):
                    # 获取采购单号 (K列)
                    po_number = str(receipt_ws[f"K{r}"].value).strip() if receipt_ws[f"K{r}"].value else ""
                    # 获取料品编码 (E列)
                    material_code = str(receipt_ws[f"E{r}"].value).strip() if receipt_ws[f"E{r}"].value else ""
                    # 获取数量 (J列)
                    receipt_quantity = receipt_ws[f"J{r}"].value

                    # 检查必要的数据是否存在
                    if not po_number:
                        log_list.append({
                            "类型": "未对账",
                            "采购收据单文件": os.path.basename(purchase_receipt_path),
                            "采购收据单工作表": receipt_sheet_name,
                            "采购收据单行号": r,
                            "详情": f"第{r}行: 采购单号为空，无法对账"
                        })
                        continue

                    if not material_code:
                        log_list.append({
                            "类型": "未对账",
                            "采购收据单文件": os.path.basename(purchase_receipt_path),
                            "采购收据单工作表": receipt_sheet_name,
                            "采购收据单行号": r,
                            "详情": f"第{r}行: 料品编码为空，无法对账 (采购单号: {po_number})"
                        })
                        continue

                    if receipt_quantity is None:
                        log_list.append({
                            "类型": "未对账",
                            "采购收据单文件": os.path.basename(purchase_receipt_path),
                            "采购收据单工作表": receipt_sheet_name,
                            "采购收据单行号": r,
                            "详情": f"第{r}行: 数量为空，无法对账 (采购单号: {po_number}, 料品编码: {material_code})"
                        })
                        continue

                    # 清理数量值
                    cleaned_receipt_quantity = clean_quantity(receipt_quantity)
                    if cleaned_receipt_quantity is None:
                        log_list.append({
                            "类型": "数量错误",
                            "采购收据单文件": os.path.basename(purchase_receipt_path),
                            "采购收据单工作表": receipt_sheet_name,
                            "采购收据单行号": r,
                            "详情": f"第{r}行: 无法解析数量值 '{receipt_quantity}' (采购单号: {po_number}, 料品编码: {material_code})"
                        })
                        continue

                    # 构建键
                    key = f"{po_number}_{material_code}"
                    found_match = False

                    if key in statement_index:
                        for statement_item in statement_index[key]:
                            t = statement_item["row"]
                            statement_quantity = statement_item["quantity"]

                            # 清理采购对账单数量值
                            cleaned_statement_quantity = clean_quantity(statement_quantity)

                            if cleaned_statement_quantity is None:
                                log_list.append({
                                    "类型": "数量错误",
                                    "采购收据单文件": os.path.basename(purchase_receipt_path),
                                    "采购收据单工作表": receipt_sheet_name,
                                    "采购收据单行号": r,
                                    "采购对账单文件": os.path.basename(purchase_statement_path),
                                    "采购对账单工作表": statement_sheet_name,
                                    "采购对账单行号": t,
                                    "采购收据单数量": receipt_quantity,
                                    "采购对账单数量": statement_quantity,
                                    "详情": f"第{r}行: 采购对账单数量值无法解析 '{statement_quantity}' (采购单号: {po_number}, 料品编码: {material_code})"
                                })
                                continue

                            # 比较数量
                            if abs(cleaned_receipt_quantity - cleaned_statement_quantity) > 0.001:
                                # 使用采购收据单的数量覆盖采购对账单的数量
                                statement_ws[f"F{t}"] = receipt_quantity
                                # 标红采购对账单的数量单元格
                                statement_ws[f"F{t}"].fill = red_fill

                                log_list.append({
                                    "类型": "数量不匹配",
                                    "采购收据单文件": os.path.basename(purchase_receipt_path),
                                    "采购收据单工作表": receipt_sheet_name,
                                    "采购收据单行号": r,
                                    "采购对账单文件": os.path.basename(purchase_statement_path),
                                    "采购对账单工作表": statement_sheet_name,
                                    "采购对账单行号": t,
                                    "采购收据单数量": receipt_quantity,
                                    "采购对账单原数量": statement_quantity,
                                    "详情": f"采购收据单{receipt_sheet_name}第{r}行与采购对账单{statement_sheet_name}第{t}行数量不匹配: {cleaned_receipt_quantity} vs {cleaned_statement_quantity}，已用采购收据单数量覆盖 (采购单号: {po_number}, 料品编码: {material_code})"
                                })
                            else:
                                # 数量匹配 - 标蓝采购对账单的数量单元格
                                statement_ws[f"F{t}"].fill = highlight

                                log_list.append({
                                    "类型": "对账成功",
                                    "采购收据单文件": os.path.basename(purchase_receipt_path),
                                    "采购收据单工作表": receipt_sheet_name,
                                    "采购收据单行号": r,
                                    "采购对账单文件": os.path.basename(purchase_statement_path),
                                    "采购对账单工作表": statement_sheet_name,
                                    "采购对账单行号": t,
                                    "采购收据单数量": receipt_quantity,
                                    "采购对账单数量": statement_quantity,
                                    "详情": f"采购单号{po_number}和料品编码{material_code}核对成功"
                                })

                            found_match = True

                    if not found_match:
                        log_list.append({
                            "类型": "未匹配",
                            "采购收据单文件": os.path.basename(purchase_receipt_path),
                            "采购收据单工作表": receipt_sheet_name,
                            "采购收据单行号": r,
                            "详情": f"第{r}行: 采购单号{po_number}和料品编码{material_code}在采购对账单中未找到匹配项"
                        })

        # 保存对账结果（采购对账单）
        statement_wb.save(result_file)

        # 记录输出路径信息
        log_list.append({
            "类型": "文件路径信息",
            "详情": f"对账结果文件路径: {result_file}"
        })
        log_list.append({
            "类型": "文件路径信息",
            "详情": f"对账日志文件路径: {log_file}"
        })

        # 重新组织日志内容，只保留重要信息
        simplified_log_list = []

        # 1. 首先添加数量有问题的页面信息（重要提示）
        quantity_issues = [log for log in log_list if log.get("类型") in ["数量不匹配", "数量错误"]]
        if quantity_issues:
            # 按工作表分组
            issue_by_sheet = {}
            for issue in quantity_issues:
                sheet_name = issue.get("采购对账单工作表", "未知工作表")
                if sheet_name not in issue_by_sheet:
                    issue_by_sheet[sheet_name] = []
                issue_by_sheet[sheet_name].append(issue)

            # 添加汇总信息
            for sheet_name, issues in issue_by_sheet.items():
                simplified_log_list.append({
                    "类型": "重要提示",
                    "采购对账单文件": os.path.basename(purchase_statement_path),
                    "采购对账单工作表": sheet_name,
                    "详情": f"工作表'{sheet_name}'中有{len(issues)}行数量不匹配，已用采购收据单数量覆盖并用红色标记"
                })

                # 添加具体问题行
                for issue in issues:
                    simplified_log_list.append({
                        "类型": issue["类型"],
                        "采购对账单文件": issue.get("采购对账单文件", ""),
                        "采购对账单工作表": issue.get("采购对账单工作表", ""),
                        "采购对账单行号": issue.get("采购对账单行号", ""),
                        "采购收据单文件": issue.get("采购收据单文件", ""),
                        "采购收据单工作表": issue.get("采购收据单工作表", ""),
                        "采购收据单行号": issue.get("采购收据单行号", ""),
                        "采购收据单数量": issue.get("采购收据单数量", ""),
                        "采购对账单原数量": issue.get("采购对账单原数量", ""),
                        "详情": issue.get("详情", "")
                    })

        # 2. 添加简洁的未对账信息
        unmatched_logs = [log for log in log_list if log.get("类型") in ["未对账", "未匹配"]]
        if unmatched_logs:
            # 分析无法核对的原因
            reasons = {}
            for log in unmatched_logs:
                detail = log.get("详情", "")
                if "采购单号为空" in detail:
                    reason = "采购单号为空"
                elif "料品编码为空" in detail:
                    reason = "料品编码为空"
                elif "数量为空" in detail:
                    reason = "数量为空"
                elif "未找到匹配项" in detail:
                    reason = "采购单号或料品编码不匹配"
                else:
                    reason = "其他原因"

                if reason not in reasons:
                    reasons[reason] = 0
                reasons[reason] += 1

            # 添加汇总信息（简洁版）
            simplified_log_list.append({
                "类型": "未对账汇总",
                "采购收据单文件": os.path.basename(purchase_receipt_path),
                "采购收据单工作表": "所有工作表",
                "详情": f"共有{len(unmatched_logs)}行数据无法对账，原因: {', '.join([f'{k}({v}行)' for k, v in reasons.items()])}"
            })

        # 3. 添加对账成功的信息
        success_logs = [log for log in log_list if log.get("类型") == "对账成功"]
        if success_logs:
            simplified_log_list.append({
                "类型": "对账成功",
                "采购收据单文件": os.path.basename(purchase_receipt_path),
                "采购收据单工作表": "所有工作表",
                "详情": f"共有{len(success_logs)}行数据对账成功，已用蓝色标记"
            })

        # 4. 添加对账汇总
        simplified_log_list.append({
            "类型": "对账汇总",
            "采购收据单文件": os.path.basename(purchase_receipt_path),
            "采购对账单文件": os.path.basename(purchase_statement_path),
            "详情": f"成功对账: {len(success_logs)}行, 未对账: {len(unmatched_logs)}行, 数量不匹配: {len(quantity_issues)}行"
        })

        # 5. 添加文件路径信息
        simplified_log_list.append({
            "类型": "文件路径信息",
            "详情": f"采购收据单文件路径: {purchase_receipt_path}"
        })
        simplified_log_list.append({
            "类型": "文件路径信息",
            "详情": f"采购对账单文件路径: {purchase_statement_path}"
        })
        simplified_log_list.append({
            "类型": "文件路径信息",
            "详情": f"输出目录路径: {output_dir}"
        })
        simplified_log_list.append({
            "类型": "文件路径信息",
            "详情": f"对账结果文件路径: {result_file}"
        })
        simplified_log_list.append({
            "类型": "文件路径信息",
            "详情": f"对账日志文件路径: {log_file}"
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
                        "采购收据单文件", "采购收据单工作表", "采购收据单行号",
                        "采购对账单文件", "采购对账单工作表", "采购对账单行号",
                        "采购收据单数量", "采购对账单原数量",
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

    except PermissionError as pe:
        logger.error(f"文件权限错误: {str(pe)}")
        return None, None
    except Exception as e:
        logger.error(f"采购对账失败: {str(e)}", exc_info=True)
        return None, None