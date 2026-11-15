import pandas as pd
import json
import os
from openpyxl import load_workbook
import logging
import datetime

# 设置日志
logger = logging.getLogger(__name__)


def process_invoice_pricing(price_list_file, invoice_file, output_dir, config_file=None):
    """
    处理对账单单价匹配和金额计算

    Args:
        price_list_file: 价格表文件路径
        invoice_file: 对账单文件路径
        output_dir: 输出目录
        config_file: 配置文件路径（可选）

    Returns:
        tuple: (result_file, error_msg) 结果文件路径和错误信息
    """
    try:
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"创建输出目录: {output_dir}")

        # 设置输出文件路径
        result_file = os.path.join(output_dir, f"对账单_处理结果_{timestamp}.xlsx")

        # 检查输入文件是否存在
        if not os.path.exists(price_list_file):
            error_msg = f"价格表文件不存在: {price_list_file}"
            logger.error(error_msg)
            return None, error_msg

        if not os.path.exists(invoice_file):
            error_msg = f"对账单文件不存在: {invoice_file}"
            logger.error(error_msg)
            return None, error_msg

        # 默认列映射配置
        column_mapping = {
            'price_list': {
                'code_column': '价表编码',
                'price_column': '单价',
                'modify_time_column': '修改时间'  # 新增修改时间列配置
            },
            'invoice': {
                'code_column': '物料编码',
                'price_column': '单价',
                'quantity_column': '数量',
                'amount_column': '金额'
            }
        }

        # 如果提供了配置文件，则加载配置
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 更新列映射配置
                if 'price_list' in config:
                    column_mapping['price_list'].update(config['price_list'])
                if 'invoice' in config:
                    column_mapping['invoice'].update(config['invoice'])

                logger.info(f"已从 {config_file} 加载列映射配置")
                logger.info(f"价格表列映射: {column_mapping['price_list']}")
                logger.info(f"对账单列映射: {column_mapping['invoice']}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")

        # 读取价格表
        try:
            price_df = pd.read_excel(price_list_file, sheet_name='Sheet1')
            logger.info(f"成功读取价格表: {len(price_df)} 条记录")

            # 验证必要的列是否存在
            price_config = column_mapping['price_list']
            required_columns = [price_config['code_column'], price_config['price_column']]
            missing_columns = [col for col in required_columns if col not in price_df.columns]

            if missing_columns:
                error_msg = f"价格表中缺少必要的列: {missing_columns}"
                logger.error(error_msg)
                logger.error(f"价格表的列: {price_df.columns.tolist()}")
                return None, error_msg

            # 检查修改时间列是否存在
            modify_time_col = price_config.get('modify_time_column')
            if modify_time_col and modify_time_col not in price_df.columns:
                logger.warning(f"价格表中未找到修改时间列 '{modify_time_col}'，将使用默认方式处理重复编码")

            # 处理重复编码：使用最新修改时间的单价
            code_column = price_config['code_column']

            # 检查是否有重复编码
            duplicate_codes = price_df[code_column].duplicated(keep=False)
            if duplicate_codes.any():
                logger.info(f"发现 {duplicate_codes.sum()} 条重复编码记录")

                if modify_time_col and modify_time_col in price_df.columns:
                    # 确保修改时间列是日期时间格式
                    try:
                        price_df[modify_time_col] = pd.to_datetime(price_df[modify_time_col])
                    except Exception as e:
                        logger.warning(f"转换修改时间列为日期时间格式失败: {e}")
                        # 如果转换失败，使用默认方式处理
                        price_df = price_df.drop_duplicates(subset=[code_column], keep='first')
                    else:
                        # 按编码分组，取修改时间最新的记录
                        price_df = price_df.sort_values(by=[code_column, modify_time_col],
                                                        ascending=[True, False])
                        price_df = price_df.drop_duplicates(subset=[code_column], keep='first')
                        logger.info("已根据修改时间处理重复编码，使用最新时间的单价")
                else:
                    # 没有修改时间列，使用默认方式（保留第一个）
                    price_df = price_df.drop_duplicates(subset=[code_column], keep='first')
                    logger.info("未配置修改时间列，使用默认方式处理重复编码（保留第一个）")
            else:
                logger.info("价格表中未发现重复编码")

            # 创建价格字典
            price_dict = price_df.set_index(code_column)[price_config['price_column']].to_dict()
            logger.info(f"创建价格字典完成: {len(price_dict)} 个唯一编码")

        except Exception as e:
            error_msg = f"读取价格表失败: {e}"
            logger.error(error_msg)
            return None, error_msg

        # 处理对账单
        try:
            # 使用openpyxl加载对账单，保留格式
            workbook = load_workbook(invoice_file)

            # 获取工作表
            if 'Sheet1' in workbook.sheetnames:
                sheet = workbook['Sheet1']
            else:
                error_msg = f"对账单中未找到工作表 'Sheet1'"
                logger.error(error_msg)
                return None, error_msg

            # 获取列索引
            invoice_config = column_mapping['invoice']

            def get_column_index(sheet, column_name):
                """获取指定列名在sheet中的索引位置"""
                for col_idx, cell in enumerate(sheet[1]):  # 第一行是标题行
                    if cell.value and str(cell.value).strip() == str(column_name).strip():
                        return col_idx
                return None

            code_col = get_column_index(sheet, invoice_config['code_column'])
            price_col = get_column_index(sheet, invoice_config['price_column'])
            quantity_col = get_column_index(sheet, invoice_config['quantity_column'])
            amount_col = get_column_index(sheet, invoice_config['amount_column'])

            if code_col is None:
                error_msg = f"在对账单中找不到列: {invoice_config['code_column']}"
                logger.error(error_msg)
                return None, error_msg
            if price_col is None:
                error_msg = f"在对账单中找不到列: {invoice_config['price_column']}"
                logger.error(error_msg)
                return None, error_msg
            if quantity_col is None:
                error_msg = f"在对账单中找不到列: {invoice_config['quantity_column']}"
                logger.error(error_msg)
                return None, error_msg
            if amount_col is None:
                error_msg = f"在对账单中找不到列: {invoice_config['amount_column']}"
                logger.error(error_msg)
                return None, error_msg

            # 处理数据
            matched_count = 0
            calculated_count = 0
            not_found_codes = set()

            for row_idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=False), start=2):
                # 获取物料编码
                code_cell = row[code_col]
                material_code = code_cell.value

                # 第一步：价格匹配
                if material_code and str(material_code).strip():
                    material_code_str = str(material_code)
                    if material_code_str in price_dict:
                        new_price = price_dict[material_code_str]
                        price_cell = row[price_col]
                        price_cell.value = new_price
                        matched_count += 1
                        logger.debug(f"匹配成功: 行 {row_idx} 物料编码 {material_code}，价格更新为 {new_price}")
                    else:
                        not_found_codes.add(material_code_str)
                        logger.warning(f"价格匹配失败: 行 {row_idx} 物料编码 {material_code} 未在价格表中找到")

                # 第二步：金额计算
                price_cell = row[price_col]
                quantity_cell = row[quantity_col]
                amount_cell = row[amount_col]

                if price_cell.value is not None and quantity_cell.value is not None:
                    try:
                        price = float(price_cell.value)
                        quantity = float(quantity_cell.value)
                        amount = price * quantity
                        amount_cell.value = amount
                        calculated_count += 1
                    except (ValueError, TypeError) as e:
                        logger.error(
                            f"计算错误: 行 {row_idx}，数量={quantity_cell.value}，单价={price_cell.value}，错误：{e}")
                        continue

            logger.info(f"价格匹配完成: {matched_count} 条记录已更新")
            logger.info(f"金额计算完成: {calculated_count} 条记录已计算")

            if not_found_codes:
                logger.warning(
                    f"以下物料编码未在价格表中找到: {list(not_found_codes)[:10]}{'...' if len(not_found_codes) > 10 else ''}")

            # 保存结果
            workbook.save(result_file)
            logger.info(f"结果已保存到: {result_file}")

            return result_file, None

        except Exception as e:
            error_msg = f"处理对账单失败: {e}"
            logger.error(error_msg)
            return None, error_msg

    except Exception as e:
        error_msg = f"处理过程中发生错误: {e}"
        logger.error(error_msg)
        return None, error_msg