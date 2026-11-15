import numpy as np
import pandas as pd
import json
import re
import os
import logging
import time
import threading
import uuid
import queue
import sqlite3
from datetime import datetime, timedelta
from PIL import Image
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from typing import List, Tuple, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

# 参数设置
BOX_TILT_THRESHOLD = 20
TEXT_MERGE_THRESHOLD = 1.5
MATCH_THRESHOLD = 0.6
COL_ALIGNMENT_THRESHOLD = 40
ROW_GAP_THRESHOLD = 50
ROW_Y_THRESHOLD = 30


class OCRResultRef:
    """存储处理后的OCR结果"""
    __slots__ = ['texts', 'boxes']

    def __init__(self, texts: List[str], boxes: List[np.ndarray]):
        self.texts = texts
        self.boxes = boxes

    def remove_items(self, indices: List[int]):
        """删除指定索引的项目"""
        for idx in sorted(indices, reverse=True):
            del self.texts[idx]
            del self.boxes[idx]

    def add_item(self, text: str, box: np.ndarray):
        """添加新项目"""
        self.texts.append(text)
        self.boxes.append(box)

    def __len__(self):
        return len(self.texts)

    def sort_by_y(self):
        """按文本框顶部y坐标排序（从上到下）"""
        sorted_pairs = sorted(zip(self.texts, self.boxes), key=lambda x: x[1][0][1])
        self.texts, self.boxes = zip(*sorted_pairs)
        self.texts = list(self.texts)
        self.boxes = list(self.boxes)


def load_config_from_json(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_box_tilt(box: np.ndarray) -> float:
    """计算文本框的倾斜程度"""
    top_diff = abs(box[0][1] - box[1][1])
    bottom_diff = abs(box[3][1] - box[2][1])
    return max(top_diff, bottom_diff)


def should_merge_boxes(box1: np.ndarray, box2: np.ndarray,
                       text1: str, text2: str,
                       column_titles: List[str],
                       threshold: float = 8.0) -> bool:
    """判断两个文本框是否应该合并"""
    for title in column_titles:
        if (text_similarity(text1, title) > MATCH_THRESHOLD or
                text_similarity(text2, title) > MATCH_THRESHOLD):
            return False
    upper_box, lower_box = (box1, box2) if box1[0][1] < box2[0][1] else (box2, box1)
    upper_bottom_center = (upper_box[2] + upper_box[3]) / 2
    lower_top_center = (lower_box[0] + lower_box[1]) / 2
    horizontal_diff = abs(upper_bottom_center[0] - lower_top_center[0])
    vertical_condition = lower_top_center[1] < (upper_bottom_center[1] + threshold)
    return horizontal_diff < threshold and vertical_condition


def box_center(box: np.ndarray) -> np.ndarray:
    """计算文本框的中心点"""
    return np.mean(box, axis=0)


def merge_boxes(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """合并两个文本框"""
    all_points = np.vstack((box1, box2))
    min_x = np.min(all_points[:, 0])
    max_x = np.max(all_points[:, 0])
    min_y = np.min(all_points[:, 1])
    max_y = np.max(all_points[:, 1])

    return np.array([
        [min_x, min_y],  # 左上
        [max_x, min_y],  # 右上
        [max_x, max_y],  # 右下
        [min_x, max_y]  # 左下
    ])


def text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（改进版）"""
    text1_clean = text1.lower().replace(' ', '')
    text2_clean = text2.lower().replace(' ', '')
    if text1_clean == text2_clean:
        return 1.0

    def levenshtein_similarity(s1, s2):
        m, n = len(s1), len(s2)
        dp = np.zeros((m + 1, n + 1))
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return 1 - (dp[m][n] / max(m, n))

    def jaccard_similarity(s1, s2, n=3):
        def get_ngrams(s):
            return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}

        ngrams1 = get_ngrams(s1)
        ngrams2 = get_ngrams(s2)
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0

    lev_sim = levenshtein_similarity(text1_clean, text2_clean)
    jac_sim = jaccard_similarity(text1_clean, text2_clean)
    return 0.7 * lev_sim + 0.3 * jac_sim


def find_closest_text(target: str, text_list: List[str]) -> Tuple[Optional[str], float]:
    """在文本列表中找到最接近目标文本的项"""
    best_match = None
    best_score = 0.0
    for text in text_list:
        score = text_similarity(text, target)
        if score > best_score:
            best_score = score
            best_match = text
        if best_score == 1.0:
            break
    return best_match, best_score


def parse_date(text: str, years: List[str]) -> Optional[str]:
    """解析日期并统一格式"""
    patterns = [
        r'(\d{4})[\-\/\.](\d{1,2})[\-\/\.](\d{1,2})',
        r'(\d{1,2})[\-\/\.](\d{1,2})[\-\/\.](\d{4})',
        r'(\d{1,2})/(\d{1,2})/(\d{2})',
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',
        r'(\d{4})年(\d{1,2})月(\d{1,2})号',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups[0]) == 4:
                y, m, d = groups
            else:
                m, d, y = groups
                if len(y) == 2:
                    y = f"20{y}" if int(y) < 50 else f"19{y}"
            if len(y) != 4:
                continue
            return f"{y}.{m.zfill(2)}.{d.zfill(2)}"
    for year in years:
        if year in text:
            return f"{year}.01.01"
    return None


def split_into_rows(result_ref: OCRResultRef) -> List[List[int]]:
    """将文本框按垂直位置划分为行，返回每行的索引列表"""
    if len(result_ref) == 0:
        return []

    # 按顶部y坐标排序
    result_ref.sort_by_y()
    rows = []
    current_row = [0]
    current_y = result_ref.boxes[0][0][1]

    for i in range(1, len(result_ref)):
        box_y = result_ref.boxes[i][0][1]
        # 若当前文本框与当前行首的y差值小于阈值，归为同一行
        if box_y - current_y <= ROW_Y_THRESHOLD:
            current_row.append(i)
        else:
            rows.append(current_row)
            current_row = [i]
            current_y = box_y
    rows.append(current_row)
    return rows


def get_column_value_in_row(row_indices: List[int], col_center_x: float, result_ref: OCRResultRef) -> Optional[str]:
    """在指定行中找到与目标列水平对齐的文本"""
    best_match_idx = None
    min_x_diff = float('inf')

    for idx in row_indices:
        box = result_ref.boxes[idx]
        box_center_x = box_center(box)[0]
        x_diff = abs(box_center_x - col_center_x)

        # 找到水平距离最近且在阈值内的文本
        if x_diff < COL_ALIGNMENT_THRESHOLD and x_diff < min_x_diff:
            min_x_diff = x_diff
            best_match_idx = idx

    return result_ref.texts[best_match_idx] if best_match_idx is not None else ""


def is_header_row(row_indices: List[int], column_titles: List[str], result_ref: OCRResultRef) -> bool:
    """判断当前行是否为列名行"""
    row_texts = [result_ref.texts[idx].strip() for idx in row_indices]
    match_count = 0
    for title in column_titles:
        for text in row_texts:
            if text_similarity(text, title) > MATCH_THRESHOLD:
                match_count += 1
                break
    # 超过一半的列标题匹配则判定为列名行
    return match_count >= len(column_titles) / 2


def process_page(ocr_page_result: Dict, config: Dict) -> Tuple[str, str, Dict, str, OCRResultRef]:
    """处理单页OCR结果"""
    # Step 1: 提取文本和文本框
    texts = ocr_page_result['rec_texts']
    boxes = ocr_page_result['rec_polys']
    result_ref = OCRResultRef(texts, boxes)

    # Step 2: 删除倾斜文本框
    tilt_indices = []
    for i, box in enumerate(result_ref.boxes):
        if calculate_box_tilt(box) > BOX_TILT_THRESHOLD:
            tilt_indices.append(i)

    result_ref.remove_items(tilt_indices)

    # Step 3: 提取送货公司名称
    company_name = ""
    company_match_score = 0.0

    for text in result_ref.texts:
        if company_match_score == 1.0:
            break
        for company in config["company_list"]:
            score = text_similarity(text, company)
            if score > MATCH_THRESHOLD and score > company_match_score:
                company_name = company
                company_match_score = score
                if company_match_score == 1.0:
                    break

    # 如果没有匹配到公司，跳过后续处理
    if not company_name:
        return company_name, "", {}, "", result_ref
    column_titles = config["company_titles"].get(company_name, [])

    # Step 4: 合并相邻文本框
    i = 0
    while i < len(result_ref):
        j = i + 1
        merged = False

        while j < len(result_ref):
            if should_merge_boxes(result_ref.boxes[i], result_ref.boxes[j],
                                  result_ref.texts[i], result_ref.texts[j],
                                  column_titles, TEXT_MERGE_THRESHOLD):
                # 确定上下关系（用于保持文本顺序）
                if result_ref.boxes[i][0][1] < result_ref.boxes[j][0][1]:  # i在上
                    merged_text = result_ref.texts[i] + result_ref.texts[j]
                else:  # j在上
                    merged_text = result_ref.texts[j] + result_ref.texts[i]

                # 合并文本框
                merged_box = merge_boxes(result_ref.boxes[i], result_ref.boxes[j])

                # 移除旧项目，添加新项目
                result_ref.remove_items([i, j])
                result_ref.add_item(merged_text, merged_box)
                merged = True
                break
            j += 1

        if not merged:
            i += 1

    # 关键修改：按行划分文本框
    rows = split_into_rows(result_ref)
    if not rows:
        return company_name, "", {}, "", result_ref

    # 提取表格列标题及其中心点x坐标
    title_info = {}  # key: 列标题, value: 中心点x坐标
    for title in column_titles:
        best_text, best_score = find_closest_text(title, result_ref.texts)
        if best_score >= MATCH_THRESHOLD:
            title_idx = result_ref.texts.index(best_text)
            title_center_x = box_center(result_ref.boxes[title_idx])[0]
            title_info[title] = title_center_x
        else:
            title_info[title] = None

    # 提取表格数据（按行匹配列内容）
    table_data = {title: [] for title in column_titles}
    # 跳过标题所在行（假设标题在第一行，可根据实际调整）
    candidate_rows = rows[1:] if len(rows) > 1 else rows

    # 关键修改：过滤列名行
    data_rows = []
    for row_indices in candidate_rows:
        if not is_header_row(row_indices, column_titles, result_ref):
            data_rows.append(row_indices)

    for row_indices in data_rows:
        for title in column_titles:
            title_center_x = title_info[title]
            if title_center_x is None:
                table_data[title].append("")
                continue
            # 在当前行中寻找对应列的内容
            value = get_column_value_in_row(row_indices, title_center_x, result_ref)
            table_data[title].append(value)

    # 提取送货日期
    delivery_date = ""
    years = [str(year) for year in config["years"]] if isinstance(config["years"], list) else [str(config["years"])]
    for text in result_ref.texts:
        date_str = parse_date(text, years)
        if date_str:
            delivery_date = date_str
            break

    # 提取单号
    delivery_no = ""
    pattern = r'送货单号[：:\s]*([A-Za-z0-9\-]+)|单号(\w+)|委外商编码[：:\s]*([A-Za-z0-9\-]+)'
    for text in result_ref.texts:
        match = re.search(pattern, text)
        if match:
            # 依次检查每个分组，取第一个非空匹配结果
            for i in range(1, len(match.groups()) + 1):
                if match.group(i):
                    delivery_no = match.group(i).strip()
                    break
            break

    return company_name, delivery_date, table_data, delivery_no, result_ref


def process_pdf_to_excel(
        pdf_path: str,
        output_dir: str = "output",
        config_path: str = "Extracted_Info_caigou.json",
        progress_callback: Optional[Callable[[int, str], None]] = None
) -> str:
    """处理PDF文件并生成Excel（采购OCR识别）"""
    # 加载配置文件
    try:
        config = load_config_from_json(config_path)
    except Exception as e:
        logger.error(f"配置文件加载失败: {str(e)}")
        raise RuntimeError(f"配置文件加载失败: {str(e)}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理PDF文件
    logger.info(f"开始处理PDF文件: {pdf_path}")
    if progress_callback:
        progress_callback(15, "开始处理PDF")

    try:
        # 为每个任务创建独立的OCR实例
        logger.info("创建任务专用OCR实例...")
        task_ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=True,
            use_textline_orientation=False,
            text_rec_score_thresh=0.6,
            lang='ch'
        )

        logger.info("开始OCR识别...")
        if progress_callback:
            progress_callback(25, "进行OCR识别")

        # 直接处理PDF文件
        result = task_ocr.ocr(pdf_path)
        logger.info(f"OCR处理完成，共识别 {len(result)} 页")

        if progress_callback:
            progress_callback(45, f"OCR处理完成，共识别 {len(result)} 页")

    except Exception as e:
        logger.error(f"PDF处理失败: {str(e)}", exc_info=True)
        raise RuntimeError(f"PDF处理失败: {str(e)}")

    # 准备Excel文件路径 - 添加时间戳后缀
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{base_name}_processed_{timestamp}.xlsx")

    # 创建Excel写入器
    logger.info(f"创建Excel文件: {output_file}")
    if progress_callback:
        progress_callback(50, "创建Excel文件")

    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            has_valid_page = False  # 标记是否有有效页面
            total_pages = len(result)
            processed_pages = 0

            for page_idx, page in enumerate(result):
                if page is None:
                    logger.warning(f"跳过空页: {page_idx + 1}")
                    continue

                logger.info(f"处理第 {page_idx + 1} 页...")
                if progress_callback:
                    progress = 50 + int(40 * page_idx / total_pages)
                    progress_callback(progress, f"处理第 {page_idx + 1}/{total_pages} 页")

                try:
                    # 处理页面
                    company, date, table_data, delivery_no, result_ref = process_page(page, config)

                    # 如果表格数据为空，跳过此页
                    if not table_data:
                        logger.warning(f"第 {page_idx + 1} 页无有效表格数据")
                        continue

                    # 关键修改：过滤空行（含采购单号为空的行），仅保留有实际数据的行
                    df_data = []
                    # 先收集所有有效数据行（非空且采购单号不为空）
                    valid_rows_data = []
                    # 获取采购单号列的索引（如果存在）
                    purchase_order_col = "采购单号"
                    col_titles = list(table_data.keys())
                    has_purchase_order_col = purchase_order_col in col_titles
                    po_col_index = col_titles.index(purchase_order_col) if has_purchase_order_col else -1

                    for i in range(len(list(table_data.values())[0]) if table_data else 0):
                        row = [
                            company,  # 第一列：公司名称
                            date,  # 第二列：送货日期
                            delivery_no  # 第三列：单号
                        ]
                        po_value = ""
                        for col_idx, col_title in enumerate(table_data):
                            if i < len(table_data[col_title]):
                                cell_value = table_data[col_title][i].strip() if table_data[col_title][i] else ""
                                row.append(cell_value)
                                # 记录采购单号的值
                                if has_purchase_order_col and col_idx == po_col_index:
                                    po_value = cell_value
                            else:
                                row.append("")
                                # 采购单号列不存在或索引不匹配时设为空
                                if has_purchase_order_col and col_idx == po_col_index:
                                    po_value = ""

                        po_format_valid = False
                        if has_purchase_order_col and po_value.strip():
                            po_format_valid = re.fullmatch(r'^[A-Za-z0-9]+', po_value.strip()) is not None

                        # 过滤条件：1.除固定列外其他列至少有一个非空 2.如果有采购单号列则其值不能为空
                        condition1 = any(cell.strip() for cell in row[3:])
                        condition2 = (not has_purchase_order_col) or (po_value.strip() != "" and po_format_valid)
                        if condition1 and condition2:
                            valid_rows_data.append(row)

                    # 如果没有有效数据行，跳过此页
                    if not valid_rows_data:
                        logger.warning(f"第 {page_idx + 1} 页无有效数据行")
                        continue

                    # 生成序号（过滤后从1开始递增）
                    for seq, row in enumerate(valid_rows_data, start=1):
                        df_data.append([row[0], row[1], seq, row[2]] + row[3:])

                    # 调整列名顺序（公司名称 → 送货日期 → 序号 → 单号 → 原有列）
                    column_names = [
                                       "公司名称",
                                       "送货日期",
                                       "序号",
                                       "送货单号"
                                   ] + list(table_data.keys())

                    df = pd.DataFrame(df_data, columns=column_names)

                    # 写入Excel
                    sheet_name = f"Page_{page_idx + 1}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                    logger.info(f"写入工作表 {sheet_name}: 公司='{company}', 日期='{date}', 单号='{delivery_no}'")
                    has_valid_page = True  # 标记有有效页面
                    processed_pages += 1

                except Exception as e:
                    logger.error(f"处理第 {page_idx + 1} 页失败: {str(e)}", exc_info=True)
                    continue

            # 如果没有有效页面，创建空工作表
            if not has_valid_page:
                logger.warning("没有有效数据，创建空工作表")
                df = pd.DataFrame([["无有效数据"]], columns=["提示"])
                df.to_excel(writer, sheet_name="空页", index=False)

    except PermissionError as pe:
        logger.error(f"文件被占用错误: {str(pe)}", exc_info=True)
        raise PermissionError(f"文件 {output_file} 被其他程序占用，请关闭后重试")
    except Exception as e:
        logger.error(f"处理失败: {str(e)}", exc_info=True)
        raise RuntimeError(f"处理失败: {str(e)}")

    logger.info(f"处理完成，结果已保存至: {output_file}")
    if progress_callback:
        progress_callback(100, "处理完成")

    return output_file