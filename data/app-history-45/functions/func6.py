import numpy as np
import pandas as pd
import json
import re
import os
import uuid
import logging
from paddleocr import PaddleOCR
from typing import List, Tuple, Dict, Any, Optional, Callable
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局参数
box_tilt_threshold = 20
match_threshold = 0.6
col_alignment_threshold = 40  # 缩小水平对齐阈值，提高精准度
row_gap_threshold = 50  # 缩小行间距阈值，避免跨行匹配
row_y_threshold = 30  # 同一行文本框y坐标允许的最大差值


class OCRResultRef:
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
        sorted_pairs = sorted(zip(self.text极, self.boxes), key=lambda x: x[1][0][1])
        self.texts, self.boxes = zip(*sorted_pairs)
        self.texts = list(self.texts)
        self.boxes = list(self.boxes)


def load_config_from_json(file_path: str) -> Dict:
    """从JSON文件加载配置信息"""
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
        if (text_similarity(text1, title) > match_threshold or
                text_similarity(text2, title) > match_threshold):
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
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ])


def text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（改进版）"""
    # 预处理：小写+去空格（保留原始字符）
    text1_clean = text1.lower().replace(' ', '')
    text2_clean = text2.lower().replace(' ', '')

    # 1. 完全匹配
    if text1_clean == text2_clean:
        return 1.0

    # 2. 对于编码类文本，使用更宽松的匹配
    # 检查是否包含数字和字母组合（类似编码）
    if re.search(r'[a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+', text1_clean) and \
            re.search(r'[a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+', text2_clean):
        # 对于编码类文本，只要有一部分匹配就给予较高分数
        if text1_clean in text2_clean or text2_clean in text1_clean:
            return 0.7

    # 3. 计算编辑距离相似度（核心方法）
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

    # 4. 计算Jaccard相似度（基于3-gram，平衡精度和粒度）
    def jaccard_similarity(s1, s2, n=3):
        def get_ngrams(s):
            return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}

        ngrams1 = get_ngrams(s1)
        ngrams2 = get_ngrams(s2)
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        return intersection / union if union > 0 else 0

    # 5. 动态权重融合（编辑距离为主，Jaccard为辅）
    lev_sim = levenshtein_similarity(text1_clean, text2_clean)
    jac_sim = jaccard_similarity(text1_clean, text2_clean)
    final_sim = 0.7 * lev_sim + 0.3 * jac_sim  # 权重可调整

    return final_sim


def find_closest_text(target: str, text_list: List[str]) -> Tuple[Optional[str], float]:
    """在文本列表中找到最接近目标文本的项"""
    best_match = None
    best_score = 0.0

    for text in text_list:
        score = text_similarity(target, text)
        if score > best_score:
            best_score = score
            best_match = text
        if best_score == 1.0:
            break

    return best_match, best_score


def parse_date(text: str, years: List[str]) -> Optional[str]:
    """解析日期并统一格式"""
    # 尝试匹配常见日期格式
    patterns = [
        r'(\d{4})[\-\/\.](\d{1,2})[\-\/\.](\d{1,2})',  # 2023-05-12 或 2023/05/12
        r'(\d{1,2})[\-\/\.](\d{1,2})[\-\/\.](\d{4})',  # 05-12-2023
        r'(\d{1,2})/(\d{1,2})/(\d{2})',  # 05/12/23
        r'(\d{4})年(\d{1,2})月(\极d{1,2})日',  # 2025年12月18日
        r'(\d{4})年(\d{1,2})月(\d{1,2})号',  # 2025年12月18号
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups[0]) == 4:  # 年份在前
                y, m, d = groups
            else:  # 年份在后
                m, d, y = groups
                if len(y) == 2:  # 两位年份的情况
                    y = f"20{y}" if int(y) < 50 else f"19{y}"

            # 统一为四位年份
            if len(y) != 4:
                continue

            # 格式化为统一格式: YYYY.MM.DD
            return f"{y}.{m.zfill(2)}.{d.zfill(2)}"

    # 如果文本中只有年份信息，检查是否在提供的年份列表中
    for year in years:
        if year in text:
            return f"{year}.01.01"  # 默认返回该年的第一天

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
        if box_y - current_y <= row_y_threshold:
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
        if x_diff < col_alignment_threshold and x_diff < min_x_diff:
            min_x_diff = x_diff
            best_match_idx = idx

    return result_ref.texts[best_match_idx] if best_match_idx is not None else ""


def is_header_row(row_indices: List[int], column_titles: List[str], result_ref: OCRResultRef) -> bool:
    """判断当前行是否为列名行"""
    row_texts = [result_ref.texts[idx].strip() for idx in row_indices]
    match_count = 0
    for title in column_titles:
        for text in row_texts:
            if text_similarity(text, title) > match_threshold:
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
        if calculate_box_tilt(box) > box_tilt_threshold:
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
            if score > match_threshold and score > company_match_score:
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
                                  column_titles, text_merge_threshold):
                # 确定上下关系（用于保持文本顺序）
                if result_ref.boxes[i][0][1] < result_ref.boxes[j][0][1]:  # i在上
                    merged_text = result_ref.texts[i] + result_ref.texts[j]
                else:  # j在上
                    merged_text = result_ref.texts[j] + result_ref.texts[i]

                # 合并文本框（merge_boxes函数会自动处理坐标范围）
                merged_box = merge_boxes(result_ref.boxes[i], result_ref.boxes[j])

                # 移除旧项目，添加新项目
                result_ref.remove_items([i, j])
                result_ref.add_item(merged_text, merged_box)
                merged = True
                break
            j += 1

        if not merged:
            i += 1

    # Step 5: 提取表格数据
    table_data = {}

    # 关键修改：按行划分文本框
    rows = split_into_rows(result_ref)
    if not rows:
        return company_name, "", {}, "", result_ref

    # 提取表格列标题及其中心点x坐标
    title_info = {}  # key: 列标题, value: 中心点x坐标
    for title in column_titles:
        # 查找最接近的列标题文本
        best_text, best_score = find_closest_text(title, result_ref.texts)

        if best_score >= match_threshold:
            title_idx = result_ref.texts.index(best_text)
            title_center_x = box_center(result_ref.boxes[title_idx])[0]
            title_info[title] = title_center_x
        else:
            title_info[title] = None

    # 提取表格数据
    table_data = {title: [] for title in column_titles}
    # 跳过标题所在行
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
    pattern = r'送货单号：([A-Za-z0-9]+)|单号(\w+)|委外商编码：([A-Za-z0-9\-]+)'
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


def process_purchase_ocr(pdf_path: str, output_dir: str = "output", config_path: str = "Extracted_Info_caigou.json",
                         progress_callback: Optional[Callable[[int, str], None]] = None) -> str:
    """处理采购单PDF文件并生成Excel"""
    # 初始化PaddleOCR实例
    logger.info("初始化PaddleOCR引擎...")
    if progress_callback:
        progress_callback(5, "初始化OCR引擎")

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=True,
        use_textline_orientation=False,
        text_rec_score_thresh=0.6,
        lang='ch'
    )

    # 加载配置文件
    logger.info(f"加载配置文件: {config_path}")
    if progress_callback:
        progress_callback(10, "加载配置文件")

    try:
        config = load_config_from_json(config_path)
    except Exception as e:
        logger.error(f"配置文件加载失败: {str(e)}")
        raise RuntimeError(f"配置文件加载失败: {str(e)}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"创建输出目录: {output_dir}")

    # 处理PDF文件
    logger.info(f"开始处理采购单PDF文件: {pdf_path}")
    if progress_callback:
        progress_callback(15, "开始处理PDF")

    try:
        # 使用OCR处理PDF
        result = ocr.ocr(pdf_path)
        logger.info(f"OCR处理完成，共识别 {len(result)} 页")
        if progress_callback:
            progress_callback(30, f"OCR处理完成，共识别 {len(result)} 页")
    except Exception as e:
        logger.error(f"OCR处理失败: {str(e)}", exc_info=True)
        raise RuntimeError(f"OCR处理失败: {str(e)}")

    # 准备Excel文件路径 - 添加时间戳后缀
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # 获取当前时间并格式化为字符串（YYYYMMDD_HHMMSS）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 在文件名中添加时间戳
    output_file = os.path.join(output_dir, f"{base_name}_purchase_processed_{timestamp}.xlsx")

    # 创建Excel写入器
    logger.info(f"创建Excel文件: {output_file}")
    if progress_callback:
        progress_callback(35, "创建Excel文件")

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
                    progress = 40 + int(50 * page_idx / total_pages)
                    progress_callback(progress, f"处理第 {page_idx + 1}/{total_pages} 页")

                try:
                    # 处理页面
                    company, date, table_data, delivery_no, result_ref = process_page(page, config)

                    # 如果表格数据为空，跳过此页
                    if not table_data:
                        logger.warning(f"第 {page_idx + 1} 页无有效表格数据")
                        continue

                    # 创建数据框
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
                            company,  # 第一列: 公司名称
                            date,  # 第二列: 采购日期
                            delivery_no  # 第三列: 单号
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

                        # 新增：检查采购单号格式是否符合[A-Za-z0-9]+（仅包含字母和数字）
                        po_format_valid = False
                        if has_purchase_order_col and po_value.strip():
                            # 使用正则表达式验证整个字符串是否仅由字母和数字组成
                            po_format_valid = re.fullmatch(r'^[A-Za-z0-9]+$', po_value.strip()) is not None

                        # 过滤条件：
                        # 1.除固定列外其他列至少有一个非空
                        # 2.如果有采购单号列则其值不能为空且格式正确（新增格式检查）
                        condition1 = any(cell.strip() for cell in row[3:])
                        condition2 = (not has_purchase_order_col) or (po_value.strip() != "" and po_format_valid)
                        if condition1 and condition2:
                            valid_rows_data.append(row)

                    # 生成序号（过滤后从1开始递增）
                    for seq, row in enumerate(valid_rows_data, start=1):
                        df_data.append([row[0], row[1], seq, row[2]] + row[3:])

                    # 调整列名顺序(公司名称→采购日期→序号→单号→原有列)
                    column_names = [
                                       "公司名称",
                                       "采购日期",
                                       "序号",
                                       "单号"
                                   ] + list(table_data.keys())  # 拼接原有列名

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