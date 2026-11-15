import numpy as np
import pandas as pd
import json
import re
import os
import logging
from paddleocr import PaddleOCR
from typing import List, Tuple, Dict, Any, Optional, Callable
import datetime
import fitz  # PyMuPDF

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OCRResultRef:
    """存储处理后的OCR结果"""
    __slots__ = ['texts', 'boxes']  # 使用__slots__限定类的实例只能有texts和boxes两个属性

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


def load_config_from_json(file_path: str) -> Dict:
    """从JSON文件加载配置信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"配置文件格式错误: {file_path}")
        raise


def calculate_box_tilt(box: np.ndarray) -> float:
    """计算文本框的倾斜程度"""
    # 计算左上角和右上角的垂直差异
    top_diff = abs(box[0][1] - box[1][1])
    # 计算右下角和左下角的垂直差异
    bottom_diff = abs(box[3][1] - box[2][1])
    # 返回最大垂直差异
    return max(top_diff, bottom_diff)


def should_merge_boxes(box1: np.ndarray, box2: np.ndarray,
                       text1: str, text2: str,
                       column_titles: List[str],
                       threshold: float = 15.0) -> bool:
    """判断两个文本框是否应该合并"""
    # 检查是否包含column_titles中的文本
    for title in column_titles:
        if (text_similarity(text1, title) > 0.5 or
                text_similarity(text2, title) > 0.5):
            return False

    # 检查文本框宽度，避免合并太窄的文本框
    box1_width = box1[1][0] - box1[0][0]
    box2_width = box2[1][0] - box2[0][0]
    min_width = min(box1_width, box2_width)

    # 如果有一个框特别窄，可能是单个字符，谨慎合并
    if min_width < 10:  # 宽度小于10像素的文本框
        return False

    # 计算两个框的垂直重叠程度
    y11, y12 = min(box1[:, 1]), max(box1[:, 1])
    y21, y22 = min(box2[:, 1]), max(box2[:, 1])

    # 计算垂直重叠比例
    y_overlap = max(0, min(y12, y22) - max(y11, y21))
    min_height = min(y12 - y11, y22 - y21)
    overlap_ratio = y_overlap / min_height if min_height > 0 else 0

    # 如果垂直重叠比例高，则更可能是同一行
    if overlap_ratio > 0.6:  # 重叠超过60%
        # 计算水平距离
        x_center1 = np.mean(box1[:, 0])
        x_center2 = np.mean(box2[:, 0])
        horizontal_dist = abs(x_center1 - x_center2)

        # 放宽同一行文本的水平距离阈值
        return horizontal_dist < threshold * 3  # 增加阈值

    # 原始的下方判断逻辑
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
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',  # 2025年12月18日
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


def post_process_table_data(table_data):
    """后处理表格数据，确保同一行的数据正确对应"""
    # 找出最长的列
    max_rows = max(len(v) for v in table_data.values())

    # 创建一个新的表格数据字典
    processed_data = {title: [""] * max_rows for title in table_data.keys()}

    # 对于每一列，尝试将数据分配到正确的行
    for title, values in table_data.items():
        # 尝试根据文本内容判断行对应关系
        for i, value in enumerate(values):
            # 简单分配：假设顺序基本正确
            if i < max_rows:
                processed_data[title][i] = value

    return processed_data


def filter_abnormal_data(table_data):
    """过滤异常数据，如单个字符或明显错误的数据"""
    filtered_data = {}

    for title, values in table_data.items():
        filtered_values = []
        for value in values:
            # 过滤掉太短的文本（可能是单个字符错误识别）
            if len(value.strip()) <= 1 and not value.isdigit():
                continue
            # 过滤掉明显不是数据的文本
            if value in ["汇总", "委外商", "制单人", "仓库", "：", ":", " "]:
                continue
            filtered_values.append(value)
        filtered_data[title] = filtered_values

    return filtered_data


def process_page(ocr_page_result: Dict, config: Dict) -> Tuple[str, str, Dict, str]:
    """处理单页OCR结果"""
    # Step 1: 提取文本和文本框
    texts = ocr_page_result['rec_texts']
    boxes = ocr_page_result['rec_polys']
    result_ref = OCRResultRef(texts, boxes)

    # Step 2: 删除倾斜文本框
    tilt_indices = []
    for i, box in enumerate(result_ref.boxes):
        if calculate_box_tilt(box) > 30:  # 倾斜阈值
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
            if score > 0.5 and score > company_match_score:  # 匹配阈值
                company_name = company
                company_match_score = score
                if company_match_score == 1.0:
                    break

    # 如果没有匹配到公司，跳过后续处理
    if not company_name:
        return company_name, "", {}, ""
    column_titles = config["company_titles"].get(company_name, [])

    # Step 4: 合并相邻文本框
    i = 0
    while i < len(result_ref):
        j = i + 1
        merged = False

        while j < len(result_ref):
            if should_merge_boxes(result_ref.boxes[i], result_ref.boxes[j],
                                  result_ref.texts[i], result_ref.texts[j],
                                  column_titles, 15.0):  # 合并阈值
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

    for title in column_titles:
        # 查找最接近的列标题文本
        best_text, best_score = find_closest_text(title, result_ref.texts)

        if best_score < 0.5:  # 匹配阈值
            table_data[title] = []  # 没有找到符合要求的标题
            continue

        # 获取标题文本框索引
        title_idx = result_ref.texts.index(best_text)
        title_box = result_ref.boxes[title_idx]

        # 计算标题框的宽度和中心点
        title_width = title_box[1][0] - title_box[0][0]
        title_center_x = np.mean(title_box[:, 0])
        title_center_y = np.mean(title_box[:, 1])

        # 收集该列的数据
        column_values = []

        # 首先收集所有可能的候选框
        candidates = []
        for idx, box in enumerate(result_ref.boxes):
            # 跳过标题本身
            if np.array_equal(box, title_box):
                continue

            box_center_x = np.mean(box[:, 0])
            box_center_y = np.mean(box[:, 1])

            # 检查水平对齐（更严格的条件）
            horizontal_dist = abs(box_center_x - title_center_x)
            if horizontal_dist > 25:  # 对齐阈值
                continue

            # 检查垂直位置（必须在标题下方，且距离不能太近）
            if box_center_y <= title_center_y:
                continue

            # 检查垂直间距，避免太靠近标题的文本
            vertical_gap = box_center_y - title_center_y
            if vertical_gap < 20:  # 最小垂直间距
                continue

            # 检查文本框宽度，避免太窄的文本框（可能是单个字符）
            box_width = box[1][0] - box[0][0]
            if box_width < title_width * 0.3:  # 宽度阈值
                continue

            candidates.append((idx, box_center_y))

        # 按垂直位置排序
        candidates.sort(key=lambda x: x[1])

        # 按顺序添加到列值中
        for idx, _ in candidates:
            column_values.append(result_ref.texts[idx])

        table_data[title] = column_values

    # 后处理表格数据
    table_data = post_process_table_data(table_data)
    table_data = filter_abnormal_data(table_data)  # 添加异常数据过滤

    # Step 6: 提取送货日期
    delivery_date = ""
    # 将年份转换为字符串列表
    years = [str(year) for year in config["years"]] if isinstance(config["years"], list) else [str(config["years"])]
    for text in result_ref.texts:
        date_str = parse_date(text, years)
        if date_str:
            delivery_date = date_str
            break

    # Step 7: 提取单号
    delivery_no = ""  # 初始化单号
    # 正则表达式：匹配多种单号格式
    pattern = r'送货单号[：:\s]*([A-Za-z0-9\-]+)'
    for text in result_ref.texts:
        match = re.search(pattern, text)
        if match:
            # 依次检查每个分组，取第一个非空匹配结果
            for i in range(1, len(match.groups()) + 1):
                if match.group(i):
                    delivery_no = match.group(i).strip()
                    break
            break

    return company_name, delivery_date, table_data, delivery_no


def process_pdf_to_excel(pdf_path: str, output_dir: str = "output", config_path: str = "Extracted_Info.json",
                         progress_callback: Optional[Callable[[int, str], None]] = None) -> str:
    """处理PDF文件并生成Excel"""
    # 初始化PaddleOCR实例 - 使用与原始代码相同的配置
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
    logger.info(f"开始处理PDF文件: {pdf_path}")
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
    output_file = os.path.join(output_dir, f"{base_name}_processed_{timestamp}.xlsx")

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
                    company, date, table_data, delivery_no = process_page(page, config)

                    # 如果表格数据为空，跳过此页
                    if not table_data:
                        logger.warning(f"第 {page_idx + 1} 页无有效表格数据")
                        continue

                    # 创建数据框
                    max_rows = max((len(v) for v in table_data.values()), default=0)
                    df_data = []

                    # 生成序号列表(1到max_rows)
                    serial_numbers = list(range(1, max_rows + 1)) if max_rows > 0 else []

                    # 添加每行数据
                    for i in range(max_rows):
                        row = [
                            company,  # 第一列: 公司名称
                            date,  # 第二列: 送货日期
                            serial_numbers[i] if i < len(serial_numbers) else "",  # 第三列: 序号
                            delivery_no  # 第四列: 单号(所有行都显示同一个单号)
                        ]

                        # 添加原有表格数据列
                        for col_title in table_data:
                            if i < len(table_data[col_title]):
                                row.append(table_data[col_title][i])
                            else:
                                row.append("")

                        df_data.append(row)

                    # 调整列名顺序(公司名称→送货日期→序号→单号→原有列)
                    column_names = [
                                       "公司名称",
                                       "送货日期",
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


# 主入口点（用于测试）
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='处理PDF文件并生成Excel')
    parser.add_argument('pdf_path', type=str, help='PDF文件路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--config', type=str, default='Extracted_Info.json', help='配置文件路径')

    args = parser.parse_args()


    # 进度回调函数（用于测试）
    def print_progress(progress, message):
        print(f"[{progress}%] {message}")


    try:
        output_path = process_pdf_to_excel(
            pdf_path=args.pdf_path,
            output_dir=args.output,
            config_path=args.config,
            progress_callback=print_progress
        )
        print(f"处理完成，结果已保存至: {output_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        exit(1)