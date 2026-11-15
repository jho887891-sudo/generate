# -*- coding: utf-8 -*-
"""
📘 通用年度降价计算脚本（含颜色标注 + 辅助列同步 + 基准价优先）
版本：v8.9.2 - 优化原单价提取优先级
----------------------------------------------------------
✅ 优化功能：
   - 严格按照优先级提取原单价：基准价表 > 模板原单价 > 自动推算
   - 只能基准价覆盖模板原单价
----------------------------------------------------------
✅ 保留功能：
   - 从需做的降价表中同步旧料号→辅助列、料品→料号名称、规格→料号规格到降价数据表中
   - 若某月"降本差额 = 0"（包括基准月），则该月的5列（单价、数量、差额、幅度、金额）全部不写。
   - 基准价优先（料号+供应商匹配，成功即不推算、不标蓝）
   - 自动推算仅在无匹配时执行并标蓝
   - 同料号多供应商：统一扫描标红
   - 降价月5列 + 物料编号标黄
   - 辅助列同步（旧料号→辅助列、料品→料号名称、规格→料号规格）
   - 强清洗键字段
   - 自动创建输出目录与详细日志
==========================================================
"""

import os, re
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from datetime import datetime
import logging

# ===================== 颜色定义 =====================
fill_yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
fill_red = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
fill_blue = PatternFill(start_color="00B0F0", end_color="00B0F0", fill_type="solid")


# ===================== 工具函数 =====================
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    return re.sub(r"[\s\u3000\xa0]+", "", s)


def normalize_col(col):
    if not isinstance(col, str):
        col = str(col)
    return col.strip().replace(" ", "").replace("\n", "").translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def build_cols_map(columns):
    return {normalize_col(c): c for c in columns}


def find_col_like(cols_map, target):
    """增强版列查找函数，支持更灵活的匹配"""
    key = normalize_col(target)

    # 1. 精确匹配
    for k, v in cols_map.items():
        if key == k:
            return v

    # 2. 包含匹配（目标在列名中）
    for k, v in cols_map.items():
        if key in k:
            return v

    # 3. 反向包含匹配（列名在目标中）
    for k, v in cols_map.items():
        if k in key:
            return v

    # 4. 部分匹配（针对带日期前缀的情况）
    for k, v in cols_map.items():
        # 如果列名包含目标关键词
        if any(word in k for word in ["原单价", "原价", "单价"]):
            return v

    return None


def find_orig_price_column(cols_map):
    """专门查找原单价列的函数"""
    # 多种可能的原单价列名模式
    patterns = ["原单价", "原价", "单价"]

    for pattern in patterns:
        col = find_col_like(cols_map, pattern)
        if col:
            return col

    # 如果以上都没找到，尝试查找包含"单价"的列
    for col_name, original_name in cols_map.items():
        if "单价" in col_name and "原" in col_name:
            return original_name
        elif "单价" in col_name and not any(x in col_name for x in ["月单价", "降价", "差额"]):
            return original_name

    return None


def find_month_columns(df, month):
    cmap = build_cols_map(df.columns)
    m = f"{month}月"
    cols = {
        "price": find_col_like(cmap, f"{m}单价"),
        "qty": find_col_like(cmap, f"{m}入库数量"),
        "diff": find_col_like(cmap, f"{m}降本差额"),
        "rate": find_col_like(cmap, f"{m}降价幅度"),
        "save": find_col_like(cmap, f"{m}降本金额"),
    }
    return cols if all(cols.values()) else None


def main_calculation(raw_path, template_path, baseline_path, output_dir, progress_callback=None):
    """网页版适配的主计算函数"""

    def update_progress(progress, message):
        """更新进度回调"""
        if progress_callback:
            progress_callback(progress, message)
        else:
            logging.info(f"[进度 {progress}%] {message}")

    # 设置输出文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    FILE_OUTPUT = os.path.join(output_dir, f"降价结果_{timestamp}.xlsx")
    FILE_RAISE = os.path.join(output_dir, f"涨价的料号_{timestamp}.xlsx")
    FILE_DONE = os.path.join(output_dir, f"原表已完成_{timestamp}.xlsx")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # ===================== Step 0: 加载文件 =====================
    update_progress(10, "正在加载 Excel 文件...")
    raw = pd.read_excel(raw_path)
    tpl = pd.read_excel(template_path)
    base = pd.read_excel(baseline_path)

    # 处理模板文件格式
    tpl.columns = tpl.iloc[0]
    tpl = tpl.drop(index=0).reset_index(drop=True)

    # 调试：输出所有列名
    update_progress(15, "正在分析模板文件列结构...")
    tpl_cols = build_cols_map(tpl.columns)
    logging.info("模板文件列名:")
    for norm_col, orig_col in tpl_cols.items():
        logging.info(f"  {norm_col} -> {orig_col}")

    # 使用增强的原单价列查找函数
    col_orig_price = find_orig_price_column(tpl_cols)
    if not col_orig_price:
        # 如果还是找不到，尝试更宽松的匹配
        for col_name, original_name in tpl_cols.items():
            if "单价" in col_name:
                col_orig_price = original_name
                update_progress(15, f"使用备选单价列: {col_orig_price}")
                break

    if not col_orig_price:
        raise ValueError("❌ 模板中未找到包含'原单价'的列！请检查模板文件结构。")

    base_cols = build_cols_map(base.columns)
    col_code = find_col_like(base_cols, "料号")
    col_base_price = find_col_like(base_cols, "基准价")
    col_vendor = find_col_like(base_cols, "供应商")

    if not col_code or not col_base_price or not col_vendor:
        raise ValueError("❌ 基准价表缺少必要列！")

    update_progress(20, f"已识别列：原单价={col_orig_price}，基准价={col_base_price}，厂商={col_vendor}")

    # ===================== Step 1: 清洗 =====================
    update_progress(30, "正在清洗数据...")
    raw["业务日期"] = pd.to_datetime(raw["业务日期"], errors="coerce")
    raw["料号"] = raw["料号"].apply(clean_text)
    raw["供应商"] = raw["供应商"].apply(clean_text)
    raw["订货价格(交易币种)"] = pd.to_numeric(raw["订货价格(交易币种)"], errors="coerce")
    raw["实收数量(计价单位)"] = pd.to_numeric(raw["实收数量(计价单位)"], errors="coerce")

    tpl["物料编号"] = tpl["物料编号"].apply(clean_text)
    tpl["厂商"] = tpl["厂商"].apply(clean_text)
    tpl[col_orig_price] = pd.to_numeric(tpl[col_orig_price], errors="coerce")

    base[col_code] = base[col_code].apply(clean_text)
    base[col_vendor] = base[col_vendor].apply(clean_text)
    base[col_base_price] = pd.to_numeric(base[col_base_price], errors="coerce")

    # ===================== Step 2: 模板同步（新增功能：同步旧料号、料品、规格） =====================
    update_progress(40, "正在同步模板数据...")

    # 识别模板中的辅助列、料号名称、料号规格列
    col_aux = find_col_like(tpl_cols, "辅助列")
    col_name = find_col_like(tpl_cols, "料号名称")
    col_spec = find_col_like(tpl_cols, "料号规格")

    # 从原始数据中提取料号对应的描述信息
    raw_desc_map = {}
    raw_grouped_desc = raw.groupby("料号").first().reset_index()
    for _, row in raw_grouped_desc.iterrows():
        code = clean_text(row["料号"])
        old_code = clean_text(row.get("旧料号", ""))
        material_name = clean_text(row.get("料品", ""))
        spec = clean_text(row.get("规格", ""))

        if code:
            raw_desc_map[code] = {
                "old_code": old_code,
                "material_name": material_name,
                "spec": spec
            }

    # 同步模板行
    tpl_pairs = set(zip(tpl["物料编号"], tpl["厂商"]))
    for _, r in raw[["料号", "供应商"]].drop_duplicates().iterrows():
        code, vendor = r["料号"], r["供应商"]
        if (code, vendor) not in tpl_pairs:
            new_row = {c: np.nan for c in tpl.columns}
            new_row["物料编号"], new_row["厂商"] = code, vendor

            # 同步描述信息
            if code in raw_desc_map:
                desc_info = raw_desc_map[code]
                if col_aux and desc_info["old_code"]:
                    new_row[col_aux] = desc_info["old_code"]
                if col_name and desc_info["material_name"]:
                    new_row[col_name] = desc_info["material_name"]
                if col_spec and desc_info["spec"]:
                    new_row[col_spec] = desc_info["spec"]

            tpl = pd.concat([tpl, pd.DataFrame([new_row])], ignore_index=True)
            tpl_pairs.add((code, vendor))
            update_progress(40, f"新增模板行：{code} / {vendor}")
        else:
            # 更新已存在的行的描述信息
            match = tpl[(tpl["物料编号"] == code) & (tpl["厂商"] == vendor)]
            if not match.empty and code in raw_desc_map:
                i = match.index[0]
                desc_info = raw_desc_map[code]
                if col_aux and desc_info["old_code"]:
                    tpl.at[i, col_aux] = desc_info["old_code"]
                if col_name and desc_info["material_name"]:
                    tpl.at[i, col_name] = desc_info["material_name"]
                if col_spec and desc_info["spec"]:
                    tpl.at[i, col_spec] = desc_info["spec"]

    # ===================== Step 3: 构建基准价映射 =====================
    update_progress(50, "正在构建基准价映射...")
    base_map = {
        (r[col_code], r[col_vendor]): r[col_base_price]
        for _, r in base.iterrows() if pd.notna(r[col_base_price])
    }
    update_progress(60, f"基准价映射构建完成：{len(base_map)} 条")

    # ===================== Step 4: 主逻辑 =====================
    color_marks, raise_records, done_indices = [], [], set()
    grouped = list(raw.groupby(["料号", "供应商"]))
    total = len(grouped)

    update_progress(65, f"开始计算降价数据，共 {total} 个料号...")

    for idx, ((code, vendor), group) in enumerate(grouped):
        # 更新进度
        progress = 65 + int(25 * (idx / total))
        update_progress(progress, f"处理料号 {idx + 1}/{total}: {code} / {vendor}")

        group = group.sort_values("业务日期")
        match = tpl[(tpl["物料编号"] == code) & (tpl["厂商"] == vendor)]
        if match.empty:
            new_row = {c: np.nan for c in tpl.columns}
            new_row["物料编号"], new_row["厂商"] = code, vendor

            # 同步描述信息
            if code in raw_desc_map:
                desc_info = raw_desc_map[code]
                if col_aux and desc_info["old_code"]:
                    new_row[col_aux] = desc_info["old_code"]
                if col_name and desc_info["material_name"]:
                    new_row[col_name] = desc_info["material_name"]
                if col_spec and desc_info["spec"]:
                    new_row[col_spec] = desc_info["spec"]

            tpl = pd.concat([tpl, pd.DataFrame([new_row])], ignore_index=True)
            i = tpl.index[-1]
        else:
            i = match.index[0]
            # 确保已存在的行也有最新的描述信息
            if code in raw_desc_map:
                desc_info = raw_desc_map[code]
                if col_aux and desc_info["old_code"]:
                    tpl.at[i, col_aux] = desc_info["old_code"]
                if col_name and desc_info["material_name"]:
                    tpl.at[i, col_name] = desc_info["material_name"]
                if col_spec and desc_info["spec"]:
                    tpl.at[i, col_spec] = desc_info["spec"]

        # ===================== 修改点：严格按照优先级提取原单价 =====================
        is_auto_baseline = False

        # 第一优先级：基准价表
        if (code, vendor) in base_map:
            orig_price = base_map[(code, vendor)]
            update_progress(progress, f"使用基准价: {code} / {vendor} = {orig_price}")

        # 第二优先级：模板中原单价（仅在基准价找不到时使用）
        elif pd.notna(tpl.at[i, col_orig_price]) and tpl.at[i, col_orig_price] > 0:
            orig_price = tpl.at[i, col_orig_price]
            update_progress(progress, f"使用模板原单价: {code} / {vendor} = {orig_price}")

        # 第三优先级：自动推算（前两个都找不到时才使用）
        else:
            # 修改点：取第一个月中最先出现的价格（按业务日期排序取第一条记录）
            first_month = group["业务日期"].dt.month.min()
            first_month_group = group[group["业务日期"].dt.month == first_month]
            if not first_month_group.empty:
                # 按业务日期排序，取最早的一条
                first_month_sorted = first_month_group.sort_values("业务日期")
                orig_price = first_month_sorted.iloc[0]["订货价格(交易币种)"]
            else:
                # 如果第一个月没有数据（理论上不会，因为group一定有数据），则取整个group的第一条
                orig_price = group.iloc[0]["订货价格(交易币种)"]
            is_auto_baseline = True
            update_progress(progress, f"自动推算原单价: {code} / {vendor} = {orig_price}")

        # 写入原单价（只能基准价覆盖模板原单价）
        tpl.at[i, col_orig_price] = orig_price

        # 标记自动推算的单元格（标蓝）
        if is_auto_baseline:
            color_marks.append((i, "blue_orig"))

        # ===================== 月度循环（保持不变） =====================
        for month, mg in group.groupby(group["业务日期"].dt.month):
            cols = find_month_columns(tpl, month)
            if not cols:
                continue

            cheaper = mg[mg["订货价格(交易币种)"] < orig_price]
            higher = mg[mg["订货价格(交易币种)"] > orig_price]

            if not cheaper.empty:
                for price, pg in cheaper.groupby("订货价格(交易币种)"):
                    qty = pg["实收数量(计价单位)"].sum()
                    diff = orig_price - price
                    if diff <= 0:
                        continue  # 差额为0，不写任何列
                    rate = diff / orig_price if orig_price != 0 else 0
                    save = qty * diff
                    tpl.at[i, cols["price"]] = price
                    tpl.at[i, cols["qty"]] = qty
                    tpl.at[i, cols["diff"]] = diff
                    tpl.at[i, cols["rate"]] = rate
                    tpl.at[i, cols["save"]] = save
                    color_marks.append((i, f"yellow_month_{month}"))
                    done_indices.update(pg.index)

            elif month == group["业务日期"].dt.month.min():
                # 基准月 → 差额为0，不写5列
                pass

            if not higher.empty:
                raise_records.append(higher)

    # ===================== Step 4.9: 同料号多厂商标红 =====================
    update_progress(95, "检查同料号多供应商情况...")
    dup_codes = tpl["物料编号"][tpl["物料编号"].duplicated(keep=False)].unique()
    for code in dup_codes:
        for i in tpl.index[tpl["物料编号"] == code]:
            color_marks.append((i, "red_vendor"))

    # ===================== Step 5: 输出与上色 =====================
    update_progress(97, "保存结果文件...")
    tpl.to_excel(FILE_OUTPUT, index=False)
    if raise_records:
        pd.concat(raise_records, ignore_index=True).to_excel(FILE_RAISE, index=False)

    # 应用颜色标记
    wb = load_workbook(FILE_OUTPUT)
    ws = wb.active
    cols_list = [c.value for c in ws[1]]

    def col_letter(name):
        for c in ws[1]:
            if c.value == name:
                return c.column_letter
        return None

    for row_index, flag in color_marks:
        excel_row = row_index + 2
        if flag == "blue_orig":
            col = col_letter(col_orig_price)
            if col: ws[f"{col}{excel_row}"].fill = fill_blue
        elif flag == "red_vendor":
            col = col_letter("厂商")
            if col: ws[f"{col}{excel_row}"].fill = fill_red
        elif flag.startswith("yellow_month_"):
            month = int(flag.split("_")[-1])
            mcols = find_month_columns(pd.DataFrame(columns=cols_list), month)
            if mcols:
                for key in mcols.values():
                    col = col_letter(key)
                    if col: ws[f"{col}{excel_row}"].fill = fill_yellow
                col_code = col_letter("物料编号")
                if col_code: ws[f"{col_code}{excel_row}"].fill = fill_yellow

    wb.save(FILE_OUTPUT)
    wb.close()

    # 标记原表已完成的行
    if done_indices:
        wb = load_workbook(raw_path)
        ws = wb.active
        for idx in done_indices:
            excel_row = idx + 2
            for cell in ws[excel_row]:
                cell.fill = fill_yellow
        wb.save(FILE_DONE)
        wb.close()

    update_progress(100, "处理完成！")

    return {
        "result": FILE_OUTPUT,
        "raise": FILE_RAISE,
        "done": FILE_DONE
    }


if __name__ == "__main__":
    # 本地测试
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 测试路径
    FILE_RAW = "需做的降价表.xlsx"
    FILE_TEMPLATE = "2025降价数据.xlsx"
    FILE_BASELINE = "2024基准价.xlsx"
    OUTPUT_DIR = "结果输出"

    # 运行计算
    result = main_calculation(
        raw_path=FILE_RAW,
        template_path=FILE_TEMPLATE,
        baseline_path=FILE_BASELINE,
        output_dir=OUTPUT_DIR
    )

    print(f"✅ 处理完成！")
    print(f"📊 降价结果文件: {result['result']}")
    if os.path.exists(result['raise']):
        print(f"⚠️  涨价格号文件: {result['raise']}")
    print(f"📋 已完成标记文件: {result['done']}")