from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import os
import torch

# ============================================================
# 4B 小尺寸端到端训练配置
# 路线：
# Qwen3-4B-Instruct-2507-bnb-4bit
#   -> Stage1 书本知识 LoRA
#   -> Stage2 参数 + 气孔溯源 LoRA
#   -> Region-aware 多模态训练
#
# 注意：
# 1. 多模态阶段默认只挂 Stage2 adapter。
# 2. Stage2 adapter 已经是在 Stage1 adapter 上继续训练得到的结果。
# 3. 不要同时再挂 Stage1，否则等于重复叠加第一次 LoRA。
# ============================================================

PROJECT_ROOT = Path(os.environ.get("DIECAST_PROJECT_ROOT", "/home/T7/wkq/end-to-end"))

BASE_DIR = Path(
    os.environ.get(
        "DIECAST_REGION_ROUTE_ROOT",
        str(PROJECT_ROOT / "diecast_region_route"),
    )
)

MODEL_PATH = os.environ.get(
    "DIECAST_YOLO_WEIGHT_PATH",
    "/home/T7/wkq/end-to-end/yolo-X-data/best_LR_75.pt",
)

CALIBRATION_JSON = os.environ.get(
    "DIECAST_CALIBRATION_JSON",
    "/home/T7/wkq/end-to-end/yolo-X-data/calibration_result.json",
)

BASE_MODEL_PATH = os.environ.get(
    "DIECAST_LLM_BASE_PATH",
    str(PROJECT_ROOT / "models" / "Qwen3-4B-stage1-book-merged-4bit"),
)

STAGE1_ADAPTER_PATH = os.environ.get(
    "DIECAST_STAGE1_ADAPTER_PATH",
    str(PROJECT_ROOT / "runs" / "qwen3_4b_stage1_expert_clean_best" / "best_adapter"),
)

STAGE2_ADAPTER_PATH = os.environ.get(
    "DIECAST_STAGE2_ADAPTER_PATH",
    str(PROJECT_ROOT / "runs" / "qwen3_4b_stage2_from_merged_Third_sft" / "best_adapter"),
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RegionAwareMMConfig:
    project_root: Path = BASE_DIR
    yolo_weight_path: str = MODEL_PATH
    calibration_json: str = CALIBRATION_JSON

    llm_base_path: str = BASE_MODEL_PATH
    llm_stage1_adapter_path: str = STAGE1_ADAPTER_PATH
    llm_stage2_adapter_path: str = STAGE2_ADAPTER_PATH

    # =========================================================
    # 4B 端到端训练数据
    # =========================================================
    train_jsonl: Path = Path(
        os.environ.get(
            "DIECAST_MM_TRAIN_JSONL",
            str(PROJECT_ROOT / "model_data" / "stage2_second_sft" / "stage2_high_impact_gas_pore_train.jsonl"),
        )
    )

    val_jsonl: Path = Path(
        os.environ.get(
            "DIECAST_MM_VAL_JSONL",
            str(PROJECT_ROOT / "model_data" / "stage2_second_sft" / "stage2_high_impact_gas_pore_val.jsonl"),
        )
    )

    test_jsonl: Path = Path(
        os.environ.get(
            "DIECAST_MM_TEST_JSONL",
            str(PROJECT_ROOT / "model_data" / "stage2_second_sft" / "stage2_high_impact_gas_pore_val.jsonl"),
        )
    )

    output_dir: Path = Path(
        os.environ.get(
            "DIECAST_MM_OUTPUT_DIR",
            str(PROJECT_ROOT / "runs" / "qwen3_4b_region_mm_from_stage2"),
        )
    )

    device: str = DEVICE

    # =========================================================
    # 视觉侧配置：控制显存和速度
    # =========================================================
    vision_image_size: int = 512
    yolo_feature_layer: int = -2
    vision_feat_dim: int = 256
    max_views_per_sample: int = 2
    max_rois_per_view: int = 3
    roi_output_size: int = 1
    freeze_yolo: bool = True

    # =========================================================
    # LLM / LoRA 配置
    # 当前路线：
    #   base 8B + Stage2 adapter
    #   继续训练 Stage2 adapter + 多模态新增模块
    # =========================================================
    load_in_4bit: bool = True
    max_seq_length: int = 2048

    continue_train_stage1_adapter: bool = False
    continue_train_stage2_adapter: bool = True

    freeze_llm_base: bool = True
    train_new_lora_adapter: bool = False

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # =========================================================
    # 多模态投影 / 融合配置
    # =========================================================
    projector_hidden_dim: int = 768
    projector_dropout: float = 0.1

    numeric_token_count: int = 2
    region_hidden_dim: int = 384
    view_vocab_size: int = 16

    max_defect_tokens: int = 8
    max_prefix_tokens: int = 48

    use_external_summary_tokens: bool = False
    use_coordinate_region_summary: bool = True
    use_prefix_fusion_mapper: bool = True
    fusion_token_count: int = 4
    fusion_num_heads: int = 4
    use_visual_summary_text: bool = False

    max_answer_tokens: int = 512
    max_user_tokens: int = 768

    # =========================================================
    # 训练参数
    # A6000 48GB + 4B：这个配置比较稳
    # =========================================================
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 2

    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    log_every: int = 10
    num_workers: int = 0
    seed: int = 42

    # =========================================================
    # 问题类型增强
    # 已按“气孔 + 高影响区域”逻辑改写
    # =========================================================
    use_question_type_hint: bool = True
    inject_question_type_to_instruction: bool = True
    inject_question_type_to_info_text: bool = False

    question_type_order: List[str] = field(default_factory=lambda: [
        "effect_analysis",
        "cause_explanation",
        "relation_mechanism",
        "adjustment_validation",
        "inspection_identification",
        "yes_no_judgment",
        "binary_decision",
        "open_diagnosis",
    ])

    question_type_hints: Dict[str, str] = field(default_factory=lambda: {
        "effect_analysis": (
            "当前问题属于影响分析类。回答时重点说明参数或条件变化对气孔数量、区域极值和高影响区域风险的影响；"
            "必须输出【结论】【溯源分析】【建议方向】三部分；背景和未知区域不参与主判断。"
        ),
        "cause_explanation": (
            "当前问题属于原因解释类。回答时重点说明气孔为什么会出现，并围绕充型稳定性、卷气倾向、压力传递和排气匹配建立主因链；"
            "必须输出【结论】【溯源分析】【建议方向】三部分；背景和未知区域不参与主判断。"
        ),
        "relation_mechanism": (
            "当前问题属于参数/机理关系类。回答时重点说明参数之间的关系，以及这种关系为什么会影响气孔在牙孔区域、焊接区域、重点区域等高影响区域的风险；"
            "必须输出【结论】【溯源分析】【建议方向】三部分。"
        ),
        "adjustment_validation": (
            "当前问题属于调整/验证类。回答时重点说明先看哪些参数链、如何对比调整前后样本、如何观察高影响区域气孔是否收敛；"
            "建议只给宏观方向，不给具体数值设定。"
        ),
        "inspection_identification": (
            "当前问题属于探伤识别/区分类。回答时重点说明气孔影像表现、区域分布、区域最大面积和区域最大等效直径，不做缩孔、缩松、冷隔、夹杂等分类判断。"
        ),
        "yes_no_judgment": (
            "当前问题属于是非判断类。回答时必须先明确给出判断，再说明依据和限制条件；判断依据应来自已给区域统计、高影响区域和压铸参数链。"
        ),
        "binary_decision": (
            "当前问题属于二选一判断类。回答时必须明确更偏向哪一项，并说明另一项为什么不是当前优先判断；"
            "牙孔区域、焊接区域、重点区域属于高影响区域。"
        ),
        "open_diagnosis": (
            "当前问题属于开放诊断类。回答时先看高影响区域，再结合区域缺陷统计、区域最大面积、区域最大等效直径和关键参数链展开；"
            "一般区域只作为整体波动参考，背景和未知区域忽略。"
        ),
    })

    # =========================================================
    # 区域字段
    # 背景、未知区域不作为训练主区域。
    # =========================================================
    region_order: List[str] = field(default_factory=lambda: [
        "一般区域",
        "牙孔区域",
        "焊接区域",
        "重点区域",
        "厚大区域",
    ])

    high_impact_regions: List[str] = field(default_factory=lambda: [
        "牙孔区域",
        "焊接区域",
        "重点区域",
    ])

    ignored_regions: List[str] = field(default_factory=lambda: [
        "背景",
        "未知区域",
        "未知区域(0)",
        "未知(0)",
    ])

    # =========================================================
    # 参数字段
    # =========================================================
    numeric_fields: List[str] = field(default_factory=lambda: [
        "慢速速度",
        "高速速度",
        "料饼厚度",
        "铸造压力",
        "循环周期",
        "建压时间",
        "射料结束位置",
        "压射压力",
        "高速行程",
        "高速时间",
        "慢速行程",
        "环境温度",
        "环境湿度",
    ])