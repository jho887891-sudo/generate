from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import e2e_token_text_only_300q_multisize_graphrag as legacy  # noqa: E402


DEFAULT_CONFIG_JSON = SCRIPT_DIR / "configs" / "e2e_token_text_config_4b_300q_graphrag.json"
CONFIG_JSON = os.environ.get("E2E_CONFIG_JSON", str(DEFAULT_CONFIG_JSON))


SYSTEM_PROMPT = f"""{legacy.GENERAL_SYSTEM_PROMPT}

当前实验方法：SVP-Text。
图片先经过视觉层识别并转写为视觉层文字总结；语言模型只接收视觉层文字总结、压铸参数、问题类型提示和可选图谱证据。

实验边界：
1. 不加载多模态映射层权重，不输入视觉/区域 prefix token。
2. 不输入参数 prefix token，压铸参数只通过提示词文本提供。
3. 每个基座只挂载对应的 `_stage2_from_merged_Third_sft` LoRA。
4. 不要输出“token_text”“多模态token”“视觉/区域前缀token”等内部实验字样。
请直接输出【结论】【溯源分析】【建议方向】。"""


MODE_DEFINITIONS = {
    "svp_text": {
        "mode_desc": "SVP-Text：视觉层文字总结 + 压铸参数 + 问题，不输入多模态 prefix token",
        "include_visual_text": True,
        "use_visual_prefix": False,
        "use_region_prefix": False,
        "use_numeric_prefix": False,
    }
}


def normalize_adapter_path(path_text: str) -> str:
    path = Path(path_text)
    candidates = [path, path / "best_adapter", path / "adapter", path / "llm_adapter"]
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return str(candidate)
    return str(path)


def build_svp_text_prompt(
    cfg_json: Dict[str, Any],
    question: str,
    visual_fact_block: str,
    graph_context: str,
    question_type_hint: str,
) -> str:
    info_text = legacy.build_info_text_for_prompt(cfg_json)
    graph_context = legacy.sanitize_graph_context_for_prompt(
        graph_context,
        max_chars=int(cfg_json.get("prompt_graph_max_chars", 700)),
    )
    graph_block = f"\n【图谱辅助证据】\n{graph_context}\n" if graph_context else ""
    qtype_block = f"\n【当前问题类型提示】\n{question_type_hint}\n" if question_type_hint else ""
    return f"""{visual_fact_block}

【压铸参数】
{info_text}
{qtype_block}{graph_block}
【问题】
{question.strip()}

【作答边界】
1. 当前输入形式为 SVP-Text：视觉信息只来自视觉层文字总结，压铸参数只通过文本提供。
2. 必须结合区域气孔数量、区域最大面积、区域最大等效直径和压铸参数链进行溯源。
3. 缺陷统一按气孔处理，不引入缩孔、缩松、裂纹、冷隔、夹杂、夹渣等非气孔缺陷。
4. 图谱证据如存在，只能作为机理辅助，不能覆盖当前样本视觉统计和参数事实。
5. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。
6. 不要输出 token_text、多模态token、视觉/区域前缀token、规则编号、调试信息或中间字段。

【必须遵守的输出格式】
第一行必须是【结论】。
【结论】必须直接给当前样本的判断，包含排查优先级或风险优先级。
【溯源分析】下面必须写 1. 2. 3. 三个编号点：1解释区域证据，2分析参数链，3给出综合优先级。
【建议方向】下面必须写 1. 2. 3. 三个编号点：1写优先排查链路，2写联动复核，3写同料号、同视角、同工况验证。

现在开始作答。"""


class SVPTextModel:
    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        load_in_4bit: bool = True,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = normalize_adapter_path(adapter_path)
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        print(f"[llm] loading base model: {self.base_model}", flush=True)
        print(f"[llm] loading stage2 adapter: {self.adapter_path}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"

        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": 0} if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path, is_trainable=False)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        max_seq_length: int,
        generation_max_time_sec: float | None,
    ) -> str:
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_seq_length,
        )
        device = next(self.model.parameters()).device
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "use_cache": True,
        }
        if generation_max_time_sec and generation_max_time_sec > 0:
            kwargs["max_time"] = float(generation_max_time_sec)
        output_ids = self.model.generate(**kwargs)
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def load_svp_text_model(cfg_json: Dict[str, Any]) -> SVPTextModel:
    base_path = cfg_json.get("llm_base_path")
    adapter_path = cfg_json.get("llm_stage2_adapter_path")
    if not base_path:
        raise ValueError("配置中缺少 llm_base_path。")
    if not adapter_path:
        raise ValueError("配置中缺少 llm_stage2_adapter_path。")
    if "_stage2_from_merged_Third_sft" not in str(adapter_path):
        print(f"[llm][warn] adapter path does not contain expected suffix: {adapter_path}", flush=True)
    return SVPTextModel(
        base_model=str(base_path),
        adapter_path=str(adapter_path),
        load_in_4bit=bool(cfg_json.get("load_in_4bit", True)),
    )


def main() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    print(f"[main] using config: {CONFIG_JSON}", flush=True)
    cfg_json = legacy.load_json(CONFIG_JSON)
    cfg_json["enable_file_upload"] = False
    cfg_json["compare_modes"] = ["svp_text"]
    cfg_json["max_questions"] = int(cfg_json.get("max_questions", 300))
    cfg_json.setdefault("use_graph_rag", True)
    cfg_json.setdefault("save_intermediate_visual_files", False)
    cfg_json.setdefault("resume", True)

    questions = legacy.get_test_questions(cfg_json)
    if not questions:
        raise ValueError("配置中需要提供 question/questions/questions_file。")
    prompt_info_text = legacy.build_info_text_for_prompt(cfg_json)
    if not prompt_info_text:
        raise ValueError("配置中缺少 params/info_text，无法生成【压铸参数】提示词。")

    print(f"[config] question_count = {len(questions)}", flush=True)
    print(f"[config] model_size = {cfg_json.get('model_size', cfg_json.get('size', 'unknown'))}", flush=True)
    print(f"[config] llm_base_path = {cfg_json.get('llm_base_path', '')}", flush=True)
    print(f"[config] llm_stage2_adapter_path = {cfg_json.get('llm_stage2_adapter_path', '')}", flush=True)
    print("[config] mode forced to SVP-Text only; multimodal heads/prefix tokens are disabled", flush=True)

    print("[main] stage 1/2: vision -> visual text summary", flush=True)
    image_paths = legacy.collect_image_paths(cfg_json)
    vision = legacy.VisualStageRunner(cfg_json)
    bundle = vision.run(image_paths)
    bundle = legacy.filter_visual_bundle(bundle)
    print(f"\n[done] visual output: {cfg_json['visual_output_dir']}", flush=True)
    vision.release()
    del vision
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    print("\n" + "=" * 100)
    print("【SVP-Text 视觉层摘要：仅保留有效工艺区域】")
    print(json.dumps(bundle["aggregate_summary"], ensure_ascii=False, indent=2), flush=True)

    legacy.wait_for_gpu(min_free_gb=float(cfg_json.get("min_free_gpu_gb", 40.0)), timeout_sec=180)
    print("[main] stage 2/2: text-only LLM + stage2 LoRA", flush=True)

    model = load_svp_text_model(cfg_json)
    model.load()

    result_dir = Path(cfg_json["result_output_dir"])
    legacy.ensure_dir(result_dir)
    visual_fact_block = legacy.build_visual_fact_block(bundle["aggregate_summary"])
    legacy.save_text(result_dir / "params_prompt_block.txt", prompt_info_text)
    legacy.save_text(result_dir / "visual_fact_block.txt", visual_fact_block)
    legacy.save_json(result_dir / "visual_summary_filtered.json", bundle["aggregate_summary"])

    cases: List[Dict[str, Any]] = []
    graph_debug_all: List[Dict[str, Any]] = []

    for idx, question in enumerate(questions, start=1):
        q_dir = result_dir / f"q{idx:02d}"
        case_file = q_dir / "case_result.json"
        if bool(cfg_json.get("resume", True)) and case_file.exists():
            try:
                cached_case = legacy.load_json(case_file)
                cases.append(cached_case)
                graph_debug_all.append(cached_case.get("graph_debug", {}))
                print(f"[resume] Q{idx:02d} already exists, skip generation: {case_file}", flush=True)
                continue
            except Exception as exc:
                print(f"[resume][warn] failed to load cached Q{idx:02d}, regenerate: {exc}", flush=True)

        question_type, question_type_hint, focus_regions, mechanism_candidates = legacy.build_question_type_hint(question)
        print(f"[main] Q{idx:02d} detected question_type: {question_type}", flush=True)

        graph_context = ""
        graph_debug = {
            "enabled": bool(cfg_json.get("use_graph_rag", False)),
            "available": False,
            "status": "disabled",
            "seed_terms": [],
            "lucene_query": "",
            "seed_entities": [],
            "defect_names": [],
            "rule_ids": [],
            "rule_hits": 0,
            "evidence_hits": 0,
            "question_type": question_type,
            "focus_regions": focus_regions,
            "mechanism_candidates": mechanism_candidates,
        }
        if cfg_json.get("use_graph_rag", False):
            try:
                graph_context, graph_debug_base = legacy.retrieve_graph_context(
                    cfg_json=cfg_json,
                    info_text=prompt_info_text,
                    question=question,
                    graph_top_k=int(cfg_json.get("graph_top_k", 10)),
                )
                graph_debug.update(graph_debug_base)
                graph_debug["question_type"] = question_type
                graph_debug["focus_regions"] = focus_regions
                graph_debug["mechanism_candidates"] = mechanism_candidates
            except Exception as exc:
                graph_debug["status"] = f"graph retrieval failed: {exc}"
                print(f"[graph][warn][Q{idx:02d}] retrieve_graph_context failed: {exc}", flush=True)
        legacy.print_graph_evidence(idx, graph_context, graph_debug)
        graph_debug_all.append(graph_debug)

        user_prompt = build_svp_text_prompt(
            cfg_json=cfg_json,
            question=question,
            visual_fact_block=visual_fact_block,
            graph_context=graph_context if cfg_json.get("use_graph_rag", False) else "",
            question_type_hint=question_type_hint,
        )
        t0 = time.time()
        print(
            f"[llm] Q{idx:02d} mode=svp_text start generate | max_new_tokens={int(cfg_json.get('max_new_tokens', 512))}",
            flush=True,
        )
        raw_answer = model.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=int(cfg_json.get("max_new_tokens", 512)),
            max_seq_length=int(cfg_json.get("max_seq_length", 2048)),
            generation_max_time_sec=float(cfg_json.get("generation_max_time_sec", 180)),
        )
        print(f"[llm] Q{idx:02d} mode=svp_text generate finished in {time.time() - t0:.1f}s", flush=True)
        answer = legacy.normalize_output(raw_answer)

        mode_result = {
            "mode": "svp_text",
            "mode_desc": MODE_DEFINITIONS["svp_text"]["mode_desc"],
            "include_visual_text": True,
            "use_visual_prefix": False,
            "use_region_prefix": False,
            "use_numeric_prefix": False,
            "raw_answer": raw_answer,
            "final_answer": answer,
            "one_pass_generation": True,
            "user_prompt": user_prompt,
            "instruction": SYSTEM_PROMPT,
        }
        case = {
            "question_index": idx,
            "workpiece_id": cfg_json.get("workpiece_id", str(cfg_json.get("sample_id", "unknown"))),
            "model_size": cfg_json.get("model_size", cfg_json.get("size", "unknown")),
            "llm_base_path": cfg_json.get("llm_base_path", ""),
            "llm_stage2_adapter_path": normalize_adapter_path(str(cfg_json.get("llm_stage2_adapter_path", ""))),
            "weight_path": "",
            "question": question,
            "question_type": question_type,
            "question_type_hint": question_type_hint,
            "focus_regions": focus_regions,
            "mechanism_candidates": mechanism_candidates,
            "visual_summary": bundle["aggregate_summary"],
            "visual_fact_block": visual_fact_block,
            "graph_context": graph_context,
            "graph_debug": graph_debug,
            "params": cfg_json["params"],
            "image_paths": bundle["image_paths"],
            "modes": {"svp_text": mode_result},
        }

        legacy.ensure_dir(q_dir)
        legacy.save_json(q_dir / "graph_debug.json", graph_debug)
        legacy.save_text(q_dir / "graph_context.txt", graph_context or "未检索到图谱证据。")
        mode_dir = q_dir / "svp_text"
        legacy.ensure_dir(mode_dir)
        legacy.save_json(mode_dir / "e2e_result.json", mode_result)
        legacy.save_text(mode_dir / "final_answer.txt", answer)
        legacy.save_text(mode_dir / "raw_answer.txt", raw_answer)
        legacy.save_text(mode_dir / "user_prompt.txt", user_prompt)
        legacy.save_json(q_dir / "case_result.json", case)
        cases.append(case)

        print("\n" + "=" * 100)
        print(f"【Q{idx:02d} svp_text：{MODE_DEFINITIONS['svp_text']['mode_desc']}】")
        print(answer)

    output_obj = {
        "schema_version": "e2e_svp_text_multisize_300q_v1",
        "config_json": CONFIG_JSON,
        "model_size": cfg_json.get("model_size", cfg_json.get("size", "unknown")),
        "workpiece_id": cfg_json.get("workpiece_id", str(cfg_json.get("sample_id", "unknown"))),
        "llm_base_path": cfg_json.get("llm_base_path", ""),
        "llm_stage2_adapter_path": normalize_adapter_path(str(cfg_json.get("llm_stage2_adapter_path", ""))),
        "weight_path": "",
        "valid_region_only": True,
        "mode_definitions": MODE_DEFINITIONS,
        "visual_summary": bundle["aggregate_summary"],
        "visual_fact_block": visual_fact_block,
        "cases": cases,
    }
    legacy.save_json(result_dir / "all_modes_results.json", output_obj)
    legacy.save_json(result_dir / "all_results.json", output_obj)
    legacy.save_json(result_dir / "graph_debug_all.json", graph_debug_all)

    with (result_dir / "e2e_compare_report.md").open("w", encoding="utf-8") as f:
        f.write("# SVP-Text 多尺寸 300 问测试结果\n\n")
        f.write(f"## 工件ID\n{cfg_json.get('workpiece_id', str(cfg_json.get('sample_id', 'unknown')))}\n\n")
        f.write(f"## 模型尺寸\n{cfg_json.get('model_size', cfg_json.get('size', 'unknown'))}\n\n")
        f.write(f"## LLM底座\n{cfg_json.get('llm_base_path', '')}\n\n")
        f.write(f"## Stage2 LoRA\n{normalize_adapter_path(str(cfg_json.get('llm_stage2_adapter_path', '')))}\n\n")
        f.write("## 生成方式\n")
        f.write("- SVP-Text：视觉层文字总结 + 压铸参数 + 问题。\n")
        f.write("- 不加载 multimodal_heads.pt，不输入视觉/区域/参数 prefix token。\n\n")
        f.write("## 过滤后视觉层摘要\n```json\n")
        f.write(json.dumps(bundle["aggregate_summary"], ensure_ascii=False, indent=2))
        f.write("\n```\n\n")
        for case in cases:
            f.write(f"## Q{case['question_index']:02d} {case['question_type']}\n")
            f.write(case["question"] + "\n\n")
            f.write("### 图谱命中摘要\n```json\n")
            f.write(json.dumps(legacy.summarize_graph_debug(case["graph_debug"]), ensure_ascii=False, indent=2))
            f.write("\n```\n\n")
            f.write("### svp_text\n\n")
            f.write(case["modes"]["svp_text"]["final_answer"] + "\n\n")

    print(f"\n[done] result output: {result_dir}", flush=True)
    print(f"[done] scoring input: {result_dir / 'all_modes_results.json'}", flush=True)


if __name__ == "__main__":
    main()

