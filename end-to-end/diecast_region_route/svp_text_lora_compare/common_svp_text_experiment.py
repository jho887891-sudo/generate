from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from e2e_compare_three_modes_graph_4b_v3 import (  # noqa: E402
    GENERAL_SYSTEM_PROMPT,
    VisualStageRunner,
    build_info_text_for_prompt,
    build_question_type_hint,
    build_visual_fact_block,
    collect_image_paths,
    filter_visual_bundle,
    normalize_output,
    retrieve_graph_context,
    sanitize_graph_context_for_prompt,
)


DEFAULT_BASE_MODEL = "/home/T7/wkq/end-to-end/models/Qwen3-4B-Instruct-2507-bnb-4bit"
DEFAULT_CONFIG_JSON = "/home/T7/wkq/end-to-end/diecast_region_route/e2e_compare_graph_config_hardcoded.json"
LOCAL_CONFIG_JSON = str(PROJECT_ROOT / "e2e_compare_graph_config_hardcoded.json")
DEFAULT_OUTPUT_ROOT = "/home/T7/wkq/end-to-end/compare_outputs/svp_text_lora_compare"


SYSTEM_PROMPT = f"""{GENERAL_SYSTEM_PROMPT}

当前实验方法：SVP-Text。
图片先由视觉层模型识别为缺陷区域统计和视觉层文字总结，再把视觉层文字总结、压铸参数和问题写入提示词，由当前挂载 LoRA 的语言模型输出溯源结果。

实验约束：
- 不输入视觉/区域 prefix token。
- 不输入参数 prefix token。
- 视觉信息只来自视觉层文字总结。
- 必须把视觉统计转化为自然语言溯源分析，不要复制 JSON 或字典。
- 请直接输出【结论】【溯源分析】【建议方向】三部分。"""


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_config_path(path_text: str | None) -> Path:
    if path_text:
        return Path(path_text)
    server_path = Path(DEFAULT_CONFIG_JSON)
    if server_path.exists():
        return server_path
    return Path(LOCAL_CONFIG_JSON)


def normalize_adapter_path(path_text: str) -> str:
    path = Path(path_text)
    candidates = [
        path,
        path / "best_adapter",
        path / "adapter",
        path / "llm_adapter",
    ]
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return str(candidate)
    return str(path)


def get_questions(cfg_json: Dict[str, Any], limit: int | None = None) -> List[str]:
    questions = cfg_json.get("questions") or []
    if isinstance(questions, str):
        questions = [questions]
    if not questions and cfg_json.get("question"):
        questions = [cfg_json["question"]]
    questions = [str(q).strip() for q in questions if str(q).strip()]
    if limit and limit > 0:
        questions = questions[:limit]
    if not questions:
        raise ValueError("配置文件中没有找到 question/questions。")
    return questions


def build_svp_text_prompt(
    cfg_json: Dict[str, Any],
    question: str,
    visual_fact_block: str,
    graph_context: str = "",
    question_type_hint: str = "",
) -> str:
    info_text = build_info_text_for_prompt(cfg_json)
    graph_context = sanitize_graph_context_for_prompt(
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
{question}

【作答边界】
1. 当前输入形式为 SVP-Text：视觉信息只来自视觉层文字总结，压铸参数只通过文本提供。
2. 必须结合区域气孔数量、区域最大面积、区域最大等效直径和压铸参数链进行溯源。
3. 缺陷统一按气孔处理，不引入缩孔、缩松、裂纹、冷隔、夹杂等非气孔缺陷。
4. 图谱证据如存在，只能作为机理辅助，不能覆盖当前样本视觉统计和参数事实。
5. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。

【必须遵守的输出格式】
第一行必须是【结论】。
【溯源分析】下面必须写 1. 2. 3. 三个编号点。
【建议方向】下面必须写 1. 2. 3. 三个编号点。

现在开始作答。"""


class TextOnlyLoraModel:
    def __init__(self, base_model: str, adapter_path: str, load_in_4bit: bool = True) -> None:
        self.base_model = base_model
        self.adapter_path = normalize_adapter_path(adapter_path)
        self.load_in_4bit = load_in_4bit
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return

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
        max_new_tokens: int = 1280,
        max_seq_length: int = 2048,
        generation_max_time_sec: float | None = None,
    ) -> str:
        self.load()
        assert self.tokenizer is not None
        assert self.model is not None

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
        new_tokens = output_ids[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_visual_stage(cfg_json: Dict[str, Any], result_dir: Path) -> Dict[str, Any]:
    cfg_for_visual = dict(cfg_json)
    cfg_for_visual["visual_output_dir"] = str(result_dir / "visual")
    cfg_for_visual["save_intermediate_visual_files"] = True

    print("[svp-text][vision] collecting images", flush=True)
    image_paths = collect_image_paths(cfg_for_visual)
    print(f"[svp-text][vision] image count: {len(image_paths)}", flush=True)

    vision = VisualStageRunner(cfg_for_visual)
    try:
        bundle = vision.run(image_paths)
        bundle = filter_visual_bundle(bundle)
    finally:
        vision.release()

    save_json(result_dir / "visual" / "vision_bundle_filtered.json", bundle)
    return bundle


def run_experiment(
    experiment_name: str,
    adapter_path: str,
    argv: List[str] | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=f"SVP-Text LoRA 对比实验：{experiment_name}")
    parser.add_argument("--config", type=str, default=None, help="e2e_compare_graph_config_hardcoded.json 路径")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=str, default=adapter_path)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 个问题；默认 0 表示跑全部问题")
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--generation-max-time-sec", type=float, default=180)
    parser.add_argument("--use-graph", action="store_true", help="启用配置中的 GraphRAG 检索")
    parser.add_argument("--no-4bit", action="store_true", help="不用 4bit 方式加载基座模型")
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    cfg_json = load_json(config_path)
    cfg_json["use_graph_rag"] = bool(args.use_graph)

    questions = get_questions(cfg_json, limit=args.limit if args.limit > 0 else None)
    max_new_tokens = args.max_new_tokens or int(cfg_json.get("max_new_tokens", 1280))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(args.output_root) / f"{experiment_name}_{stamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    adapter_resolved = normalize_adapter_path(args.adapter)
    meta = {
        "experiment_name": experiment_name,
        "config_path": str(config_path),
        "base_model": args.base_model,
        "adapter_path": adapter_resolved,
        "question_count": len(questions),
        "use_graph_rag": bool(args.use_graph),
        "max_new_tokens": max_new_tokens,
        "max_seq_length": args.max_seq_length,
    }
    save_json(result_dir / "experiment_meta.json", meta)

    print("=" * 100, flush=True)
    print(f"[svp-text] experiment: {experiment_name}", flush=True)
    print(f"[svp-text] base model: {args.base_model}", flush=True)
    print(f"[svp-text] adapter: {adapter_resolved}", flush=True)
    print(f"[svp-text] questions: {len(questions)}", flush=True)
    print(f"[svp-text] output: {result_dir}", flush=True)

    bundle = run_visual_stage(cfg_json, result_dir)
    visual_fact_block = build_visual_fact_block(bundle["aggregate_summary"])
    save_text(result_dir / "visual_fact_block.txt", visual_fact_block)
    save_json(result_dir / "visual_summary.json", bundle["aggregate_summary"])

    llm = TextOnlyLoraModel(
        base_model=args.base_model,
        adapter_path=adapter_resolved,
        load_in_4bit=not args.no_4bit,
    )
    llm.load()

    cases = []
    answers_md = [
        f"# SVP-Text LoRA 对比实验：{experiment_name}",
        "",
        f"- base_model: `{args.base_model}`",
        f"- adapter: `{adapter_resolved}`",
        f"- question_count: {len(questions)}",
        "",
    ]

    for idx, question in enumerate(questions, start=1):
        t0 = time.time()
        print(f"[svp-text][{experiment_name}] Q{idx:03d}/{len(questions)} generating", flush=True)

        question_type, question_type_hint, focus_regions, mechanism_candidates = build_question_type_hint(question)
        graph_context = ""
        graph_debug: Dict[str, Any] = {
            "enabled": bool(args.use_graph),
            "available": False,
            "status": "disabled",
            "question_type": question_type,
            "focus_regions": focus_regions,
            "mechanism_candidates": mechanism_candidates,
        }
        if args.use_graph:
            try:
                graph_context, graph_debug_base = retrieve_graph_context(
                    cfg_json=cfg_json,
                    info_text=build_info_text_for_prompt(cfg_json),
                    question=question,
                    graph_top_k=int(cfg_json.get("graph_top_k", 10)),
                )
                graph_debug.update(graph_debug_base)
            except Exception as exc:
                graph_debug["status"] = f"graph retrieval failed: {exc}"

        user_prompt = build_svp_text_prompt(
            cfg_json=cfg_json,
            question=question,
            visual_fact_block=visual_fact_block,
            graph_context=graph_context,
            question_type_hint=question_type_hint,
        )
        raw_answer = llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            max_seq_length=args.max_seq_length,
            generation_max_time_sec=args.generation_max_time_sec,
        )
        answer = normalize_output(raw_answer)
        elapsed = time.time() - t0

        case = {
            "question_index": idx,
            "question": question,
            "question_type": question_type,
            "focus_regions": focus_regions,
            "mechanism_candidates": mechanism_candidates,
            "graph_context": graph_context,
            "graph_debug": graph_debug,
            "user_prompt": user_prompt,
            "raw_answer": raw_answer,
            "answer": answer,
            "elapsed_sec": round(elapsed, 3),
        }
        cases.append(case)

        q_dir = result_dir / f"q{idx:03d}"
        save_json(q_dir / "case_result.json", case)
        save_text(q_dir / "user_prompt.txt", user_prompt)
        save_text(q_dir / "raw_answer.txt", raw_answer)
        save_text(q_dir / "answer.txt", answer)

        answers_md.extend([
            f"## Q{idx:03d}",
            "",
            question,
            "",
            answer,
            "",
        ])
        save_json(result_dir / "partial_results.json", {"meta": meta, "cases": cases})
        save_text(result_dir / "answers.md", "\n".join(answers_md))
        print(f"[svp-text][{experiment_name}] Q{idx:03d} done in {elapsed:.1f}s", flush=True)

    output = {
        "schema_version": "svp_text_lora_compare_v1",
        "meta": meta,
        "visual_summary": bundle["aggregate_summary"],
        "visual_fact_block": visual_fact_block,
        "cases": cases,
    }
    save_json(result_dir / "all_results.json", output)
    save_text(result_dir / "answers.md", "\n".join(answers_md))
    print(f"[svp-text] finished: {result_dir / 'all_results.json'}", flush=True)

