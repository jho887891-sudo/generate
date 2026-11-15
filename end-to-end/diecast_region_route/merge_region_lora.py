from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="把当前 adapter merge 回底座，导出 standalone LLM。")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    quant_config = BitsAndBytesConfig(load_in_4bit=True) if args.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, local_files_only=True)
    tok.save_pretrained(out)
    print(f"merged model saved to {out}")


if __name__ == "__main__":
    main()
