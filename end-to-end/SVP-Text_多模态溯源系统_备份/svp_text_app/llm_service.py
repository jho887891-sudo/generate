from __future__ import annotations

import gc
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMPaths:
    base_path: str
    adapter_path: str = ""


class SVPTextLLM:
    def __init__(self, paths: LLMPaths, load_in_4bit: bool = True, generation_max_time_sec: float = 180) -> None:
        self.paths = paths
        self.load_in_4bit = load_in_4bit
        self.generation_max_time_sec = generation_max_time_sec
        self.tokenizer = None
        self.model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.paths.base_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.truncation_side = "right"
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
                self.paths.base_path,
                quantization_config=quant_config,
                trust_remote_code=True,
                local_files_only=True,
                device_map={"": 0} if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            )
            if self.paths.adapter_path:
                self.model = PeftModel.from_pretrained(self.model, self.paths.adapter_path, is_trainable=False)
            self.model.eval()

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.tokenizer = None
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    def generate_with_stats(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1280,
        max_seq_length: int = 4096,
    ) -> Dict[str, Any]:
        import torch

        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        with self._lock:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False,
            )
            prompt_token_count = int(enc.input_ids.shape[-1])
            safe_max_seq_length = max(256, int(max_seq_length or 4096))
            was_truncated = prompt_token_count > safe_max_seq_length
            input_ids_cpu = enc.input_ids
            attention_mask_cpu = enc.attention_mask
            if was_truncated:
                head_len = int(safe_max_seq_length * 0.72)
                tail_len = safe_max_seq_length - head_len
                input_ids_cpu = torch.cat(
                    [input_ids_cpu[:, :head_len], input_ids_cpu[:, -tail_len:]],
                    dim=1,
                )
                attention_mask_cpu = torch.ones_like(input_ids_cpu)

            device = self.model.device
            input_ids = input_ids_cpu.to(device)
            attention_mask = attention_mask_cpu.to(device)
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
            if self.generation_max_time_sec and self.generation_max_time_sec > 0:
                kwargs["max_time"] = float(self.generation_max_time_sec)

            with torch.no_grad():
                output_ids = self.model.generate(**kwargs)
            new_tokens = output_ids[0][input_ids.shape[-1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return {
                "text": text,
                "prompt_char_count": len(prompt_text),
                "prompt_token_count": prompt_token_count,
                "max_seq_length": safe_max_seq_length,
                "was_truncated": was_truncated,
            }

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1280,
        max_seq_length: int = 4096,
    ) -> str:
        return self.generate_with_stats(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            max_seq_length=max_seq_length,
        )["text"]


_GLOBAL_LLM: Optional[SVPTextLLM] = None


def get_llm(paths: LLMPaths, generation_max_time_sec: float) -> SVPTextLLM:
    global _GLOBAL_LLM
    if _GLOBAL_LLM is None or _GLOBAL_LLM.paths != paths:
        if _GLOBAL_LLM is not None:
            _GLOBAL_LLM.unload()
        _GLOBAL_LLM = SVPTextLLM(paths=paths, generation_max_time_sec=generation_max_time_sec)
    return _GLOBAL_LLM
