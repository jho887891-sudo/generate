from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import torch
from peft import PeftModel


class LLMMODEL():
    
    def __init__(self,cfg_json):
        self.__build_llm_eval(cfg_json)
    def __build_llm_eval(self,cfg_json):
        print("build_llm_eval")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg_json['tokenizer_path'],trust_remote_code=True,local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bytes_config = BitsAndBytesConfig(
            load_in_4bit=True,#4bit量化linear等参数数量多的非骨干权重
            bnb_4bit_compute_dtype=torch.bfloat16,#反量化计算时的权重精度
            bnb_4bit_quant_type = "nf4",#4bit量化的算法
            bnb_4bit_use_double_quant = True#双重量化
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg_json['model_path'],
            quantization_config=bytes_config,
            trust_remote_code=True,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            #attn_implementation="flash_attention_2",
            use_cache=True,
        )
        self.llm = PeftModel.from_pretrained(self.model, cfg_json['lora_path'], is_trainable=False)
        self.llm.eval()
    def _tokenize(self,system_prompt,user_prompt,max_length=8192):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        #注入特殊字符
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=True,add_generation_prompt=True)
        print(f"prompt_text: {prompt_text}")
        #tokenize
        input_dict = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return input_dict
    def _generate(self,system_prompt,user_prompt,max_new_tokens):
        input_dict = self._tokenize(system_prompt,user_prompt)
        input_ids = input_dict["input_ids"].to("cuda:0")
        attention_mask = input_dict["attention_mask"].to("cuda:0")
        with torch.no_grad():
            output_ids = self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_ips = output_ids[:, input_ids.shape[1]:]
        output_text = self.tokenizer.batch_decode(new_ips, skip_special_tokens=True)[0]
        print(f"output_text: {output_text}")
        return output_text
    
