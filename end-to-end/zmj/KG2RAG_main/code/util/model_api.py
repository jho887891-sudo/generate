from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig,AutoTokenizer, AutoModel
import torch
from peft import PeftModel
import torch.nn.functional as F
from torch import Tensor
from typing import List,Dict,Any
from llama_index.core.base.embeddings.base import BaseEmbedding


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
    def _generate(self,system_prompt,user_prompt,max_new_tokens=8192):
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
    


# from transformers import AutoTokenizer, AutoModel

# class Embed_model:
#     def __init__(self, model_path: str, device: str = "cuda"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#         self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
#         self.device = device

#     def encode(self, texts, batch_size=16):
#         #单条语句查询，构建batch维度
#         if isinstance(texts, str):
#             texts = [texts]
#         all_embeddings = []
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]
#             inputs = self.tokenizer(
#                 batch, return_tensors="pt", padding=True,
#                 truncation=True, max_length=8192,
#             ).to(self.device)

#             with torch.no_grad():
#                 hidden = self.model(**inputs).last_hidden_state    # (batch, seq, dim)

#             # 取最后 token 的 hidden state（Qwen Embedding 用 last token pooling）
#             # 其他模型可能需要 mean pooling 或 CLS token，取决于训练时的设计
#             attention_mask = inputs["attention_mask"]
#             # mean pooling:
#             # (b,n) -> (b,n,1) -> (b,n,dim)
#             mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
#             #hidden(b,n,dim)-> (b,dim),每一个batch的token都去乘以该token的mask指，用于筛去原文中pad等无关token
#             sum_hidden = (hidden * mask_expanded).sum(dim=1)
#             #mask_expanded计算每一个batch的有效token数，并转化为sum_hidden的shape
            
#             sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
#             #mean pooling
#             embedding = sum_hidden / sum_mask

#             # L2 归一化
#             #按照每一个token的向量维度进行归一化，映射到[0,1]区间，加快计算效率
#             #！！！需要用什么维度进行操作，dim就是什么。比如需要在特征值的维度进行操作，dim就是特征值的维度-1
            
#             embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
#             all_embeddings.append(embedding.cpu().numpy())

#         return np.concatenate(all_embeddings, axis=0)




class EmbedModel(BaseEmbedding):
    def __init__(self, cfg_json, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(cfg_json["embed_model_tokenizer_path"], padding_side='left')
        if self._tokenizer.pad_token is None:              
            self._tokenizer.pad_token = self._tokenizer.eos_token 
        print(f"[embed_model_tokenizer.pad_token]:{self._tokenizer.pad_token}")
        self._model = AutoModel.from_pretrained(cfg_json["embed_model_path"], trust_remote_code=True).to("cuda:0")

    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
            ]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def _tokenize(self, texts, max_length=8192):
        if isinstance(texts, str):
            input_texts = [texts]
        elif isinstance(texts, list):
            input_texts = texts
        batch_dict = self._tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return batch_dict.to(self._model.device)

    def _get_text_embeddings(self, texts):
        batch_dict = self._tokenize(texts)
        outputs = self._model(**batch_dict)
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

    def _get_text_embedding(self, text: str):
        return self._get_text_embeddings([text])[0]

    def _get_query_embedding(self, query: str):
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)
    