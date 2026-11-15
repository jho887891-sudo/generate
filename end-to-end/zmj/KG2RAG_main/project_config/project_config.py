import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
#数据类路径
QA_INPUT_DIR = os.path.join(BASE_DIR, "data/data_vk3/raw_data/problem_answer")
QA_DATA_PATH = os.path.join(BASE_DIR, "data/data_vk3/qa_data/qa.json")
CHUNKS_PATH = os.path.join(BASE_DIR, "data/data_vk3/chunks/chunks.json")

CONTEXT_PATH = os.path.join(BASE_DIR, "data/data_vk3/context_add_model/context.json")
KGS_DIR = os.path.join(BASE_DIR, "data/data_vk3/kgs_add_model")
VECTOR_PATH = os.path.join(BASE_DIR, "data/data_vk3/vector_add_model")
CHUNKS_INDEX_PATH = os.path.join(BASE_DIR, "data/data_vk3/chunks_index_add_model/chunks_index.json")

TEST_DATA_PATH = os.path.join(BASE_DIR, "data/data_vk3/questions/questions.json")
SAMPLE_PATH = os.path.join(BASE_DIR, "data/data_vk3/sample/sample.json")
TOP_K = 10

#embed、reranker模型类路径
EMBED_MODEL_PATH = os.path.join(BASE_DIR, "model/Qwen3-Embedding-0.6B")
RERANKER_MODEL_PATH = os.path.join(BASE_DIR, "model/Qwen3-Reranker-0.6B")

#LLM配置路径
MODEL_PATH = os.path.join(BASE_DIR, "model/Qwen3-4B-stage1-book-merged-4bit")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model/Qwen3-4B-stage1-book-merged-4bit")
LORA_PATH = os.path.join(ROOT_DIR, "runs/qwen3_4b_region_mm_frozen_llm_from_merged_stage1/best/llm_adapter")
MAX_NEW_TOKENS = 8192
#nohup cp -av /home/T7/zmj/generate/end-to-end/runs/qwen3_4b_region_mm_frozen_llm_from_merged_stage1/best/llm_adapter /home/T7/zmj/generate/end-to-end/zmj/KG2RAG_main/model >> /home/T7/zmj/generate/end-to-end/zmj/KG2RAG_main/log/stack_info/cp_lora.log 2>&1 &