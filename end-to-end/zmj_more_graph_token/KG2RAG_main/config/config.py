import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

#模型类路径
EMBED_MODEL = 'mixedbread-ai-mxbai-embed-large-v1' 
MODEL_NAME = 'Orenguteng-llama3-8b'
RERANKER_MODEL_NAME = os.path.join(BASE_DIR, "model/bge-reranker-large-safetensors")

#LLM配置路径
MODEL_PATH = "/home/T7/wkq/end-to-end/models/Qwen3-4B-stage1-book-merged-4bit"
TOKENIZER_PATH = "/home/T7/wkq/end-to-end/models/Qwen3-4B-stage1-book-merged-4bit"
LORA_PATH = "/home/T7/wkq/end-to-end/runs/qwen3_4b_region_mm_frozen_llm_from_merged_stage1/best/llm_adapter"

