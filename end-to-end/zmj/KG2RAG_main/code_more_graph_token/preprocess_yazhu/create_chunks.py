import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

import json
import re
from tqdm import tqdm
from pathlib import Path
from config.config import EMBED_MODEL, QA_INPUT_DIR,QA_DATA_PATH,CHUNKS_PATH
from log.logger import setup_logger

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding

logger = setup_logger("create_chunks")

# ============================================================
# 统一输出格式：{"_id": str, "question": str, "answer": str}
# ============================================================

def filter_qa():
    """
    从 docx 目录读取 Q&A 文档，正则抽取问题-答案对，
    统一输出为 {"_id": counter, "question": ..., "answer": ...} 的 JSON 列表。
    """
    documents = SimpleDirectoryReader(input_dir=QA_INPUT_DIR).load_data()
    pattern = re.compile(
        r'(?:^|\n)\s*'
        r'(\d+)'
        r'\s*\.\s*'
        r'(.+?\？)'
        r'\s*\n\s*'
        r'(.+?)'
        r'(?=\s*\n\s*\d+\s*\.\s*|\s*\Z)',
        re.DOTALL
    )

    filter_qa_data = []
    for doc in tqdm(documents, desc="正则抽取 QA 问答对"):
        doc_id = Path(doc.metadata["file_name"]).stem
        pairs = pattern.findall(doc.text)
        for q_id, question, answer in pairs:
            filter_qa_data.append({
                "chunk_id": f"{doc_id}_{q_id.strip()}",
                "question": question.strip(),
                "answer": answer.strip().replace("\n", " ")
            })

    os.makedirs(os.path.dirname(QA_DATA_PATH), exist_ok=True)
    try:
        with open(QA_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(filter_qa_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[filter_qa] 抽取完成，共 {len(filter_qa_data)} 条，已保存至 {QA_DATA_PATH}")
    except Exception as e:
        logger.exception(f"[filter_qa] 抽取失败: {e}")
        raise
    return filter_qa_data


def spliter(filter_data):
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=1,
        breakpoint_percentile_threshold=95
    )

    spliter_data = []
    for filter_chunk in tqdm(filter_data, desc="语义切块"):
        try:
            doc = Document(
                text=filter_chunk["answer"],
                metadata={
                    "question": filter_chunk["question"],
                    "chunk_id": filter_chunk["chunk_id"]
                }
            )
            chunks = splitter.get_nodes_from_documents([doc])
            for chunk_idx, chunk in enumerate(chunks):
                spliter_data.append({
                    "chunk_id": f"{chunk['chunk_id']}_{chunk_idx}",
                    "question": chunk["question"],
                    "answer": chunk.text
                })
        except Exception as e:
            logger.exception(f"遇到切块失败的chunk: {filter_chunk}\n问题是{e}\n跳过该chunk")
            continue

    return spliter_data

#清洗出chunks的对外接口
def chunks_creater():
    qa_data=filter_qa()
    try:
        print(f"[spliter] 切块开始")
        spliter_data = spliter(qa_data)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(spliter_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[spliter] 切块完成，共 {len(spliter_data)} 条，已保存至 {CHUNKS_PATH}")
    except Exception as e:
        logger.exception(f"[spliter] 切块失败: {e}")
        raise
    

if __name__ == "__main__":
    # 示例用法
    chunks_creater()
#nohup python   /home/T7/wkq/end-to-end/zmj/KG2RAG-main/code/kg_rag_main.py > /dev/null 2>&1 &