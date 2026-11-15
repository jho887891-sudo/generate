import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)

import json
import re
import logging as _logging
from tqdm import tqdm
from pathlib import Path
from config.config import EMBED_MODEL, QA_INPUT_DIR, QA_DATA_PATH, CHUNKS_PATH

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding


def filter_qa(logger=None):
    """从 docx 目录读取 Q&A 文档，正则抽取问题-答案对。"""
    log = logger or _logging.getLogger(__name__)
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
        log.info(f"[filter_qa] 抽取完成，共 {len(filter_qa_data)} 条，已保存至 {QA_DATA_PATH}")
    except Exception as e:
        log.exception(f"[filter_qa] 抽取失败: {e}")
        raise
    return filter_qa_data


def spliter(filter_data, logger=None):
    log = logger or _logging.getLogger(__name__)
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
            log.exception(f"遇到切块失败的chunk: {filter_chunk}\n问题是{e}\n跳过该chunk")
            continue

    return spliter_data


def chunks_creater(logger=None):
    log = logger or _logging.getLogger(__name__)
    qa_data = filter_qa(logger=log)
    try:
        print("[spliter] 切块开始")
        spliter_data = spliter(qa_data, logger=log)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(spliter_data, f, ensure_ascii=False, indent=2)
        log.info(f"[spliter] 切块完成，共 {len(spliter_data)} 条，已保存至 {CHUNKS_PATH}")
    except Exception as e:
        log.exception(f"[spliter] 切块失败: {e}")
        raise


if __name__ == "__main__":
    from log.logger import setup_logger
    _main_logger = setup_logger("create_chunks_use_4bit_llm")
    chunks_creater(logger=_main_logger)
