import os
import sys
import copy
import re
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)
from config.config import MODEL_NAME,EMBED_MODEL, RERANKER_MODEL_NAME, CONTEXT_PATH, KGS_DIR, TOP_K
import ujson as json
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from FlagEmbedding import FlagReranker
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from util.kg_post_processor import NaivePostprocessor, KGRetrievePostProcessor, ngram_overlap, GraphFilterPostProcessor
from log.logger import setup_logger

logger = setup_logger("kg_rag_zmj")

# ============================================================
# 外部工具函数
# ============================================================

def read_context():
    if not os.path.exists(CONTEXT_PATH):
        logger.exception(f"[Context data] 文件不存在: {CONTEXT_PATH}")
        raise Exception(f"Contexts file {CONTEXT_PATH} does not exist.")
    try:
        with open(CONTEXT_PATH, 'r', encoding='utf-8') as f:
            contexts = json.load(f)
            logger.info(f"Read {len(contexts)} contexts from {CONTEXT_PATH}")
    except Exception as e:
        logger.exception(f"[Context data] 加载失败: {CONTEXT_PATH}\n错误: {e}")
        raise
    return contexts


def read_kg(contexts):
    ents = set()
    for context in contexts:
        ent = context[0]
        ents.add(ent)
    kg_dir = KGS_DIR
    doc2kg = dict()
    logger.info('[Loading KGs]')
    for ent in tqdm(ents):
        subkg_path = os.path.join(kg_dir, f'{ent.replace("/","_")}.json')
        if os.path.exists(subkg_path):
            with open(subkg_path, 'r', encoding='utf-8') as f:
                subkg = json.load(f)
                repkg = copy.deepcopy(subkg)
                if subkg and len(subkg.keys()) > 0:
                    for seq in subkg.keys():
                        if len(repkg[seq]) == 0:
                            del repkg[seq]
                    if len(repkg.keys()) > 0:
                        doc2kg[ent] = repkg
    logger.info(f'[Loaded kg] for {len(doc2kg.keys())} entities')
    return doc2kg


def load_kgs_vectors_index(ctxs, kgs):
    logger.info('Loading kgs、vectors and chunks_index')
    subkg = dict()
    doc_chunks = []
    chunks_index = dict()

    ents = [ctx[0] for ctx in ctxs]

    for ctx in ctxs:
        ent = ctx[0]
        chunks_index[ent] = {}
        for text_dict in ctx[1]:
            chunk_id = text_dict.get('chunk_id', '')
            text = f'{ent}: {text_dict.get("text", " ")}'

            if (ent in kgs) and (chunk_id in kgs[ent]) and (len(kgs[ent][chunk_id]) > 0):
                if ent not in subkg:
                    subkg[ent] = dict()
                target_kg = kgs[ent][chunk_id]
                for triplet in target_kg:
                    h, r, t = triplet
                    if ngram_overlap(h, ent) >= 0.90 or ngram_overlap(ent, h) >= 0.90:
                        h = ent
                    if ngram_overlap(t, ent) >= 0.90 or ngram_overlap(ent, t) >= 0.90:
                        t = ent
                subkg[ent][chunk_id] = target_kg

            doc_chunk = TextNode(text=text, id_=f'{ent}##{chunk_id}')
            doc_chunks.append(doc_chunk)
            chunks_index[ent][chunk_id] = text

    index = VectorStoreIndex(doc_chunks)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    logger.info('[成功加载]: kgs、vectors and chunks_index')
    return ents, retriever, subkg, chunks_index


def create_postprocessors(ents, subkg, chunks_index):
    """创建扩展器 + 过滤器 + 排序器（不依赖 LLM 生成）。"""
    expansion_pp = KGRetrievePostProcessor(ents=set(ents), doc2kg=subkg, chunks_index=chunks_index)
    bge_reranker = FlagReranker(model_name_or_path=RERANKER_MODEL_NAME, device=3)
    filter_pp = GraphFilterPostProcessor(
        use_tpt=False, topk=TOP_K, ents=set(ents),
        doc2kg=subkg, chunks_index=chunks_index, reranker=bge_reranker,
    )
    naive_pp = NaivePostprocessor()
    return expansion_pp, filter_pp, naive_pp


def init_model():
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    llm = Ollama(model=MODEL_NAME,request_timeout=600.0)
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm,embed_model


def extract_triplets(llm, ctx):
        query = f'''按照示例从文本中抽取信息三元组。必须仅使用文本中出现的原文内容，不推理、不添加、不编造。严格按照格式输出，不输出多余内容、不换行、无多余空格。
                --------------------
                Text: 高速时间与高速速度成反比，与高速行程成正比。在高速行程固定时，速度越快，时间越短。
                Triplets:<高速时间##与##高速速度>$$<高速时间##成反比##高速速度>$$<高速时间##与##高速行程>$$<高速时间##成正比##高速行程>$$
                --------------------
                Text: 温度过高会使牙孔区域过热，缩孔面积增大。温度过低则充填不良，气孔面积增大。
                Triplets:<温度过高##导致##牙孔区域过热>$$<温度过高##导致##缩孔面积增大>$$<温度过低##导致##充填不良>$$<温度过低##导致##气孔面积增大>$$
                --------------------
                Text: 湿度高时空气中水汽多，慢速速度过快会加剧卷气，将更多潮湿空气卷入铝液。
                Triplets:<湿度高##导致##空气中水汽多>$$<慢速速度过快##加剧##卷气>$$<慢速速度过快##卷入##铝液>$$
                --------------------
                Text: {ctx}
                Triplets:'''
        try:
            resp = llm.complete(query)
        
            resp = resp.text
            triplets = set()
            triplet_texts = resp.split('$$')
            for triplet_text in triplet_texts:
                if len(triplet_text) <= 6:
                    continue
                triplet_text = triplet_text[1:-1]
                tokens = triplet_text.split('##')
                if not len(tokens) == 3:
                    continue
                h = tokens[0].strip()
                r = tokens[1].strip()
                t = tokens[2].strip()
                if ('no ' in h) or ('no ' in t) or ('unknown' in h) or ('unknown' in t) or ('No ' in h) or ('No ' in t) or ('Unknown' in h) or ('Unknown' in t) or ('null' in h) or ('null' in t) or ('Null' in h) or ('Null' in t) or ('NULL' in h) or ('NULL' in t) or ('NO' in h) or ('NO' in r) or ('NO' in t) or (h==t):
                    continue
                if (r not in ctx) and (t not in ctx):
                    continue

                triplets.add((h, r, t))
            triplets = [[h,r,t] for (h,r,t) in triplets]
        except Exception as e:
            logger.error(f"推理错误：{e}\n跳过一条样本: {ctx}")
            return []
        return triplets


# ============================================================
# KGRAG 检索器
# ============================================================

class KGRAGQueryEngine():
    def __init__(self):
        self.llm,self.embed_model = init_model()
        self.contexts = read_context()
        self.kgs = read_kg(self.contexts)
        self.ents, self.vector_retriever, self.subkg, self.chunks_index = load_kgs_vectors_index(
            self.contexts, self.kgs,
        )
        self.expansion_pp, self.filter_pp, self.naive_pp = create_postprocessors(
            self.ents, self.subkg, self.chunks_index,
        )
        self._sorted_ent_names = sorted(self.ents, key=len, reverse=True)
        logger.info('[KGRAGQueryEngine] 初始化完成')
    
    def question_ent(self,llm,question):
        ents = set()
        triplets = extract_triplets(llm,question)
        for s,p,o in triplets:     
            if s not in ents:
                ents.add(s)
            if o not in ents:
                ents.add(o)
        return [ent for ent in ents]
    


    def retrieve(self, query: str,top_k: int = 10):
        seed_terms = self.question_ent(self.llm,query)
        # ── 2. 向量检索（不经过 query_engine，直接调 retriever）──
        try:
            self.vector_retriever.similarity_top_k = top_k   # 改 top_k
            nodes = self.vector_retriever.retrieve(query)  # 用新的 top_k 检索
            self.vector_retriever.similarity_top_k = TOP_K # 改回来

        except Exception as e:
            logger.error(f'[retrieve error]: {e}')
            return []

        if not nodes:
            return []

        # ── 3. 图谱扩展 → 过滤 → 排序 ──
        query_bundle = QueryBundle(query_str=query)
        nodes = self.expansion_pp._postprocess_nodes(nodes, query_bundle)
        nodes = self.filter_pp._postprocess_nodes(nodes, query_bundle)
        nodes = self.naive_pp._postprocess_nodes(nodes, query_bundle)

        # ── 4. 提取 source_nodes 信息 ──
    
        source_nodes = [
            {"node_id": str(node.node.id_), "source_text": node.node.text, "score": node.score}
            for node in nodes
        ]
        
        return seed_terms, source_nodes
        