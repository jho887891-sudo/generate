import os
import sys
import copy
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)
from config.config import KGS_DIR,CONTEXT_PATH,MODEL_NAME,EMBED_MODEL,RERANKER_MODEL_NAME,TOP_K
import ujson as json
from tqdm import tqdm
from FlagEmbedding import FlagReranker
from llama_index.core import Settings,VectorStoreIndex,PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import Settings,StorageContext,load_index_from_storage
from util.kg_post_processor import NaivePostprocessor,KGRetrievePostProcessor,ngram_overlap, GraphFilterPostProcessor
from util.kg_response_synthesizer import get_response_synthesizer
from log.logger import setup_logger

logger = setup_logger("kg_rag_zmj")

def read_context():
    if not os.path.exists(CONTEXT_PATH):
        logger.exception(f"[Context data] 文件不存在: {CONTEXT_PATH}")
        raise Exception(f"Contexts file {CONTEXT_PATH} does not exist.")
    try:
        with open(CONTEXT_PATH,'r',encoding='utf-8') as f:
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

def load_kgs_vectors_index(ctxs,kgs):
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
            
    #创建索引
    index = VectorStoreIndex(doc_chunks)
    #为向量库创建检索器
    retriever = VectorIndexRetriever(index=index,similarity_top_k=TOP_K)
    logger.info('[成功加载]: kgs、vectors and chunks_index')
    return ents,retriever,subkg,chunks_index


def create_query_engine(ents,retriever,subkg,chunks_index):
    qa_rag_template_str = (
        '以下内容来自知识图谱检索，仅作为辅助证据与约束先验；若与当前样本统计冲突，以样本输入为准。\n'
        '{context_str}\n'
        '你是一名铝合金压铸气孔缺陷溯源助手。\n'
        '硬性规则：\n'
        '1. 最终回答必须直接从【结论】开始，必须且只能包含【结论】【溯源分析】【建议方向】三部分。\n'
        '2. 缺陷类型统一按气孔处理；牙孔区域、焊接区域、重点区域属于高影响区域；一般区域作为整体波动参考。\n'
        '3. 图谱证据只辅助机理判断，不能覆盖当前样本；不要输出图谱命中列表、规则编号或调试信息。\n'
        '4. 建议方向只给宏观排查方向、联动关系和验证方法，不给具体参数设定值。\n'
        '5. 不要输出思考过程，不要复述整段输入。\n'
        '6. 【结论】必须是诊断判断句，必须包含明确优先级判断，不能使用提问式表达。\n'
        '7. 【溯源分析】三个编号点分别围绕"区域证据→参数链→综合优先级"展开。\n'
        '8. 【建议方向】三个编号点分别对应"优先排查→联动复核→对照验证"。\n'
        'Q: {query_str}\n'
        'A: '
    )
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    #创建响应器
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT,text_qa_template=qa_rag_prompt_template)
    #创建根据知识图谱对实体进行扩展的处理器
    expansion_pp = KGRetrievePostProcessor(ents=ents, doc2kg=subkg, chunks_index=chunks_index)
    bge_reranker = FlagReranker(model_name_or_path=RERANKER_MODEL_NAME, device=3)
    filter_pp = GraphFilterPostProcessor(use_tpt=False, topk=TOP_K, ents=ents, doc2kg=subkg, chunks_index=chunks_index, reranker=bge_reranker)
    naive_pp = NaivePostprocessor()
    #创建最终的查询引擎
    query_engine = RetrieverQueryEngine(retriever=retriever,response_synthesizer=response_synthesizer,node_postprocessors=[expansion_pp,filter_pp,naive_pp])
    return query_engine

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


def question_ent(llm,question):
    ents = set()
    triplets = extract_triplets(llm,question)
    for s,p,o in triplets:     
        if s not in ents:
            ents.add(s)
        if o not in ents:
            ents.add(o)
    return [ent for ent in ents]

def init_model():
    embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    llm = Ollama(model=MODEL_NAME,request_timeout=600.0)
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm,embed_model

class KGRAGQueryEngine():
    def __init__(self):
        self.llm, embed_model = init_model()  # 必须最先调用（Settings.embed_model 被后续依赖）
        self.contexts = read_context()
        self.kgs = read_kg(self.contexts)
        self.ents, self.retriever, self.subkg, self.chunks_index = load_kgs_vectors_index(self.contexts, self.kgs)
        self.query_engine = create_query_engine(ents=self.ents, retriever=self.retriever, subkg=self.subkg, chunks_index=self.chunks_index)
        
    
    def question_ent(self,llm,question):
        ents = set()
        triplets = extract_triplets(llm,question)
        for s,p,o in triplets:     
            if s not in ents:
                ents.add(s)
            if o not in ents:
                ents.add(o)
        return [ent for ent in ents]


    #需要你根据推理脚本的方法来修改这个方法build_lucene_query
    def build_lucene_query(self,seed_terms):
        lucene_query = " OR ".join([f"({term})" for term in seed_terms])
        return lucene_query
    
    def _format_graph_context(self, seed_terms, source_nodes, max_chars=2000):
        """将检索结果格式化为结构化文本块（对应原始 format_graph_context）。"""
        if not source_nodes:
            return ""
        lines = [
            "【KG2RAG 检索证据】",
            "以下内容来自知识图谱检索（KG2RAG），仅作为辅助证据与约束先验；"
            "若与当前样本统计冲突，必须以当前样本输入为准。",
        ]
        if seed_terms:
            lines.append(f"一、命中检索词：{'；'.join(seed_terms[:10])}")
        if source_nodes:
            lines.append("二、命中实体：")
            seen_ents = set()
            for sn in source_nodes:
                ent = sn['node_id'].split('##')[0]
                if ent not in seen_ents:
                    lines.append(f"- {ent}")
                    seen_ents.add(ent)
            lines.append("三、检索文本：")
            seen_texts = set()
            for sn in source_nodes:
                t = sn['source_text'].strip()
                if t and t not in seen_texts:
                    lines.append(f"- {t}")
                    seen_texts.add(t)
        text = "\n".join(lines)
        return text[:max_chars]

    def retriever(self,query:str):
        graph_context = ""
        graph_debug = {
            "enabled": True,
            "available": False,
            "status": "disabled",
            "seed_terms": [],#问题检索关键词
            "lucene_query": "",#问题检索关键词拼接
            "seed_entities": [],#检索到的实体
            "defect_names": [],
            "rule_ids": [],#知识来源的引用编号
            "rule_hits": 0,#多少条知识来源
            "evidence_hits": 0,#扩展了多少个实体
            "question_type": None,
            "focus_regions": None,
            "mechanism_candidates": None,
        }
        seed_terms = self.question_ent(self.llm,query)
        graph_debug['seed_terms'] = seed_terms

        try:
            response = self.query_engine.query(query)
            prediction = response.response
            source_nodes = [{"node_id": str(source_node.node.id_), "source_text": source_node.node.text,"score": source_node.score} for source_node in response.source_nodes]
        except Exception as e:
            logger.error(f'Error: {e}')
            prediction = ''
            source_nodes = []
        graph_debug['lucene_query'] = self.build_lucene_query(seed_terms)
        for source_node in source_nodes:
            graph_debug['rule_ids'].append(source_node['node_id'])
            graph_debug['rule_hits'] += 1
            graph_debug['evidence_hits'] += 1
            graph_debug['seed_entities'].append(source_node['node_id'].split('##')[0])

        # ── 组装 graph_context（对应原始 format_graph_context）──
        graph_context = self._format_graph_context(seed_terms, source_nodes)

        # ── 填充 graph_debug 剩余字段 ──
        graph_debug['available'] = len(source_nodes) > 0
        graph_debug['status'] = "connected" if source_nodes else "refused connection"

        return graph_context, graph_debug