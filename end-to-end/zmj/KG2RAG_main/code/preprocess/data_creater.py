import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
CODE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)
    
from util.model_api import LLMMODEL,EmbedModel
from project_config.project_config import MODEL_PATH,TOKENIZER_PATH,LORA_PATH,MAX_NEW_TOKENS,EMBED_MODEL_PATH, CHUNKS_PATH,CONTEXT_PATH, KGS_DIR,VECTOR_PATH,CHUNKS_INDEX_PATH

import ujson as json
from tqdm import tqdm
import logging as _logging

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface.pooling import Pooling

class Creater:
    def __init__(self, logger=None):
        self._log = logger or _logging.getLogger(__name__)
        self._log.info("初始化图谱制作器 Creater ...")
        self.llm = LLMMODEL(cfg_json={"tokenizer_path": TOKENIZER_PATH, "model_path": MODEL_PATH, "lora_path": LORA_PATH})
        self.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_PATH,device="cuda:0",trust_remote_code=True)
        self.chunks_path = CHUNKS_PATH
        self.out_dir = KGS_DIR
        self.contexts_path = CONTEXT_PATH
        self.vector_path = VECTOR_PATH
        self.chunks_index_path = CHUNKS_INDEX_PATH

    def extract_triplets(self, llm, ctx):
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
        system_prompt = "你是一个信息抽取助手，按照用户的要求从文本中抽取三元组，必须严格按照用户要求的格式输出，不要添加任何多余内容。"
        try:
            resp = llm._generate(system_prompt=system_prompt, user_prompt=query, max_new_tokens=MAX_NEW_TOKENS)
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
            triplets = [[h, r, t] for (h, r, t) in triplets]
        except Exception as e:
            self._log.error(f"推理错误：{e}，跳过一条样本: {ctx}")
            return []
        return triplets

    def create_ctxs(self, chunks_texts):
        ents = set()
        ent_context = {}
        for ctx in tqdm(chunks_texts, desc="contexts制作"):
            if not ctx:
                self._log.warning("跳过一个空样本")
                continue
            text = ctx.get("text", " ")
            triplets = self.extract_triplets(self.llm, text)
            for s, p, o in triplets:
                if s not in ent_context:
                    ent_context[s] = []
                if o not in ent_context:
                    ent_context[o] = []
                ent_context[s].append(ctx)
                ent_context[o].append(ctx)
                if s not in ents:
                    ents.add(s)
                if o not in ents:
                    ents.add(o)
        self._log.info(f"三元组抽取完毕，共 {len(ents)} 个实体")
        contexts = [[ent, ent_context[ent]] for ent in ents]
        self._log.info(f"[contexts] 制作完毕，共 {len(contexts)} 条\n示例: {contexts[0]}...")
        return contexts

    def create_kgs(self):
        out_dir = self.out_dir
        count = 0
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        triplets = {}
        for ctx in tqdm(self.ctxs, desc="图谱制作"):
            ent = ctx[0]
            if ent in triplets:
                continue
            out_path = os.path.join(out_dir, f'{ent.replace("/", "_")}.json')
            if os.path.exists(out_path):
                continue
            triplets[ent] = {}
            for i in range(len(ctx[1])):
                ctx_text = f'{ent}: {ctx[1][i].get("text", " ")}'
                chunk_id = ctx[1][i].get("chunk_id", " ")
                ext_triplets = self.extract_triplets(self.llm, ctx_text)
                if len(ext_triplets) == 0:
                    continue
                triplets[ent][str(chunk_id)] = ext_triplets
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(triplets[ent], f, ensure_ascii=False, indent=2)
                    self._log.info(f"成功保存实体 {ent} 的三元组到 {out_path}")
            except Exception as e:
                self._log.exception(f"保存实体 {ent} 的三元组时发生错误: {e}")
                raise
            count += 1
        self._log.info(f"Newly extract entity KGs number: {count}")
        self._log.info("KGs 创建完成.")

    def create_vector(self):
        contexts = self.ctxs
        vector_path = self.vector_path
        embed_model = self.embed_model
        chunks_index_path = self.chunks_index_path
        if not os.path.exists(vector_path):
            os.makedirs(vector_path)
        chunks_index = {}
        nodes = []
        for context in contexts:
            ent = context[0]
            text_dicts = context[1]
            chunks_index[ent] = {}
            for i in range(len(text_dicts)):
                text = f'{ent}: {text_dicts[i].get("text", " ")}'
                chunk_id = text_dicts[i].get("chunk_id", " ")
                chunks_index[ent][str(chunk_id)] = text
                node = TextNode(text=text, id_=f'{ent}##{str(chunk_id)}')
                nodes.append(node)
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        index.storage_context.persist(persist_dir=vector_path)
        try:
            with open(chunks_index_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_index, f, ensure_ascii=False, indent=2)
                self._log.info(f"成功保存 chunks_index 到 {chunks_index_path}")
        except Exception as e:
            self._log.exception(f"保存 chunks_index 时发生错误: {e}")
            raise
        self._log.info("chunks_index 和 vector 创建完成")

    def __call__(self):
        chunks_path = self.chunks_path
        contexts_path = self.contexts_path
        if not os.path.exists(contexts_path):
            self._log.info(f"contexts 不存在于 {contexts_path}, 开始制作 contexts...")
            try:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                    self._log.info(f"[chunks] Read {len(chunks)} chunks from {chunks_path}")
            except Exception as e:
                self._log.exception(f"Error reading {chunks_path}: {e}")
                raise
            chunks_texts = [{"chunk_id": chunk["chunk_id"], "text": chunk["answer"]} for chunk in chunks]
            self.ctxs = self.create_ctxs(chunks_texts)
            try:
                with open(contexts_path, "w", encoding="utf-8") as f:
                    json.dump(self.ctxs, f, ensure_ascii=False, indent=2)
                    self._log.info(f"成功保存 contexts 到 {contexts_path}")
            except Exception as e:
                self._log.exception(f"保存 contexts 时发生错误: {e}")
                raise
        else:
            self._log.info(f"contexts 存在于 {contexts_path}, 开始加载 contexts...")
            try:
                with open(contexts_path, "r", encoding="utf-8") as f:
                    self.ctxs = json.load(f)
                    self._log.info(f"[contexts] 加载成功, len={len(self.ctxs)} 条\n示例: {self.ctxs[0]}...")
            except Exception as e:
                self._log.exception(f"加载 contexts 时发生错误: {e}")
                raise
        self.create_kgs()
        self.create_vector()
        self._log.info("All done.")


def datas_creater(logger=None):
    log = logger or _logging.getLogger(__name__)
    log.info("开始创建 KGs、vectors 和 chunks_index...")
    creater = Creater(logger=log)
    creater()


if __name__ == "__main__":
    from log.logger import setup_logger
    _main_logger = setup_logger("data_creater")
    datas_creater(logger=_main_logger)
#nohup python /home/T7/wkq/end-to-end/zmj/KG2RAG_main/code/preprocess/data_creater.py >> /home/T7/wkq/end-to-end/zmj/KG2RAG_main/log/stack_info/data_creater_use_4bit_llm.log 2>&1 &