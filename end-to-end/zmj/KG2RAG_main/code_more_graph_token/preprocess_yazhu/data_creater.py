import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
import ujson as json
from tqdm import tqdm
from config.config import EMBED_MODEL,MODEL_NAME,CHUNKS_PATH,CONTEXT_PATH, KGS_DIR,VECTOR_PATH,CHUNKS_INDEX_PATH
from log.logger import setup_logger

from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding

logger = setup_logger("data_creater")



#图谱制作器
class Creater():
    def __init__(self):
        self.llm = Ollama(
            model=MODEL_NAME,
            request_timeout=600.0,  # 增加超时时间到300秒（5分钟）
        )
        self.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
        self.chunks_path = CHUNKS_PATH
        self.out_dir = KGS_DIR
        self.contexts_path = CONTEXT_PATH
        self.vector_path = VECTOR_PATH
        self.chunks_index_path = CHUNKS_INDEX_PATH
    def extract_triplets(self,llm, ctx):
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
            logger.error(f"推理错误：{e}，跳过一条样本: {ctx}")
            return []
        return triplets
    
    #创建编号id为chunk_id的contexts
    def create_ctxs(self,chunks_texts):
        ents = set()
        ent_context={}
        for ctx in tqdm(chunks_texts,desc="contexts制作"):
            if not ctx:
                logger.warning("跳过一个空样本")
                continue
            text = ctx.get("text"," ")
            triplets = self.extract_triplets(self.llm,text)
            for s,p,o in triplets:
                if s not in ent_context.keys():
                    ent_context[s]=list()
                if o not in ent_context.keys():
                    ent_context[o]=list()  
                ent_context[s].append(ctx)
                ent_context[o].append(ctx)      
                if s not in ents:
                    ents.add(s)
                if o not in ents:
                    ents.add(o)
        logger.info(f"三元组抽取完毕，共 {len(ents)} 个实体")
        contexts = [[ent,ent_context[ent]] for ent in ents]
        logger.info(f"[contexts] 制作完毕，共 {len(contexts)} 条\n示例: {contexts[0]}...")
        return contexts
    
    def __call__(self):
        chunks_path = self.chunks_path
        contexts_path = self.contexts_path
        if not os.path.exists(contexts_path):
            logger.info(f"contexts 不存在于 {contexts_path}, 开始制作 contexts...")
            try:
                with open(chunks_path,"r",encoding="utf-8") as f:
                    chunks = json.load(f)
                    logger.info(f"[chunks] Read {len(chunks)} chunks from {chunks_path}")
            except Exception as e:
                logger.exception(f"Error reading {chunks_path}: {e}")
                raise
            chunks_texts = [{"chunk_id":chunk["chunk_id"],"text":chunk["answer"]} for chunk in chunks]
            self.ctxs = self.create_ctxs(chunks_texts)
            try:
                with open(contexts_path,"w",encoding="utf-8") as f:
                    json.dump(self.ctxs,f,ensure_ascii=False,indent=2)
                    logger.info(f"成功保存 contexts 到 {contexts_path}")
                    
            except Exception as e:
                logger.exception(f"保存 contexts 时发生错误: {e}")
                raise
        else:
            logger.info(f"contexts 存在于 {contexts_path}, 开始加载 contexts...")
            try:
                with open(contexts_path,"r",encoding="utf-8") as f:
                    self.ctxs = json.load(f)
                    logger.info(f"[contexts] 加载成功, len={len(self.ctxs)} 条\n示例: {self.ctxs[0]}...")
            except Exception as e:
                logger.exception(f"加载 contexts 时发生错误: {e}")
                raise
        #创建图谱
        self.create_kgs()
        #创建向量索引
        self.create_vector()
        logger.info("All done.")
    #图谱制作器入口
    def create_kgs(self):
        out_dir = self.out_dir
        count = 0
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        triplets = {}
        for ctx in tqdm(self.ctxs,desc="图谱制作"):
            ent = ctx[0]
            if ent in triplets:
                continue
            out_path = os.path.join(out_dir,f'{ent.replace("/","_")}.json')
            if os.path.exists(out_path):
                continue
            triplets[ent] = {}
            for i in range(len(ctx[1])):
                # 每一个文本和实体关联
                ctx_text = f'{ent}: {ctx[1][i].get("text"," ")}'
                chunk_id = ctx[1][i].get("chunk_id"," ")
                
                ext_triplets = self.extract_triplets(self.llm,ctx_text)
                if len(ext_triplets)==0:
                    continue
                triplets[ent][str(chunk_id)] = ext_triplets
            try:
                with open(out_path,'w',encoding='utf-8') as f:
                    json.dump(triplets[ent],f,ensure_ascii=False,indent=2)
                    logger.info(f"成功保存实体 {ent} 的三元组到 {out_path}")
            except Exception as e:
                logger.exception(f"保存实体 {ent} 的三元组时发生错误: {e}")
                raise
            count += 1
        logger.info(f"Newly extract entity KGs number: {count}")
        logger.info("KGs 创建完成.")
    

    # 创建vector和chunks_index,节点id和chunks_index索引均是使用chunk_id制作的
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
                
                text = f'{ent}: {text_dicts[i].get("text"," ")}'
                chunk_id = text_dicts[i].get("chunk_id"," ")
                chunks_index[ent][str(chunk_id)] = text
                node = TextNode(text=text,id_=f'{ent}##{str(chunk_id)}')
                nodes.append(node)
        #创建向量库
        index = VectorStoreIndex(nodes,embed_model=embed_model)
        index.storage_context.persist(persist_dir=vector_path)
        #保存chunks_index
        try:
            with open(chunks_index_path, 'w', encoding='utf-8') as f:
                json.dump(chunks_index, f, ensure_ascii=False, indent=2)
                logger.info(f"成功保存 chunks_index 到 {chunks_index_path}")
        except Exception as e:
            logger.exception(f"保存 chunks_index 时发生错误: {e}")
            raise
        logger.info("chunks_index 和 vector 创建完成")


#对外统一入口
def datas_creater():
    logger.info("开始创建 KGs、vectors 和 chunks_index...")
    creater = Creater()
    creater()
            
        
if __name__ == "__main__":

    datas_creater()
    
