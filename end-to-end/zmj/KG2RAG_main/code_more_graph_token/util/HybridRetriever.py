from typing import List,Dict,Any
import json
import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
    
from config.config import CHUNKS_PATH

from rank_bm25 import BM25Okapi
from llama_index.core import Settings,StorageContext,load_index_from_storage
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore,TextNode,QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever


def tokenize(text:str)->List[str]:
    return text.lower().split()
    #import jieba
    #return list(jieba.cut(text.lower()))

class BM25Retriever(BaseRetriever):
    def __init__(
        self,
        top_k: int = 5,
    ):
        super().__init__()
        self.top_k = top_k
        try:
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            print(f"[chunks]:Read {len(chunks)} chunks from {CHUNKS_PATH}")
        except Exception as e:
            raise Exception(f"Error occurred while reading chunks file {CHUNKS_PATH}: {e}")
        
        
        #创建BM25检索节点和检索文本
        bm25_corpus = []
        bm25_nodes = []
        for chunk in chunks:
            text = chunk["answer"].replace("\n", " ")
            bm25_corpus.append(text)
            node = TextNode(text=text, id_=chunk["chunk_id"])
            bm25_nodes.append(node)
        self.bm25_corpus = bm25_corpus
        self.bm25_nodes = bm25_nodes
        # 创建BM25模型
        self.bm25 = BM25Okapi(self.bm25_corpus)
        
    def _retrieve(self,query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str
        print(f"[BM25检索开始] 查询: {query_str}")
        tokenized_query = tokenize(query_str)
        bm25_scores = self.bm25_retriever.get_scores(tokenized_query)
        sorted_bm25 = sorted(enumerate(bm25_scores),key=lambda  x:x[1],reverse=True)[:self.top_k]
        
        bm25_results = []
        for idx,score in sorted_bm25:
            if score <= 0:
                continue
            node = self.bm25_nodes[idx]
            bm25_results.append(NodeWithScore(node=node,score=score))
            
        print(f"BM25检索结果数: {len(bm25_results)}")    
        return bm25_results[:self.top_k]

class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        vector_retriever:VectorIndexRetriever,
        top_k: int = 10,
        vector_weight: float = 0.5,
        rrf_k: int = 60
    ):
        super().__init__()
        #初始化向量检索器
        self.vector_retriever = vector_retriever
        #初始化BM25检索器
        self.bm25_retriever = BM25Retriever(top_k=top_k)

        #检索器配置
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k
        self.bm25_weight = 1.0 - vector_weight
        
        # 初始化日志
        print(f"[混合检索器初始化完成]")
        print(f"最终返回结果数：{top_k}，向量权重：{vector_weight}")
        
    def _retrieve(self,query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str
        print(f"\n[检索开始] 查询: {query_str}")
        # 1. 向量检索
        vector_results = self.vector_retriever.retrieve(query_str)
        print(f"向量检索结果数: {len(vector_results)}")
        
        # 2. BM25检索
        bm25_results = self.bm25_retriever.retrieve(query_bundle)
        
        
        #3.边界情况检查：单一检索无结果直接返回另一边
        if not vector_results:
            print("[边界情况] 向量检索无结果，直接返回BM25结果")
            return bm25_results[:self.top_k]
        if not bm25_results:
            print("[边界情况] BM25检索无结果，直接返回向量检索结果")
            return vector_results[:self.top_k]
        
        # 3. 结果融合（RRF）
        rrf_scores_map: Dict[str, float] = {}
        node_data_map: Dict[str, NodeWithScore] = {}
        
        for rank_idx,res in enumerate(vector_results[:self.top_k]):
            node_id = res.node.node_id
            rank = rank_idx + 1
            rrf_score = 1.0/(self.rrf_k + rank)
            rrf_scores_map[node_id] = rrf_score * self.vector_weight
            node_data_map[node_id] = res
        
        for rank_idx,res in enumerate(bm25_results[:self.top_k]):
            node_id = res.node.node_id
            rank = rank_idx + 1
            rrf_score = 1.0/(self.rrf_k + rank)
            
            if node_id not in node_data_map:
                rrf_scores_map[node_id] = rrf_score * self.bm25_weight
                node_data_map[node_id] = res
            
            else:
                if res.score > node_data_map[node_id].score:
                    node_data_map[node_id] = res    
                rrf_scores_map[node_id] += rrf_score * self.bm25_weight
            
        # 按照RRF分数排序并返回前K个结果
        sorted_nodes = sorted(rrf_scores_map.items(),key=lambda x:x[1],reverse=True)
        final_results = []
        for node_id,final_score in sorted_nodes[:self.top_k]:
            result_node = node_data_map[node_id]
            result_node.score = final_score
            final_results.append(result_node)
        
        print(f"融合后结果数: {len(final_results)}")
        return final_results
    
def init_model(model_name:str,embed_model_name:str):
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama
    Settings.embed_model = OllamaEmbedding(model_name=embed_model_name)
    Settings.llm = Ollama(model=model_name,request_timeout=300.0)
    
def create_hybrid_retriever(
    vector_path : str = "vector_index",
    chunks_path : str = "chunks_index.json",
    top_k : int = 10,
    vector_weight : float = 0.5,
    rrf_k : int = 60,
) -> HybridRetriever:
    
    #加载向量检索器
    storage_context = StorageContext.from_defaults(persist_dir=vector_path)
    vector_index = load_index_from_storage(storage_context)
    vector_retriever = VectorIndexRetriever(index=vector_index,similarity_top_k=top_k)
    print(f"向量检索器加载成功：{vector_path}") 
    
    #读取chunks数据
    if not os.path.exists(chunks_path):
        raise Exception(f"Chunks index file {chunks_path} does not exist.")
    try:
        with open(chunks_path,'r',encoding='utf-8') as f:
            chunks = json.load(f)
            print(f"Read {len(chunks)} chunks from {chunks_path}")
            print(f"Example chunk: {chunks[0]}...")  # 打印第一个chunk作为示例
    except Exception as e:
        raise Exception(f"Error occurred while reading chunks index file {chunks_path}: {e}")
    
    
    #加载BM25检索器
    bm25_corpus = []
    bm25_nodes = []
    for chunk in chunks:
        text = chunk.get("text","").strip()
        node_id = chunk.get("node_id","")
        if text and node_id:
            bm25_corpus.append(text)
            bm25_nodes.append(TextNode(text=text,id_=node_id,metadata=chunk.get("metadata",{})))
    
    print(f"BM25检索器准备完成，文档数量: {len(bm25_corpus)}")
    #创建混合检索器
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_corpus=bm25_corpus,
        bm25_nodes=bm25_nodes,
        top_k=top_k,
        vector_weight=vector_weight,
        rrf_k=rrf_k
    )
    print(f"混合检索器创建成功，配置 - top_k: {top_k}, vector_weight: {vector_weight}, rrf_k: {rrf_k}")
    return hybrid_retriever


def query_with_hybrid_retriever(
    query: str,
    chunks_path: str = "chunks_index.json",
    vector_path: str = "vector_index"
    ) -> List[NodeWithScore]:
    hybrid_retriever = create_hybrid_retriever(
        vector_path = vector_path,
        chunks_path = chunks_path,
        top_k = 10,
        vector_weight = 0.5,
        rrf_k = 60,
        
    )
    query_bundle = QueryBundle(query_str=query)
    results = hybrid_retriever.retrieve(query_bundle)
    print(f"查询完成，返回结果数: {len(results)}")
    for idx, res in enumerate(results):
        print(f"结果 {idx+1}: Node ID: {res.node.node_id}, Score: {res.score}, Text: {res.node.get_content()[:100]}...")
    
    return results

#外部检索接口
def retriever_with_hybrid_retriever(
    query: str,
    chunks_path: str = "chunks_index.json",
    vector_path: str = "vector_index"
    ) -> str :
    results = query_with_hybrid_retriever(query, chunks_path, vector_path)
    result = make_prompt(query, results)
    return result
    
def make_prompt(query:str,results:List[NodeWithScore])->str:
    prompt = f"用户查询: {query}\n\n"
    prompt += "检索到的相关信息:\n"
    for idx,res in enumerate(results):
        prompt += f"{idx+1}. (Score: {res.score:.4f}) {res.node.get_content()}\n"
    prompt += "\n请基于以上信息回答用户查询。"
    return prompt

def pred_with_hybrid_retriever(args) -> str:
    model_name = args.model_name
    embed_model_name = args.embed_model_name
    query = args.query.strip()
    chunks_path = args.chunks_path
    vector_path = args.vector_path
    init_model(model_name,embed_model_name)
    model = Settings.llm
    results = query_with_hybrid_retriever(query, chunks_path, vector_path)
    prompt = make_prompt(query, results)
    response = model.complete(prompt).text
    response_json = {
        "query": query,
        "retrieved_results": [
            {
                "node_id": res.node.node_id,
                "score": res.score,
                "text": res.node.get_content()
            } for res in results
        ],
        "response": response
    }
    print(f"模型响应: {response_json}")
    return response_json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="使用混合检索器进行查询")
    parser.add_argument("--model_name", type=str, default="Ollama-Model", help="LLM模型名称")
    parser.add_argument("--embed_model_name", type=str, default="Ollama-Embedding", help="嵌入模型名称")
    parser.add_argument("--query", type=str, required=True, help="用户查询文本")
    parser.add_argument("--vector_path", type=str, default="vector_index", help="向量索引存储路径")
    parser.add_argument("--chunks_path", type=str, default="chunks_index.json", help="chunks索引文件路径")
    args = parser.parse_args()
    
    pred_with_hybrid_retriever(args)
        