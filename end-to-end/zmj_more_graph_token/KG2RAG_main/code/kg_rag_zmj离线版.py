import os
import copy
import ujson as json
import argparse
from tqdm import tqdm
from FlagEmbedding import FlagReranker
from llama_index.core import Settings,VectorStoreIndex,PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core.schema import TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from util.kg_post_processor import NaivePostprocessor,KGRetrievePostProcessor,ngram_overlap, GraphFilterPostProcessor
from util.kg_response_synthesizer import get_response_synthesizer

from log.logger import setup_logger
logger = setup_logger("kg_rag")


def read_data(args):
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'{data_path} not found')
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f'Loaded {len(data)} samples from {data_path}')
    except Exception as e:
        logger.exception(f'Error reading data from {data_path}: {e}')
        raise e
    return data


def init_model(args):

    Settings.llm = Ollama(model=args.model_name,request_timeout=200)
    Settings.embed_model = OllamaEmbedding(model_name=args.embed_model_name)

def read_kg(args, data):
    ents = set()
    for sample in data:
        for ctx in sample['context']:
            ents.add(ctx[0])
    kg_dir = args.kg_dir
    doc2kg = dict()
    logger.info('Loading KGs')
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
    logger.info(f'Loaded kg for {len(doc2kg.keys())} entities')
    return doc2kg

#保存每个sample推理结果
write_dict = {}
def write_pre_dict(args,prediction,sample_id=None):
    result_dir = args.result_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if sample_id in write_dict:
        logger.warning(f"保存推理结果时出现重复 sample_id: {sample_id}, 跳过保存该 prediction")
        return
    out_path = os.path.join(result_dir, f'{sample_id.replace("/","_")}.json')
    write_dict[sample_id] = prediction
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(prediction, f, ensure_ascii=False, indent=2)
        logger.info(f'Prediction written to {out_path}')
    except Exception as e:
        logger.exception(f'Error writing prediction to {out_path}: {e}')
        raise e


def write_prediction(args, prediction):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    with open(args.result_path, 'w', encoding='utf-8') as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)
    logger.info(f'Prediction written to {args.result_path}')


# 后处理器，知识图谱增强索引结果
def process_sample(args, sample, kg):
    sample_id = sample['sample_id']
    sample_question = sample['question']

    subkg = dict()
    doc_chunks = []
    chunks_index = dict()

    ctxs = sample['context']
    ents = [ctx[0] for ctx in ctxs]

    for ctx in ctxs:
        ent = ctx[0]
        chunks_index[ent] = {}
        for text_dict in ctx[1]:
            chunk_id = text_dict.get('chunk_id', '')
            text = f'{ent}: {text_dict.get("text", " ")}'

            if (ent in kg) and (chunk_id in kg[ent]) and (len(kg[ent][chunk_id]) > 0):
                if ent not in subkg:
                    subkg[ent] = dict()
                target_kg = kg[ent][chunk_id]
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

    #将节点列表向量化，创建向量库
    index = VectorStoreIndex(doc_chunks)
    #为向量库创建检索器
    retriever = VectorIndexRetriever(index=index,similarity_top_k=args.top_k)
    #创建提示词模板
    qa_rag_template_str = 'Context information is below.\n{context_str}\nThink step by step but give a short factoid answer (as few words as possible) based on the context and your own knowledge.\nQ: Were Scott Derrickson and Ed Wood of the same nationality?\nA: Yes.\nQ: Who was born earlier, Emma Bull or Virginia Woolf?\nA: Adeline Virginia Woolf.\nQ: The arena where the Lewiston Maineiacs played their home games can seat how many people?\nA: 3,677 seated.\nQ: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?\nA: Chief of Protocol.\n---------------------\nQ: {query_str}\nA: '
    qa_rag_prompt_template = PromptTemplate(qa_rag_template_str)
    #创建响应器
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT,text_qa_template=qa_rag_prompt_template)
    #创建根据知识图谱对实体进行扩展的处理器
    expansion_pp = KGRetrievePostProcessor(ents=ents, doc2kg=subkg, chunks_index=chunks_index)
    bge_reranker = FlagReranker(model_name_or_path=args.reranker, device=3)
    filter_pp = GraphFilterPostProcessor(use_tpt=args.use_tpt, topk=args.top_k, ents=ents, doc2kg=subkg, chunks_index=chunks_index, reranker=bge_reranker)
    naive_pp = NaivePostprocessor()
    #创建最终的查询引擎
    query_engine = RetrieverQueryEngine(retriever=retriever,response_synthesizer=response_synthesizer,node_postprocessors=[expansion_pp,filter_pp,naive_pp])

    try:
        response = query_engine.query(sample_question)
        prediction = response.response
        source_nodes = [{"node_id": str(source_node.node.id_), "source_text": source_node.node.text,"score": source_node.score} for source_node in response.source_nodes]
    except Exception as e:
        logger.error(f'Sample {sample_id}, Error: {e}')
        prediction = ''
        source_nodes = []
    return sample_id, prediction, source_nodes

def kgrag_distractor_predict(args,data,kg):
    prediction = {'answer':{},'source_nodes':{}}
    sps_count = 0
    for sample in tqdm(data):
        sample_prediction_dict = {'answer':{},'source_nodes':{}}
        sample_id,sample_prediction,sample_source_nodes = process_sample(args,sample,kg)
        prediction['answer'][sample_id] = sample_prediction
        sample_prediction_dict['answer'][sample_id] = sample_prediction
        prediction['source_nodes'][sample_id] = sample_source_nodes
        sample_prediction_dict['source_nodes'][sample_id] = sample_source_nodes
        logger.info(f"尝试保存 sample_id: {sample_id} 的推理结果")
        try:
            write_pre_dict(args, sample_prediction_dict, sample_id)
            logger.info(f"sample_id: {sample_id} 的推理结果保存完成")
        except Exception as e:
            logger.exception(f"保存 sample_id: {sample_id} 的推理结果时发生错误: {e}")
            raise e
        sps_count += len(sample_source_nodes)
    logger.info(f'Average number of supporting facts: {sps_count/len(data)}')
    return prediction

def main(args):
    data = read_data(args)
    init_model(args)
    kg = read_kg(args,data)
    prediction = kgrag_distractor_predict(args,data,kg)
    logger.info("尝试保存 dataset 的推理结果")
    try:
        write_prediction(args, prediction)
        logger.info("dataset 的推理结果保存完成")
    except Exception as e:
        logger.exception(f"保存 dataset 的推理结果时发生错误: {e}")
        raise e

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./data/data_vk3/sample/sample.json', help='Path to the sample data file')
    parser.add_argument('--kg_dir', type=str, default='./data/data_vk3/kgs', help='Directory of the KGs')
    parser.add_argument('--use_tpt', type=bool, default=False, help='Whether to use triplet representation')
    parser.add_argument('--result_path', type=str, default='./output/cleared_test/results.json', help='Path to the result file')
    parser.add_argument('--result_dir', type=str, default='./output/cleared_test/result_dir', help='Directory of individual sample result files')
    
    
    
    parser.add_argument('--embed_model_name', type=str, default='mixedbread-ai-mxbai-embed-large-v1', help='Ollama embedding model name')
    parser.add_argument('--model_name', type=str, default='Orenguteng-llama3-8b', help='Ollama model name')
    parser.add_argument('--reranker', type=str, default='./model/bge-reranker-large-safetensors', help='Path of the reranker model')
    parser.add_argument('--top_k', type=int, default=10, help='Top k similar documents')

    args = parser.parse_args()
    main(args)