import os
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_PATH not in sys.path:
    sys.path.insert(0, ROOT_PATH)
import ujson as json
from tqdm import tqdm
from config.config import MODEL_NAME,CONTEXT_PATH,TEST_DATA_PATH, SAMPLE_PATH
from llama_index.llms.ollama import Ollama
from log.logger import setup_logger

logger = setup_logger("create_sample")

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


def read_questions():
    logger.info("加载测试问题 question_data...")
    if not os.path.exists(TEST_DATA_PATH):
        logger.exception(f"[question_data] 文件不存在: {TEST_DATA_PATH}")
        raise FileNotFoundError(f"{TEST_DATA_PATH} not found")
    try:
        with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
            question_data = json.load(f)
            logger.info(f"[question_data] 加载成功: {TEST_DATA_PATH}\nlen={len(test_data)}")
    except Exception as e:
        logger.exception(f"[question_data] 加载失败: {TEST_DATA_PATH}\n错误: {e}")
        raise
    return question_data

def read_context():
    logger.info("加载上下文数据 context_data...")
    if not os.path.exists(CONTEXT_PATH):
        logger.exception(f"[Context data] 文件不存在: {CONTEXT_PATH}")
        raise FileNotFoundError(f"{CONTEXT_PATH} not found")
    try:
        with open(CONTEXT_PATH, 'r', encoding='utf-8') as f:
            context_data = json.load(f)
            logger.info(f"[Context data] 加载成功: {CONTEXT_PATH}\nlen={len(context_data)}")
    except Exception as e:
        logger.exception(f"[Context data] 加载失败: {CONTEXT_PATH}\n错误: {e}")
        raise
    return context_data


def create_sample():
    logger.info(f"加载 {MODEL_NAME} 模型...")
    llm = Ollama(model=MODEL_NAME,request_timeout=600.0)
    logger.info(f"模型 [{MODEL_NAME}] 加载成功")
    question_data = read_questions()

    context_data = read_context()
    sample_data = []
    questions  = question_data['questions']
    print("开始创建sample样本...")
    for i,question in tqdm(enumerate(questions),desc="创建sample样本"):
        supporting_facts = question_ent(llm, question)
        sample_data.append(
            {
                "sample_id": str(i),
                "question": question,
                "answer": " ",
                "context": context_data,
                "type": "relation_mechanism",
                "level": "hard",
                "supporting_facts": supporting_facts
            }
        )
    try:
        with open(SAMPLE_PATH , "w" , encoding="utf-8") as f:
            json.dump(sample_data,f,ensure_ascii=False, indent=2)
            logger.info(f"[sample] 创建成功, len={len(sample_data)}, 已保存至 {SAMPLE_PATH}")
    except Exception as e:
        logger.exception(f"保存 sample_data 失败: {SAMPLE_PATH}")
        raise
    
if __name__ == "__main__":
    create_sample()