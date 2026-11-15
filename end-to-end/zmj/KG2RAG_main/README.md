## 项目架构
KG2RAG-main/
├─data_yazhu/
    ├─hotpotqa
        ├─kgs/
            ├─extract_subkgs/  #知识图谱路径
                ├─实体1相关文本的关系三元组: `{"_id": [[s,p,o]],......}
                ├─实体2相关text的关系三元组:
                ├─.....
        ├─modified_data_sample.json  #知识图谱抽取与检索系统问答的完整语料
                                     ```
                                     [{
                                            "_id": f"{sample_id}_{q_id}",
                                            "question": question,
                                            "answer": ans_v4 if ans_v4 else ans,
                                            "type": "relation_mechanism",
                                            "level": "hard",
                                            "supporting_facts": supporting_facts
                                        },.......]
                                    ```
    ├─message_data.jsonl #压铸的原始语料（未清洗成此RAG算法兼容的格式）
├─data/
    ├─hotpotqa      #语料一（代码中使用的检索语料路径）
        ├─kgs/      #利用上述语料抽取的知识图谱
            ├─extract_subkgs/  #知识图谱路径
                ├─实体1相关文本的关系三元组: `{"_id": [[s,p,o]],......}
                ├─实体2相关text的关系三元组:
                ├─.....
        ├─hotpot_dev_distractor_v1.json     #代码中使用的检索语料，格式同modified_data_sample.json
        
    ├─MusiQue/      #语料二
        ├─musique_ans_v1.0_dev_mapped.jsonl

    ├─trivia_qa/    #语料三
        ├─trivia.json

├─model/       #本检索算法使用的embed模型和LLm模型，以及重排序的语义模型
    ├─bge-reranker-large-safetensors    #重排序模型
    ├─mxbai-embed-large-gguf        #embed模型
    ├─Orenguteng-llama3-8b-gguf     #LLM模型
├─code/
    ├─clear/                        #清洗模块
        └─create_context.py     #清洗脚本，把messages语料清洗为hotpot_dev_distractor_v1.json格式
    ├─preprocess/
        ├─hotpot_extraction_yazhu.py        #压铸语料创建知识图谱的脚本
        ├─hotpot_extraction.py          #原始data语料一hotpotqa创建知识图谱脚本
        └─musique_extraction.py         #原始data语料二MusiQue创建知识图谱脚本
        ├─trivia_extraction.py          #原始data语料三trivia创建知识图谱脚本
    ├─util/                         #工具模块
        ├─hotpot_evaluate_v1.py
        ├─kg_post_processor.py      #KG2RAG知识图谱进行节点扩展和组织过滤的核心代码，详见注释
        ├─kg_response_synthesizer.py        #响应器
        ├─musique_evaluate_v1.0.py          #因为这是语料二相关的模块，所以我没debug这个代码哈哈
        └─prompt_helper.py              #提示词构建模块
    ├─kg_rag_distractor.py          #KG2RAG实验入口，包含语料sample遍历、检索器创建与检索、问题查询
    ├─kg_rag_full.py                #同上，也是实验入口，只不过是这里data没有给出的变种语料的检索入口
