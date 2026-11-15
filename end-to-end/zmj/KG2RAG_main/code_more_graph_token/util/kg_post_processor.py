import random
import re
import time
import networkx as nx

from typing import List,Dict,Optional,Set
from FlagEmbedding import FlagReranker
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.bridge.pydantic import Field
from llama_index.core.instrumentation import get_dispatcher
dispatcher = get_dispatcher(__name__)

import string
def ngram_overlap(span,sent,n=3):
    while (len(span)<n) or (len(sent)<n):
        n -= 1
    if n<=0:
        return 0.0
    span = span.lower()
    sent = sent.lower()
    span_tokens = [token for token in span.split() if token not in string.punctuation]
    span_tokens = ''.join(span_tokens)
    sent_tokens = [token for token in sent.split() if token not in string.punctuation]
    sent_tokens = ''.join(sent_tokens)
    span_tokens = set([span_tokens[i:i+n] for i in range(len(span_tokens)-n+1)])
    sent_tokens = set([sent_tokens[i:i+n] for i in range(len(sent_tokens)-n+1)])
    overlap = span_tokens.intersection(sent_tokens)
    return float((len(overlap)+0.01)/(len(span_tokens)+0.01))

class NaivePostprocessor(BaseNodePostprocessor):
    """Naive Node Postprocessor."""

    @classmethod
    def class_name(cls) -> str:
        return "NaivePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes"""
        entity_order = {}
        sorted_nodes = []
        for i,node in enumerate(nodes):
            node_id = node.node.id_
            ent, seq_str = node_id.split('##')
            ctx_seq = int(seq_str)
            if ent not in entity_order:
                entity_order[ent] = len(entity_order)
            sorted_nodes.append((ent,ctx_seq,node))
        sorted_nodes.sort(key=lambda x:(entity_order[x[0]],x[1]))
        sorted_nodes = [node for _,_,node in sorted_nodes]

        prev_ent = ''
        for i in range(0,len(sorted_nodes)):
            temp_ent = sorted_nodes[i].node.id_.split('##')[0]
            if (prev_ent == temp_ent):
                sorted_nodes[i].node.text = sorted_nodes[i].node.text[len(temp_ent+': '):]
            if i<len(sorted_nodes)-1:
                next_ent = sorted_nodes[i+1].node.id_.split('##')[0]
                if next_ent!=temp_ent:
                    sorted_nodes[i].node.text += '\n'
            prev_ent = temp_ent
     
        return sorted_nodes
    
class KGRetrievePostProcessor(BaseNodePostprocessor):

    """KnowledgeGraph-based Node processor."""

    ents: Set[str] = Field
    doc2kg: Dict[str,Dict[str,List[List[str]]]] = Field
    chunks_index: Dict[str,Dict[str,str]] = Field

    @classmethod
    def class_name(cls) -> str:
        return "KGRetrievePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes"""
        
        #检索出来的节点数量
        top_k = len(nodes)
        #检索出来的节点中的实体id(实体+索引)
        retrieved_ids = set()
        #检索出来的节点中的实体名
        retrieved_ents = set()

        #扩展的实体集合
        related_ents = set()
        #高分实体集合
        highly_related_ents = set()

        textid2score = dict()
        #检索出来的实体的次数和得分
        ent_count = dict()
        ent_score = dict()
        
        
        #遍历检索出的节点，统计实体出现次数和得分，并将检索出的前一半节点中的实体作为高分实体加入
        for i in range(len(nodes)):
            node = nodes[i]
            node_id = node.node.id_
            retrieved_ids.add(node_id)
            textid2score[node_id] = node.score
            entity, seq_str = node_id.split('##')

            if (i<(top_k//2)) and (entity in retrieved_ents):
                highly_related_ents.add(entity)
            retrieved_ents.add(entity)

            if entity not in ent_count:
                ent_count[entity] = 0
                ent_score[entity] = 0.0
            ent_count[entity] += 1
            ent_score[entity] += node.score
        
        #将刚刚取出的实体中按照平均得分进行排序，取前两个加入高分实体
        sorted_ents = sorted(ent_count.keys(),key=lambda x:(ent_score[x]/ent_count[x]),reverse=True)
        for i in range(min(2,len(sorted_ents))):
            highly_related_ents.add(sorted_ents[i])

        #储存检索出来的、未在评分初筛的实体集合中的实体
        additional_ents = set()
        
        #遍历检索出来的节点，储存retrieved_ents相关的实体，作为第一次扩展
        for node in nodes:
            node_id = node.node.id_
            entity, seq_str = node_id.split('##')
            if (entity not in self.doc2kg) or (seq_str not in self.doc2kg[entity]):
                continue
            for triplet in self.doc2kg[entity][seq_str]:
                h,r,t = triplet
                triplet = [h,r,t]
                if (h in self.ents) and (h not in retrieved_ents):
                    additional_ents.add(h)
                    if h not in ent_count:
                        ent_count[h] = 0
                        ent_score[h] = 0.0
                    ent_count[h] += 1
                    ent_score[h] += node.score
                if (t in self.ents) and (t not in retrieved_ents):
                    additional_ents.add(t)
                    if t not in ent_count:
                        ent_count[t] = 0
                        ent_score[t] = 0.0
                    ent_count[t] += 1
                    ent_score[t] += node.score

        #基点实体集合
        additional_ents = additional_ents.union(retrieved_ents)
        
        #扩展次数
        hops = 1
        
        #以初步扩展后的实体为基点，遍历与其相关的每个实体，统计其出现次数和总得分，并加入扩展实体集合中
        for hop in range(hops):
            related_ents = related_ents.union(additional_ents)
            temp_ents = set(additional_ents)
            additional_ents = set()
            for ent in temp_ents:

                if (ent not in self.doc2kg) or (len(self.doc2kg[ent])==0):
                    continue
                for idx_seq_str in self.doc2kg[ent]:
                    if len(self.doc2kg[ent][idx_seq_str])==0:
                        continue
                    ctx_id = f'{ent}##{idx_seq_str}'
                    if ctx_id in retrieved_ids:
                        continue
                    
                    for triplet in self.doc2kg[ent][idx_seq_str]:
                        h,r,t = triplet

                        if (h in self.ents) and (h not in related_ents):
                            additional_ents.add(h)
                            if h not in ent_count:
                                ent_count[h] = 0
                                ent_score[h] = 0.0
                            ent_count[h] += 1
                            ent_score[h] += float((ent_score[ent])/(ent_count[ent]))
                        if (t in self.ents) and (t not in related_ents):
                            additional_ents.add(t)
                            if t not in ent_count:
                                ent_count[t] = 0
                                ent_score[t] = 0.0
                            ent_count[t] += 1
                            ent_score[t] += float((ent_score[ent])/(ent_count[ent]))

        related_ents = related_ents.union(additional_ents)
        
        #储存与扩展出的实体有关系的文本id，因为最终的目的是根据扩展出的实体对原来检索出来的文本进行补充，所以要记录id，用于寻找文本
        additional_ids = set()

        avg_score = float(sum([node.score for node in nodes])/len(nodes))
        
        retrieved_ents = retrieved_ents-highly_related_ents
        for ent in (related_ents-retrieved_ents):
            if (ent not in self.chunks_index) or (len(self.chunks_index[ent])==0) or (ent not in self.doc2kg):
                continue
            for idx_seq_str in self.chunks_index[ent]:
                ctx_id = f'{ent}##{idx_seq_str}'
                if ctx_id in retrieved_ids:
                    continue
                additional_ids.add(ctx_id)
                textid2score[ctx_id] = 0.0
                if ent in ent_score:
                    textid2score[ctx_id] += (ent_score[ent]+avg_score)/(ent_count[ent]+1)

        added_nodes = []
        for ctx_id in additional_ids:
            ent, seq_str = ctx_id.split('##')
            if ent in self.chunks_index:
                if seq_str in self.chunks_index[ent]:
                    ctx_text = self.chunks_index[ent][seq_str]
                    node = TextNode(id_=ctx_id,text=ctx_text)
                    node = NodeWithScore(node=node,score=textid2score[ctx_id])
                    added_nodes.append(node)
        added_nodes = sorted(added_nodes,key=lambda x:x.score,reverse=True)

        nodes = nodes+added_nodes
        return nodes


#过滤的目的是为了将节点通过实体组织起来，排除串不起来的孤立点和串起来但是语义和查询问题不符的节点集群
#1.将节点文本中的实体关联起来，每个节点一条边，节点表示一个文本块，相当于把不同文本块通过实体组织起来
#2.计算连通子图，将图中能关联起来的实体放在一个集群，这些实体之间的边也就是节点对应的文档能表达一个完整的逻辑
#3.目标是将检索到的文本通过知识图谱的实体关系关联起来，同时去除冗余边，保证得分尽可能大，即寻找最大生成树
#4.将每一个子图的节点对应的文本以及文本里面的关系组织拼接起来，和查询问题进行匹配，通过匹配的得分对每一个子图也就是每一种逻辑关系进行排序。

class GraphFilterPostProcessor(BaseNodePostprocessor):

    """KnowledgeGraph-based Node processor."""
    topk: int = Field
    use_tpt: bool = Field
    ents: Set[str] = Field
    doc2kg: Dict[str,Dict[str,List[List[str]]]] = Field
    chunks_index: Dict[str,Dict[str,str]] = Field
    reranker: FlagReranker = Field

    @classmethod
    def class_name(cls) -> str:
        return "GraphFilterPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes"""
        ents = set()
        rels = set()

        g = nx.MultiGraph()

        #获取节点中的实体
        for node in nodes:
            ent, seq_str = node.node.id_.split('##')
            ents.add(ent)

        #基于节点实体，将与之相关的实体加入ents，并添加关系边
        for node in nodes:
            ent, seq_str = node.node.id_.split('##')
            if (ent not in self.doc2kg) or (seq_str not in self.doc2kg[ent]) or (len(self.doc2kg[ent][seq_str])==0):
                continue
            for triplet in self.doc2kg[ent][seq_str]:
                h,r,t = triplet
                h = h.strip()
                r = r.strip()
                t = t.strip()
                triplet = [h,r,t]
                ents.add(h)
                ents.add(t)
                rels.add(r)
                #！！！添加关系边（可能出现自回路的边，自己连自己）
                g.add_edge(h,t,rel=r,source=node.node.id_,weight=node.score)

        mentioned_ents = set()
        mentioned_rels = set()

        #遍历每一个实体和关系，利用词匹配算法计算其与查询问题的相似度，无embedding，不是余弦相似度，不具备语义匹配
        for ent in ents:
            #ngram_overlap是将两个字符串按固定词数分词，计算两个分词列表之间的重复次数除以总词数得到得分
            overlap_score = ngram_overlap(ent,query_bundle.query_str)
            if overlap_score>=0.90:
                mentioned_ents.add(ent)
        for rel in rels:
            overlap_score = ngram_overlap(rel,query_bundle.query_str)
            if overlap_score>=0.90:
                mentioned_rels.add(rel)

        #遍历每一个实体，将存在mentioned_ents和mentioned_rels的三元组实体补全，即将每一个三元组中漏选的实体加入mentioned_ents
        for node in nodes:
            ent, seq_str = node.node.id_.split('##')
            if (ent not in self.doc2kg) or (seq_str not in self.doc2kg[ent]) or (len(self.doc2kg[ent][seq_str])==0):
                continue
            for triplet in self.doc2kg[ent][seq_str]:
                h,r,t = triplet
                triplet = [h,r,t]
                if (h in mentioned_ents) and (r in mentioned_rels) and (not t in mentioned_ents):
                    mentioned_ents.add(t)
                if (t in mentioned_ents) and (r in mentioned_rels) and (not h in mentioned_ents):
                    mentioned_ents.add(h)

        #遍历扩展之后的实体集合mentioned_ents，添加其中任意两个实体的关系边
        mentioned_ents_list = list(mentioned_ents)
        for i in range(len(mentioned_ents_list)):
            for j in range(i+1,len(mentioned_ents_list)):
                if (g.has_edge(mentioned_ents_list[i],mentioned_ents_list[j])) or (g.has_edge(mentioned_ents_list[j],mentioned_ents_list[i])):
                    continue
                g.add_edge(mentioned_ents_list[i],mentioned_ents_list[j],rel='cooccurrence',source='query',weight=0.0)
        
        wanted_ents = set(mentioned_ents)
        wanted_rels = set(mentioned_rels)

        early_exit = False

        #寻找关系图中的最大连通分量，每一个连通分量是一个实体集群
        wccs = list(nx.connected_components(g))
        #集群中实体数量越多，信息越多      
        sorted_wccs = sorted(wccs,key=len,reverse=True)

        cand_ctxs_lists = list()
        #遍历连通子图，通过最大生成树的算法去将子图中得分最高的边保留下来，并将边对应的node_id加入该连通子图的cand_ctxs_list
        for i in range(len(sorted_wccs)):
            if (early_exit) and (len(wanted_ents)==0):
                break
            cand_ctxs_list = []
            wcc = sorted_wccs[i]
            for ent in wcc:
                if ent in wanted_ents:
                    wanted_ents.remove(ent)
            if len(wcc)>1:
                subgraph = g.subgraph(wcc)
                mst = nx.maximum_spanning_tree(subgraph,weight='weight')
                cand_ctx_list = []
                for cand_edge in mst.edges(data=True):
                    if cand_edge[2]['source']=='query':
                        continue
                    else:
                        cand_ctx_list.append(cand_edge[2]['source'])
                cand_ctxs_list.append(cand_ctx_list)
            else:
                sorted_edges = sorted(g.subgraph(wcc).edges(data=True),key=lambda x:x[2]['weight'],reverse=True)
                for edge in sorted_edges:
                    if edge[2]['source']=='query':
                        continue
                    else:
                        #疑点！！！如果自己跟自己的关系有很多，那么这里是否会漏掉关系边？
                        cand_ctxs_list.append([edge[2]['source'],])
                        break
            cand_ctxs_lists.append(cand_ctxs_list)

        #原本是[[[node_id,node_id]],[[node_id,node_id]]]，最里面的列表是某一个连通子图中得分最高的关系边表示的node_id，
        #外面只需要一层列表表示整个图即可，但是构造cand_ctxs_lists的写法造成冗余，这里要压缩一个维度，去除列表维度的冗余
        cand_ids_lists = list()
        for cand_ctxs_list in cand_ctxs_lists:
            cand_ids_lists.extend(cand_ctxs_list)
        
        cand_tpts = []
        cand_strs = []
        #将每一个子图的节点对应的文本以及文本里面的关系组织拼接起来，和查询问题进行匹配，通过匹配的得分对每一个子图也就是每一种逻辑关系进行排序。
        #遍历每一个集群，将集群中的node_id对应的文本和实体关系拼接起来作为长文本，每个集群一个长文本ctx_str，类似"A文本 Relational facts: A的关系. B文本 Relational facts: B的关系."
        for cand_ids_list in cand_ids_lists:
            ctx_str = ''
            tpt_str = ''
            for cand_id in cand_ids_list:
                cand_ent, seq_str = cand_id.split('##')
                ctx_str += self.chunks_index[cand_ent][seq_str]
                if (self.use_tpt) and (cand_ent in self.doc2kg) and (seq_str in self.doc2kg[cand_ent]) and (len(self.doc2kg[cand_ent][seq_str])>0):
                    tpt_str += ', '.join([f'{h} has/is {r} {t}' for h,r,t in self.doc2kg[cand_ent][seq_str][:min(len(self.doc2kg[cand_ent][seq_str]),3)]])
                    if len(tpt_str)>0:
                        ctx_str = f'{ctx_str} Relational facts: {tpt_str}.'
            cand_strs.append(ctx_str)
            cand_tpts.append(tpt_str)
        

        if len(cand_strs)==0:
            scores = self.reranker.compute_score([(query_bundle.query_str,node.node.text) for node in nodes])
            sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
            wanted_nodes = [nodes[sorted_seqs[i]] for i in range(min(self.topk,len(sorted_seqs)))]
            return wanted_nodes
        
      
        wanted_ctxs = []
        #将每一个连通子图中的节点组织起来的文本和查询向量进行语义匹配，并根据得分进行排序
        scores = self.reranker.compute_score([(query_bundle.query_str,cand_str) for cand_str in cand_strs])
        # scores = self.reranker.compute_score([(query_bundle.query_str,cand_tpt) for cand_tpt in cand_tpts])
        sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
        
        #在目标节点列表的长度小于检索允许的最大数量之前，一直贪心地添加每一个连通子图的节点
        for seq in sorted_seqs:
            if len(set(wanted_ctxs)|set(cand_ids_lists[seq]))>self.topk:
                break
            wanted_ctxs.extend(cand_ids_lists[seq])
            wanted_ctxs = list(set(wanted_ctxs))

        #如果选出来的节点数量小于检索允许的最大数量的一半，那么就从被筛掉的节点中中选择一些节点得分高的加入凑数
        if len(wanted_ctxs)<self.topk//2:
            cands = [(query_bundle.query_str,node.node.text,) for node in nodes if node.node.id_ not in wanted_ctxs]
            if len(cands)>0:
                scores = self.reranker.compute_score(cands)
                sorted_seqs = sorted(range(len(scores)),key=lambda x:scores[x],reverse=True)
                for seq in sorted_seqs[:self.topk]:
                    if nodes[seq].node.id_ not in wanted_ctxs:
                        wanted_ctxs.append(nodes[seq].node.id_)
                        if len(wanted_ctxs)>=self.topk:
                            break

        wanted_nodes = []
        for node in nodes:
            if node.node.id_ in wanted_ctxs:
                wanted_nodes.append(node)
        return wanted_nodes