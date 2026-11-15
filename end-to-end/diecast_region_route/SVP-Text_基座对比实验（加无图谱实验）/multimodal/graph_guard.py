from __future__ import annotations

from typing import List


def get_graph_evidence(info_text: str, question: str, top_k: int = 8) -> List[str]:
    """推理后置图谱证据接口。训练阶段不调用。"""
    try:
        from llm_backend import retrieve_graph_context
    except Exception:
        return []
    try:
        ctx, _ = retrieve_graph_context(info_text, question, graph_top_k=top_k)
        if not ctx:
            return []
        return [x for x in ctx.splitlines() if x.strip()]
    except Exception:
        return []
