# -*- coding: utf-8 -*-
"""知识库检索 Skill — 基于 DocumentRAG 的文档型 RAG。

依赖 DocumentRAG 实例，由 orchestrator 在 _context 里注入 {"document_rag": ...}。
"""

from typing import Any

from voice_agent.agent.skill_manager import Skill


def _search_knowledge(query: str, limit: int = 3, _context: Any = None) -> str:
    if _context is None or "document_rag" not in _context:
        return "知识库未启用。"
    rag = _context["document_rag"]
    if rag is None or rag.doc_count == 0:
        return "知识库为空，请先通过 /api/rag/load 加载文档。"

    passages = rag.search(query, top_k=limit, threshold=0.05)
    if not passages:
        return "未在知识库中找到相关内容。"
    return rag.format_context(passages)


skill_knowledge_search = Skill(
    name="search_knowledge",
    description=(
        "在本地知识库中检索相关文档内容。当用户询问专业问题、"
        "需要查阅文档/资料、或问'有没有相关资料'时使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词或问题",
            },
            "limit": {
                "type": "integer",
                "description": "返回段落数量，默认3",
            },
        },
        "required": ["query"],
    },
    handler=_search_knowledge,
    category="rag",
)