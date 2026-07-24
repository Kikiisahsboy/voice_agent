# -*- coding: utf-8 -*-
"""记忆搜索 Skill — 通过依赖注入获取 memory_manager。

设计：Skill 的 handler 接收 `_context` 参数（由 orchestrator 注入），
避免使用模块级全局变量，符合依赖注入原则。
"""

from typing import Any

from voice_agent.agent.skill_manager import Skill


def _search_memory(query: str, limit: int = 3, _context: Any = None) -> str:
    """搜索长期记忆。`_context` 由 orchestrator 注入 {"memory": MemoryManager}。"""
    if _context is None or "memory" not in _context or _context["memory"] is None:
        return "记忆系统未初始化。"
    mm = _context["memory"]
    results = mm.search_long_term(query, top_k=limit)
    if not results:
        return "未找到相关历史记忆。"
    lines = ["找到以下相关记忆："]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r}")
    return "\n".join(lines)


skill_memory_search = Skill(
    name="search_memory",
    description="搜索对话历史记忆。当用户询问'之前聊过什么'、'我记得我们讨论过...'、'回忆一下'时使用。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或主题",
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数量，默认3",
            },
        },
        "required": ["query"],
    },
    handler=_search_memory,
    category="memory",
)