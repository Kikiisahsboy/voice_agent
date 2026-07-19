# -*- coding: utf-8 -*-
"""记忆搜索 Skill — 搜索对话历史。"""

from voice_agent.agent.skill_manager import Skill

# handler 在 orchestrator 初始化时注入 _memory_manager 引用
_memory_manager = None  # type: ignore


def _set_memory_manager(mm):
    global _memory_manager
    _memory_manager = mm


def _search_memory(query: str, limit: int = 3) -> str:
    if _memory_manager is None:
        return "记忆系统未初始化。"
    results = _memory_manager.search_long_term(query, top_k=limit)
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
