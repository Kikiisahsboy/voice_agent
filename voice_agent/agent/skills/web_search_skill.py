# -*- coding: utf-8 -*-
"""联网搜索 Skill — 基于 DuckDuckGo HTML（无需 API Key）。"""

import re

from voice_agent.agent.skill_manager import Skill


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html)


def _web_search(query: str, limit: int = 5) -> str:
    if not query or not query.strip():
        return "请提供搜索关键词。"

    try:
        import requests
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query.strip(), "kl": "cn-zh"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if resp.status_code != 200:
            return f"搜索失败（HTTP {resp.status_code}）。"

        html = resp.text
        # 提取 result__a（标题）和 result__snippet（摘要）
        results = re.findall(
            r'<a[^>]*class="result__a"[^>]*>(.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html,
            flags=re.DOTALL,
        )
        if not results:
            return "未找到相关结果。"

        lines = [f"搜索 '{query}' 的结果："]
        for i, (title_html, snippet_html) in enumerate(results[:limit], 1):
            title = _strip_tags(title_html).strip()
            snippet = _strip_tags(snippet_html).strip()
            if title:
                lines.append(f"{i}. {title}")
            if snippet:
                lines.append(f"   {snippet[:160]}")
        return "\n".join(lines)
    except Exception as e:
        return f"搜索出错: {e}"


skill_web_search = Skill(
    name="web_search",
    description=(
        "联网搜索最新信息。当用户询问新闻、近期事件、"
        "知识截止之后的信息，或需要实时数据时使用。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或问题",
            },
            "limit": {
                "type": "integer",
                "description": "返回结果数量，默认5",
            },
        },
        "required": ["query"],
    },
    handler=_web_search,
    category="search",
)