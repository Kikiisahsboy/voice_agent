# -*- coding: utf-8 -*-
"""MCP Server — 将 Agent 能力以 MCP 协议暴露。"""

import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AgentMCPServer:
    """将 Agent 能力暴露为 MCP Server。"""

    def __init__(
        self,
        memory_manager=None,
        skill_manager=None,
    ):
        self._memory_manager = memory_manager
        self._skill_manager = skill_manager
        self._server = None

    async def start(self):
        """启动 MCP Server（stdio 模式）。"""
        try:
            from mcp.server import Server
            from mcp.server.stdio import stdio_server
            from mcp.types import Tool, TextContent

            server = Server("voice-agent")
            self._server = server

            # ── 注册工具 ────────────────────────────

            @server.list_tools()
            async def list_tools() -> list[Tool]:
                tools = []

                if self._memory_manager:
                    tools.append(Tool(
                        name="search_memory",
                        description="搜索对话历史记忆",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "搜索关键词",
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "返回条数，默认3",
                                },
                            },
                            "required": ["query"],
                        },
                    ))

                    tools.append(Tool(
                        name="store_memory",
                        description="存储一条记忆",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "要存储的记忆内容",
                                },
                                "tags": {
                                    "type": "string",
                                    "description": "逗号分隔的标签",
                                },
                            },
                            "required": ["text"],
                        },
                    ))

                if self._skill_manager:
                    tools.append(Tool(
                        name="list_skills",
                        description="列出所有可用的技能",
                        inputSchema={
                            "type": "object",
                            "properties": {},
                        },
                    ))

                    tools.append(Tool(
                        name="execute_skill",
                        description="执行指定的技能",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "skill_name": {
                                    "type": "string",
                                    "description": "技能名称",
                                },
                                "args": {
                                    "type": "object",
                                    "description": "技能参数（JSON 对象）",
                                },
                            },
                            "required": ["skill_name"],
                        },
                    ))

                return tools

            @server.call_tool()
            async def call_tool(name: str, arguments: dict) -> list[TextContent]:
                if name == "search_memory" and self._memory_manager:
                    results = self._memory_manager.search_long_term(
                        arguments.get("query", ""),
                        top_k=arguments.get("limit", 3),
                    )
                    return [TextContent(
                        type="text",
                        text=json.dumps(results, ensure_ascii=False),
                    )]

                if name == "store_memory" and self._memory_manager:
                    from voice_agent.agent.memory_manager import MemoryManager
                    raw_tags = arguments.get("tags", "")
                    tag_list = [t.strip() for t in raw_tags.split(",") if t.strip()] if raw_tags else []
                    meta = {}
                    if tag_list:
                        meta["tags"] = tag_list
                    self._memory_manager._chroma.store(
                        text=arguments["text"],
                        metadata=meta,
                    )
                    return [TextContent(type="text", text="记忆已存储")]

                if name == "list_skills" and self._skill_manager:
                    desc = self._skill_manager.get_skill_descriptions()
                    return [TextContent(type="text", text=desc)]

                if name == "execute_skill" and self._skill_manager:
                    result = self._skill_manager.execute(
                        arguments["skill_name"],
                        arguments.get("args", {}),
                    )
                    return [TextContent(type="text", text=result)]

                return [TextContent(
                    type="text",
                    text=f"未知工具: {name}",
                )]

            # ── 启动 stdio server ────────────────────

            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

            logger.info("MCP Server 已启动")

        except ImportError:
            logger.warning("mcp SDK 未安装，MCP Server 不可用")
        except Exception as e:
            logger.error("MCP Server 启动失败: %s", e)

    async def stop(self):
        if self._server:
            try:
                await self._server.close()
            except Exception:
                pass
            self._server = None
