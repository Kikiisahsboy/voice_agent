# -*- coding: utf-8 -*-
"""MCP Server — 将 Agent 能力以 MCP 协议暴露。"""

import asyncio
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AgentMCPServer:
    """将 Agent 能力暴露为 MCP Server（stdio 模式）。"""

    def __init__(
        self,
        memory_manager=None,
        skill_manager=None,
    ):
        self._memory_manager = memory_manager
        self._skill_manager = skill_manager
        self._server = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None

    async def start(self):
        """启动 MCP Server（stdio 模式）。通过 task 在后台常驻。"""
        try:
            from mcp.server import Server
            from mcp.server.stdio import stdio_server
            from mcp.types import Tool, TextContent

            server = Server("voice-agent")
            self._server = server

            @server.list_tools()
            async def list_tools() -> list[Tool]:
                tools: list[Tool] = []

                if self._memory_manager:
                    tools.append(Tool(
                        name="search_memory",
                        description="搜索对话历史记忆",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "搜索关键词"},
                                "limit": {"type": "integer", "description": "返回条数，默认3"},
                                "memory_type": {
                                    "type": "string",
                                    "enum": ["fact", "preference", "event"],
                                    "description": "限定记忆类型",
                                },
                            },
                            "required": ["query"],
                        },
                    ))
                    tools.append(Tool(
                        name="store_memory",
                        description="存储一条长期记忆",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "记忆内容"},
                                "memory_type": {
                                    "type": "string",
                                    "enum": ["fact", "preference", "event"],
                                    "description": "记忆类型",
                                },
                                "tags": {"type": "string", "description": "逗号分隔的标签"},
                            },
                            "required": ["text"],
                        },
                    ))
                    tools.append(Tool(
                        name="update_user_profile",
                        description="更新用户画像",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "key": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["key", "value"],
                        },
                    ))

                if self._skill_manager:
                    tools.append(Tool(
                        name="list_skills",
                        description="列出所有可用的技能",
                        inputSchema={"type": "object", "properties": {}},
                    ))
                    tools.append(Tool(
                        name="execute_skill",
                        description="执行指定的技能",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "skill_name": {"type": "string"},
                                "args": {"type": "object"},
                            },
                            "required": ["skill_name"],
                        },
                    ))

                return tools

            @server.call_tool()
            async def call_tool(name: str, arguments: dict) -> list[TextContent]:
                if name == "search_memory" and self._memory_manager:
                    types = arguments.get("memory_type")
                    mt_list = [types] if types else None
                    results = self._memory_manager.search_long_term(
                        arguments.get("query", ""),
                        top_k=arguments.get("limit", 3),
                        memory_types=mt_list,
                    )
                    return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False))]

                if name == "store_memory" and self._memory_manager:
                    tags_raw = arguments.get("tags", "")
                    meta = {
                        "tags": [t.strip() for t in tags_raw.split(",") if t.strip()]
                    } if tags_raw else {}
                    self._memory_manager.store_long_term(
                        text=arguments["text"],
                        memory_type=arguments.get("memory_type", "event"),
                        metadata=meta,
                    )
                    return [TextContent(type="text", text="记忆已存储")]

                if name == "update_user_profile" and self._memory_manager:
                    self._memory_manager.update_user_profile(
                        arguments["key"], arguments["value"]
                    )
                    return [TextContent(type="text", text="用户画像已更新")]

                if name == "list_skills" and self._skill_manager:
                    desc = self._skill_manager.get_skill_descriptions()
                    return [TextContent(type="text", text=desc)]

                if name == "execute_skill" and self._skill_manager:
                    result = self._skill_manager.execute(
                        arguments["skill_name"],
                        arguments.get("args") or {},
                        context={"memory": self._memory_manager},
                    )
                    return [TextContent(type="text", text=result)]

                return [TextContent(type="text", text=f"未知工具: {name}")]

            # 在后台 task 中常驻
            self._stop_event = asyncio.Event()

            async def _serve():
                async with stdio_server() as (read, write):
                    await server.run(
                        read, write, server.create_initialization_options()
                    )

            self._task = asyncio.create_task(_serve())
            logger.info("MCP Server 已启动（后台 task）")

        except ImportError:
            logger.warning("mcp SDK 未安装，MCP Server 不可用")
        except Exception as e:
            logger.error("MCP Server 启动失败: %s", e)

    async def stop(self):
        if self._stop_event:
            self._stop_event.set()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
        self._task = None
        self._server = None