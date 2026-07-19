# -*- coding: utf-8 -*-
"""MCP Client — 连接外部 MCP Server，将其 tools 转为 Ollama 格式。"""

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP 客户端，通过 stdio 连接外部 MCP Server。"""

    def __init__(self):
        self._sessions: dict[str, Any] = {}
        self._tools: list[dict] = []
        self._connected = False
        self._server_command: Optional[str] = None
        self._server_args: list[str] = []

    async def connect(
        self,
        server_command: str,
        server_args: Optional[list[str]] = None,
        server_name: str = "default",
    ) -> bool:
        """
        连接到 MCP Server 进程并获取其工具列表。

        Args:
            server_command: 启动 MCP Server 的命令 (如 'npx', 'python')
            server_args: 命令参数 (如 ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'])
            server_name: 服务名称标识
        """
        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client, StdioServerParameters

            self._server_command = server_command
            self._server_args = server_args or []

            params = StdioServerParameters(
                command=server_command,
                args=self._server_args,
            )

            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    self._tools = []
                    for tool in result.tools:
                        self._tools.append({
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema or {
                                "type": "object",
                                "properties": {},
                            },
                        })
                    self._sessions[server_name] = session
                    self._connected = True
                    logger.info(
                        "MCP Client 已连接 '%s': %d 个工具",
                        server_name,
                        len(self._tools),
                    )
                    return True
        except ImportError:
            logger.warning("mcp SDK 未安装，MCP Client 不可用")
            return False
        except Exception as e:
            logger.error("MCP Client 连接失败: %s", e)
            return False

    def get_tools_as_ollama_format(self) -> list[dict]:
        """将 MCP tools 转为 Ollama tool calling 格式。"""
        result = []
        for tool in self._tools:
            params = tool.get("parameters", {})
            # 确保 parameters 有 type 字段
            if "type" not in params:
                params = {"type": "object", "properties": params, "required": []}

            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": params,
                },
            })
        return result

    async def call_tool_async(self, tool_name: str, args: dict) -> str:
        """异步调用 MCP Server 的工具。"""
        if not self._connected:
            return "MCP Client 未连接"

        try:
            from mcp.types import CallToolResult

            session = next(iter(self._sessions.values()))
            result: CallToolResult = await session.call_tool(tool_name, args)
            if result.content:
                return result.content[0].text
            return str(result)
        except Exception as e:
            return f"MCP 工具调用失败: {e}"

    def call_tool(self, tool_name: str, args: dict) -> str:
        """同步包装器，供 orchestrator 调用。"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                future = asyncio.run_coroutine_threadsafe(
                    self.call_tool_async(tool_name, args), loop
                )
                return future.result(timeout=30)
            else:
                return asyncio.run(self.call_tool_async(tool_name, args))
        except Exception as e:
            return f"MCP 工具调用失败: {e}"

    async def close(self):
        for session in self._sessions.values():
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        self._sessions.clear()
        self._tools = []
        self._connected = False
