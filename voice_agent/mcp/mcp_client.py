# -*- coding: utf-8 -*-
"""MCP Client — 长连接到外部 MCP Server，将其 tools 转为 Ollama 格式。

设计要点：
- 内部维护一个独立 asyncio event loop 线程，session 在该 loop 中常驻，
  避免 `async with` 退出后连接失效的问题。
- call_tool 通过 `loop.call_soon_threadsafe` 投递，避免
  `run_coroutine_threadsafe + future.result` 的死锁陷阱。
"""

import asyncio
import logging
import queue
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP 客户端，通过 stdio 长连接到外部 MCP Server。"""

    def __init__(self):
        self._sessions: dict[str, Any] = {}
        self._session_meta: dict[str, dict] = {}  # name -> {read, write, task}
        self._tools: list[dict] = []
        self._tool_index: dict[str, str] = {}  # tool_name -> server_name

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._call_responses: "queue.Queue[tuple[str, Any]]" = queue.Queue()

    # ── 长连接管理 ─────────────────────────────────────

    def start(self):
        """启动后台 event loop 线程。必须在 connect 之前调用。"""
        if self._thread and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name="mcp-client-loop", daemon=True
        )
        self._thread.start()
        logger.info("MCP Client 后台 loop 已启动")

    def stop(self):
        """停止后台 loop 并断开所有 session。"""
        if not self._loop:
            return
        try:
            # 在后台 loop 上调度断开
            async def _shutdown():
                for meta in self._session_meta.values():
                    for sess in meta.get("sessions", []):
                        try:
                            await sess.__aexit__(None, None, None)
                        except Exception:
                            pass
                tasks = [t for t in asyncio.all_tasks(self._loop)]
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            future = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            try:
                future.result(timeout=5)
            except Exception:
                pass

            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=3)
        finally:
            self._loop = None
            self._thread = None
            self._sessions.clear()
            self._session_meta.clear()
            self._tools.clear()
            self._tool_index.clear()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit_coro(self, coro) -> Any:
        """从任意线程把 coroutine 投递到后台 loop 并等待结果。"""
        if not self._loop or not self._thread or not self._thread.is_alive():
            raise RuntimeError("MCP Client 后台 loop 未启动")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    # ── 连接外部 Server ─────────────────────────────────

    def connect(
        self,
        server_command: str,
        server_args: Optional[list[str]] = None,
        server_name: str = "default",
    ) -> bool:
        """
        长连接到 MCP Server。

        在后台 loop 内建立 stdio_client + ClientSession 并常驻。
        """
        self.start()
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=server_command,
                args=server_args or [],
            )

            async def _setup():
                # 注意：stdio_client 的 async with 不能直接在后台 task 里
                # 反复 enter/exit，正确做法是把整个生命周期放在这个 task 里。
                from contextlib import AsyncExitStack

                stack = AsyncExitStack()
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                result = await session.list_tools()

                tools = []
                tool_index = {}
                for tool in result.tools:
                    tname = tool.name
                    tools.append({
                        "name": tname,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {
                            "type": "object",
                            "properties": {},
                        },
                    })
                    tool_index[tname] = server_name

                self._sessions[server_name] = session
                self._session_meta[server_name] = {
                    "stack": stack,
                    "sessions": [session],
                }
                self._tools.extend(tools)
                self._tool_index.update(tool_index)
                logger.info(
                    "MCP Client 已连接 '%s': %d 个工具（累计 %d）",
                    server_name,
                    len(tools),
                    len(self._tools),
                )

            self._submit_coro(_setup())
            return True

        except ImportError:
            logger.warning("mcp SDK 未安装，MCP Client 不可用")
            return False
        except Exception as e:
            logger.error("MCP Client 连接失败 '%s': %s", server_name, e)
            return False

    # ── 工具查询 ────────────────────────────────────────

    def get_tools_as_ollama_format(self) -> list[dict]:
        """将 MCP tools 转为 Ollama tool calling 格式。"""
        result = []
        for tool in self._tools:
            params = tool.get("parameters", {})
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

    # ── 工具调用 ────────────────────────────────────────

    def call_tool(self, tool_name: str, args: dict) -> str:
        """同步接口：调用 MCP 工具并返回结果字符串。"""
        if tool_name not in self._tool_index:
            return f"未找到 MCP 工具 '{tool_name}'"
        server_name = self._tool_index[tool_name]
        session = self._sessions.get(server_name)
        if session is None:
            return f"MCP 工具 '{tool_name}' 未连接"

        async def _call():
            try:
                result = await session.call_tool(tool_name, args)
                if result.content:
                    return result.content[0].text
                return str(result)
            except Exception as e:
                return f"MCP 工具调用失败: {e}"

        try:
            return self._submit_coro(_call())
        except Exception as e:
            logger.error("MCP call_tool 投递失败: %s", e)
            return f"MCP 工具调用失败: {e}"