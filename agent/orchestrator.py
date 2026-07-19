# -*- coding: utf-8 -*-
"""Agent 核心编排器 — ReAct 循环协调 ASR → Memory → LLM → TTS 全流程。"""

import json
import logging
from typing import Callable, Generator, Optional

from voice_agent.agent.memory_manager import MemoryManager
from voice_agent.agent.skill_manager import SkillManager
from voice_agent.agent.skills.memory_skill import _set_memory_manager
from voice_agent.asr.stream_asr import StreamASR
from voice_agent.llm.stream_ollama_client import StreamOllamaClient
from voice_agent.tts.stream_tts import StreamTTS

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Agent 核心编排器。"""

    def __init__(
        self,
        llm_client: StreamOllamaClient,
        asr_service: StreamASR,
        tts_service: StreamTTS,
        memory_manager: MemoryManager,
        skill_manager: SkillManager,
        mcp_client=None,
        max_react_rounds: int = 5,
    ):
        self._llm = llm_client
        self._asr = asr_service
        self._tts = tts_service
        self._memory = memory_manager
        self._skills = skill_manager
        self._mcp_client = mcp_client
        self._max_react_rounds = max_react_rounds

        # 让 memory_skill 能访问 memory_manager
        _set_memory_manager(memory_manager)

    # ── 文本对话 ──────────────────────────────────────

    def process_text_input(
        self,
        user_text: str,
        on_llm_chunk: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
    ) -> Generator[dict, None, None]:
        """
        处理文本输入，执行 ReAct 循环。

        Yields:
            {'type': 'status', 'content': str}
            {'type': 'llm_chunk', 'content': str}
            {'type': 'sentence', 'content': str}
            {'type': 'tool_call', 'name': str, 'args': dict}
            {'type': 'tool_result', 'name': str, 'result': str}
            {'type': 'done', 'content': str}
            {'type': 'error', 'content': str}
        """
        yield {"type": "status", "content": "thinking"}

        # 检索长期记忆
        long_term_memories = None
        try:
            long_term_memories = self._memory.search_long_term(user_text)
            if long_term_memories:
                logger.info("检索到 %d 条长期记忆: %s", len(long_term_memories), [m[:30] for m in long_term_memories])
            else:
                logger.info("未检索到相关长期记忆 (query: %s)", user_text[:50])
        except Exception as e:
            logger.warning("长期记忆检索失败: %s", e)

        # 构建初始消息
        messages = self._memory.build_context_messages(
            user_text, long_term_memories
        )

        # 获取所有可用工具
        tools = self._get_all_tools()

        # ReAct 循环
        full_reply = ""
        for round_idx in range(self._max_react_rounds):
            tool_called = False
            round_reply = ""

            for event in self._llm.stream_chat(
                messages,
                tools=tools if tools else None,
                on_text_chunk=on_llm_chunk,
                on_sentence=on_sentence,
            ):
                if event["type"] == "error":
                    yield event
                    return

                if event["type"] == "text_chunk":
                    round_reply += event["content"]
                    yield event

                elif event["type"] == "sentence":
                    full_reply += event["content"]
                    yield event

                elif event["type"] == "tool_calls":
                    tool_called = True
                    # 将之前的文本回复追加到消息列表
                    if round_reply.strip():
                        messages.append({
                            "role": "assistant",
                            "content": round_reply,
                        })
                        full_reply += round_reply

                    for tc in event["calls"]:
                        tool_name = tc["name"]
                        tool_args = tc.get("args", {})

                        if on_tool_call:
                            on_tool_call(tool_name, tool_args)

                        yield {
                            "type": "tool_call",
                            "name": tool_name,
                            "args": tool_args,
                        }

                        # 执行工具
                        result = self._execute_tool(tool_name, tool_args)
                        yield {
                            "type": "tool_result",
                            "name": tool_name,
                            "result": result,
                        }

                        # 将工具调用和结果追加到消息
                        messages.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "function": {
                                    "name": tool_name,
                                    "arguments": (
                                        tool_args if isinstance(tool_args, str)
                                        else json.dumps(tool_args, ensure_ascii=False)
                                    ),
                                }
                            }],
                        })
                        messages.append({
                            "role": "tool",
                            "content": result,
                            "name": tool_name,
                        })

                    break  # 跳出 stream_chat 循环，重新发送含工具结果的消息

                elif event["type"] == "done":
                    full_reply += event.get("content", "")
                    # 追加到记忆
                    self._memory.add_user_message(user_text)
                    final_text = full_reply or event.get("content", "")
                    self._memory.add_assistant_message(final_text)

                    # 异步生成长期记忆摘要
                    try:
                        self._memory.generate_and_store_summary(
                            self._llm, user_text, final_text
                        )
                    except Exception as e:
                        logger.warning("生成摘要失败: %s", e)

                    yield {"type": "done", "content": final_text}
                    return

            if not tool_called:
                break

        # 兜底：到了最大轮次仍有输出
        if full_reply:
            self._memory.add_user_message(user_text)
            self._memory.add_assistant_message(full_reply)
            yield {"type": "done", "content": full_reply}
        else:
            error_msg = "抱歉，我暂时无法回答这个问题。"
            self._memory.add_user_message(user_text)
            self._memory.add_assistant_message(error_msg)
            yield {"type": "error", "content": error_msg}

    # ── 语音对话 ──────────────────────────────────────

    def process_voice_input(
        self,
        audio_chunks: list[bytes],
        on_partial_text: Optional[Callable[[str], None]] = None,
        on_llm_chunk: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
    ) -> Generator[dict, None, None]:
        """
        处理语音输入：ASR → LLM → TTS。

        Args:
            audio_chunks: PCM 音频数据块列表
        """
        yield {"type": "status", "content": "listening"}

        # ASR 阶段
        self._asr.reset()
        recognized_text = ""
        for chunk in audio_chunks:
            result = self._asr.feed(chunk)
            if result:
                if result["type"] == "partial" and on_partial_text:
                    on_partial_text(result["text"])
                    yield {"type": "asr_partial", "content": result["text"]}
                elif result["type"] == "final":
                    recognized_text = result["text"]

        # 获取最终结果
        final_result = self._asr.finalize()
        if final_result and final_result.get("type") == "final":
            recognized_text = final_result["text"]

        if not recognized_text:
            yield {"type": "error", "content": "未能识别语音内容"}
            return

        yield {"type": "asr_final", "content": recognized_text}

        # LLM 阶段（复用 process_text_input）
        yield from self.process_text_input(
            recognized_text,
            on_llm_chunk=on_llm_chunk,
            on_tool_call=on_tool_call,
            on_sentence=on_sentence,
        )

    # ── 内部方法 ──────────────────────────────────────

    def _get_all_tools(self) -> list[dict]:
        """合并 SkillManager 和 MCP Client 的工具列表。"""
        tools = self._skills.get_all_tools()
        if self._mcp_client:
            try:
                mcp_tools = self._mcp_client.get_tools_as_ollama_format()
                tools.extend(mcp_tools)
            except Exception as e:
                logger.warning("获取 MCP 工具失败: %s", e)
        return tools

    def _execute_tool(self, name: str, args: dict) -> str:
        """执行工具：先在 SkillManager 找，再在 MCP Client 找。"""
        # 先在 SkillManager 中查找
        if name in self._skills.get_skill_names():
            return self._skills.execute(name, args)

        # 再在 MCP Client 中查找
        if self._mcp_client:
            try:
                return self._mcp_client.call_tool(name, args)
            except Exception as e:
                logger.warning("MCP 工具 '%s' 执行失败: %s", name, e)
                return f"MCP 工具执行失败: {e}"

        return f"未找到工具 '{name}'"

    def reset_conversation(self):
        self._memory.reset_session()
