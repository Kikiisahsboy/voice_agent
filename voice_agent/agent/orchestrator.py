# -*- coding: utf-8 -*-
"""Agent 核心编排器 — Plan-and-Execute + ReAct + Reflection。

流程：
    1. 检索长期记忆（含类型过滤）
    2. （可选）生成执行计划
    3. 进入 ReAct 循环：LLM 推理 → 工具调用 → 反馈 → 再推理
    4. 工具调用失败时自动重试（最多 2 次）
    5. 反思阶段：检查是否完成用户目标，未完成则补一轮
    6. 完成后抽取结构化记忆
"""

import json
import logging
from typing import Callable, Generator, Optional

from voice_agent.agent.memory_manager import MemoryManager
from voice_agent.agent.skill_manager import SkillManager
from voice_agent.asr.stream_asr import StreamASR
from voice_agent.llm.stream_ollama_client import StreamOllamaClient
from voice_agent.tts.stream_tts import StreamTTS

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Agent 核心编排器（Plan-and-Execute + Reflection）。"""

    def __init__(
        self,
        llm_client: StreamOllamaClient,
        asr_service: StreamASR,
        tts_service: StreamTTS,
        memory_manager: MemoryManager,
        skill_manager: SkillManager,
        mcp_client=None,
        max_react_rounds: int = 5,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        tool_retry_limit: int = 2,
        document_rag=None,
    ):
        self._llm = llm_client
        self._asr = asr_service
        self._tts = tts_service
        self._memory = memory_manager
        self._skills = skill_manager
        self._mcp_client = mcp_client
        self._max_react_rounds = max_react_rounds
        self._enable_planning = enable_planning
        self._enable_reflection = enable_reflection
        self._tool_retry_limit = tool_retry_limit
        self._document_rag = document_rag

    # ── 入口：文本 ──────────────────────────────────────

    def process_text_input(
        self,
        user_text: str,
        on_llm_chunk: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
    ) -> Generator[dict, None, None]:
        """处理文本输入。

        Yields:
            status / plan / llm_chunk / sentence / tool_call /
            tool_result / reflection / done / error
        """
        yield {"type": "status", "content": "thinking"}

        # 1. 检索长期记忆（事实 + 偏好 + 事件）
        long_term_ranked: list[dict] = []
        try:
            long_term_ranked = self._memory.search_long_term_ranked(
                user_text, top_k=3
            )
            if long_term_ranked:
                logger.info(
                    "检索到 %d 条长期记忆 (top: %s)",
                    len(long_term_ranked),
                    long_term_ranked[0]["text"][:30],
                )
        except Exception as e:
            logger.warning("长期记忆检索失败: %s", e)

        # 2. 构建上下文
        messages = self._memory.build_context_messages(
            user_text, long_term_ranked=long_term_ranked
        )

        tools = self._get_all_tools()

        # 3. （可选）Plan
        plan_text = ""
        if self._enable_planning:
            try:
                plan_text = self._make_plan(user_text, messages)
                if plan_text:
                    yield {"type": "plan", "content": plan_text}
                    messages.append({
                        "role": "system",
                        "content": f"执行计划（请按步骤推进）：\n{plan_text}",
                    })
            except Exception as e:
                logger.warning("计划生成失败: %s", e)

        # 4. ReAct 循环
        full_reply = ""
        final_text = ""
        reflection_used = False

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
                    full_reply += event["content"]
                    yield event

                elif event["type"] == "sentence":
                    yield event

                elif event["type"] == "tool_calls":
                    tool_called = True
                    if round_reply.strip():
                        messages.append({
                            "role": "assistant",
                            "content": round_reply,
                        })

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

                        result = self._call_tool_with_retry(
                            tool_name, tool_args, messages
                        )

                        yield {
                            "type": "tool_result",
                            "name": tool_name,
                            "result": result,
                        }

                    break

                elif event["type"] == "done":
                    final_text = event.get("content", full_reply).strip()
                    break

            if not tool_called:
                break

            # Reflection：检查是否已回答完成
            if (
                self._enable_reflection
                and not reflection_used
                and round_idx >= 1
            ):
                reflection_used = True
                if not self._is_task_complete(user_text, full_reply, plan_text):
                    messages.append({
                        "role": "system",
                        "content": (
                            "反思：上面的回答似乎还没完全满足用户需求。"
                            "请检查是否还需要调用其他工具或补充信息。"
                        ),
                    })
                    yield {"type": "reflection", "content": "incomplete"}
                    continue
                yield {"type": "reflection", "content": "complete"}

        # 5. 兜底
        if not final_text:
            final_text = full_reply.strip()
        if not final_text:
            final_text = "抱歉，我暂时无法回答这个问题。"

        # 6. 持久化到短期记忆
        self._memory.add_user_message(user_text)
        self._memory.add_assistant_message(final_text)

        # 7. 异步抽取长期记忆
        try:
            self._memory.generate_and_store_summary(
                self._llm, user_text, final_text
            )
        except Exception as e:
            logger.warning("生成摘要失败: %s", e)

        yield {"type": "done", "content": final_text}

    # ── 入口：语音 ──────────────────────────────────────

    def process_voice_input(
        self,
        audio_chunks: list[bytes],
        on_partial_text: Optional[Callable[[str], None]] = None,
        on_llm_chunk: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, dict], None]] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
    ) -> Generator[dict, None, None]:
        """处理语音输入：ASR → 复用 process_text_input。"""
        yield {"type": "status", "content": "listening"}

        self._asr.reset()
        recognized_text = ""
        for chunk in audio_chunks:
            result = self._asr.feed(chunk)
            if result:
                if result["type"] == "partial":
                    if on_partial_text:
                        on_partial_text(result["text"])
                    yield {"type": "asr_partial", "content": result["text"]}
                elif result["type"] == "final":
                    recognized_text = result["text"]

        final_result = self._asr.finalize()
        if final_result and final_result.get("type") == "final":
            recognized_text = final_result["text"]

        if not recognized_text:
            yield {"type": "error", "content": "未能识别语音内容"}
            return

        yield {"type": "asr_final", "content": recognized_text}

        yield from self.process_text_input(
            recognized_text,
            on_llm_chunk=on_llm_chunk,
            on_tool_call=on_tool_call,
            on_sentence=on_sentence,
        )

    # ── 内部：Planning ──────────────────────────────────

    def _make_plan(self, user_text: str, base_messages: list[dict]) -> str:
        """调用 LLM 生成简短执行计划。"""
        plan_prompt = [
            {
                "role": "system",
                "content": (
                    "你是任务规划助手。请根据用户输入生成简洁的执行计划，"
                    "输出 3-5 步以内的编号列表。如果无需规划（如闲聊）输出 'NO_PLAN'。"
                ),
            },
            {"role": "user", "content": user_text},
        ]
        plan_text = ""
        for event in self._llm.stream_chat(plan_prompt, on_text_chunk=lambda c: None):
            if event["type"] == "text_chunk":
                plan_text += event["content"]
            elif event["type"] == "done":
                plan_text = event.get("content", plan_text)
                break

        plan_text = plan_text.strip()
        if not plan_text or "NO_PLAN" in plan_text.upper():
            return ""
        return plan_text

    # ── 内部：Reflection ────────────────────────────────

    def _is_task_complete(
        self, user_text: str, current_reply: str, plan: str
    ) -> bool:
        """判断当前回复是否完成用户目标。"""
        if not current_reply.strip():
            return False
        if not plan:
            return True  # 无计划则不再反思

        reflect_prompt = [
            {
                "role": "system",
                "content": (
                    "判断助手是否已完整回答用户问题。"
                    "如果已满足用户需求输出 YES，否则输出 NO。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题: {user_text}\n"
                    f"执行计划: {plan}\n"
                    f"当前回复: {current_reply[:500]}\n"
                ),
            },
        ]
        verdict = ""
        for event in self._llm.stream_chat(reflect_prompt, on_text_chunk=lambda c: None):
            if event["type"] == "text_chunk":
                verdict += event["content"]
            elif event["type"] == "done":
                verdict = event.get("content", verdict)
                break
        return "YES" in verdict.upper()

    # ── 内部：工具调用（含重试）───────────────────────

    def _call_tool_with_retry(
        self, name: str, args: dict, messages: list[dict]
    ) -> str:
        """调用工具，失败时把错误塞回 prompt 让 LLM 重试。

        只有 SkillManager.execute 返回的"执行技能 ... 出错"才视为可重试错误，
        业务返回（如"未找到相关历史记忆"）视为成功结果。
        """
        last_error = None
        for attempt in range(self._tool_retry_limit + 1):
            try:
                result = self._execute_tool(name, args)
                is_tool_error = (
                    isinstance(result, str)
                    and result.startswith("执行技能")
                    and "出错" in result
                )

                # 把 tool 调用记录写进 messages（无论成功失败，下一轮 LLM 都要看到）
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "function": {
                            "name": name,
                            "arguments": (
                                args if isinstance(args, str)
                                else json.dumps(args, ensure_ascii=False)
                            ),
                        }
                    }],
                })

                if not is_tool_error:
                    # 成功（含业务"未找到"等正常返回）
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "name": name,
                    })
                    return result

                last_error = result
                logger.warning(
                    "工具 '%s' 第 %d 次调用失败: %s", name, attempt + 1, result
                )

                if attempt < self._tool_retry_limit:
                    messages.append({
                        "role": "tool",
                        "content": f"[错误] {result}\n请检查参数后重新调用。",
                        "name": name,
                    })
                    # 返回错误结果让外层继续下一轮（不 break）
                    return result
                # 已达重试上限，附加错误结果
                messages.append({
                    "role": "tool",
                    "content": result,
                    "name": name,
                })
            except Exception as e:
                last_error = f"异常: {e}"
                logger.error("工具 '%s' 抛出异常: %s", name, e)
                if attempt >= self._tool_retry_limit:
                    break

        return last_error or f"工具 '{name}' 调用失败"

    def _execute_tool(self, name: str, args: dict) -> str:
        """执行工具：先 SkillManager，再 MCP Client。"""
        if name in self._skills.get_skill_names():
            return self._skills.execute(
                name, args,
                context={"memory": self._memory, "document_rag": self._document_rag},
            )

        if self._mcp_client:
            try:
                return self._mcp_client.call_tool(name, args)
            except Exception as e:
                logger.warning("MCP 工具 '%s' 执行失败: %s", name, e)
                return f"MCP 工具执行失败: {e}"

        return f"未找到工具 '{name}'"

    def _get_all_tools(self) -> list[dict]:
        tools = self._skills.get_all_tools()
        if self._mcp_client:
            try:
                tools.extend(self._mcp_client.get_tools_as_ollama_format())
            except Exception as e:
                logger.warning("获取 MCP 工具失败: %s", e)
        return tools

    def reset_conversation(self):
        self._memory.reset_session()