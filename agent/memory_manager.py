# -*- coding: utf-8 -*-
"""双层记忆管理 — 短期记忆（滑动窗口）+ 长期记忆（ChromaDB 检索+摘要）。"""

import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """双层记忆管理器。"""

    def __init__(
        self,
        chroma_store,
        short_term_max_turns: int = 10,
        long_term_enabled: bool = True,
    ):
        self._chroma = chroma_store
        self._short_term_max_turns = short_term_max_turns
        self._long_term_enabled = long_term_enabled

        # 短期记忆: 消息列表 [{"role": ..., "content": ...}, ...]
        self._messages: list[dict] = []
        self._system_prompt: str = ""

        # 本次对话期间都聊了什么，用于结束后生成摘要
        self._current_session_topics: list[str] = []

    def set_system_prompt(self, prompt: str):
        self._system_prompt = prompt
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = prompt
        else:
            self._messages.insert(0, {"role": "system", "content": prompt})

    def add_user_message(self, text: str):
        self._messages.append({"role": "user", "content": text})
        self._current_session_topics.append(text)
        self._trim_short_term()

    def add_assistant_message(self, text: str):
        self._messages.append({"role": "assistant", "content": text})

    def add_tool_message(self, tool_name: str, result: str):
        self._messages.append({
            "role": "tool",
            "content": result,
            "name": tool_name,
        })

    def get_messages(self) -> list[dict]:
        """返回当前完整的消息列表（含 system prompt 和短期记忆）。"""
        return list(self._messages)

    def build_context_messages(
        self, user_text: str, long_term_memories: Optional[list[str]] = None
    ) -> list[dict]:
        """
        构建发送给 LLM 的完整消息列表：
        system(含长期记忆) + 短期消息 + 当前用户输入。

        注意：当前用户输入不在这里追加到短期记忆，
        由 orchestrator 决定在对话成功后再追加。
        """
        messages = []

        # System prompt（含长期记忆）
        system_parts = [self._system_prompt] if self._system_prompt else []
        if long_term_memories:
            system_parts.append("以下是与此话题相关的历史对话记忆：")
            for i, mem in enumerate(long_term_memories, 1):
                system_parts.append(f"- {mem}")

        if system_parts:
            messages.append({"role": "system", "content": "\n".join(system_parts)})

        # 短期记忆
        for msg in self._messages:
            if msg.get("role") != "system":
                messages.append(msg)

        # 当前用户输入
        messages.append({"role": "user", "content": user_text})

        return messages

    def search_long_term(self, query: str, top_k: int = 3) -> list[str]:
        """检索长期记忆。"""
        if not self._long_term_enabled:
            return []
        results = self._chroma.search(query, top_k=top_k)
        return [r["text"] for r in results]

    def generate_and_store_summary(
        self, llm_client, user_text: str, assistant_reply: str
    ):
        """对话结束后生成摘要并存入长期记忆。"""
        if not self._long_term_enabled:
            return

        conversation = f"用户: {user_text}\n助手: {assistant_reply}"
        try:
            summary_prompt = [
                {
                    "role": "system",
                    "content": "请将以下对话压缩为一句简洁摘要（不超过50字），用中文输出。",
                },
                {"role": "user", "content": conversation},
            ]
            summary = ""
            for event in llm_client.stream_chat(
                summary_prompt,
                on_text_chunk=lambda c: None,
            ):
                if event["type"] in ("done", "sentence"):
                    summary = event.get("content", summary)
                    break
                elif event["type"] == "text_chunk":
                    summary += event["content"]

            if summary.strip():
                self._chroma.store(
                    text=summary.strip(),
                    metadata={
                        "user_text": user_text[:200],
                        "assistant_reply": assistant_reply[:200],
                    },
                )
                logger.info("长期记忆已存储: %s", summary[:50])
        except Exception as e:
            logger.warning("生成记忆摘要失败: %s", e)

    def _trim_short_term(self):
        """保持短期记忆在滑动窗口内。"""
        # system 不计入轮数，每轮 = user + assistant
        non_system = [m for m in self._messages if m.get("role") != "system"]
        max_messages = self._short_term_max_turns * 2  # N 轮 * 2 条消息
        if len(non_system) > max_messages:
            excess = len(non_system) - max_messages
            system_msg = (
                [self._messages[0]]
                if self._messages and self._messages[0].get("role") == "system"
                else []
            )
            self._messages = system_msg + non_system[excess:]

    def reset_session(self):
        """重置对话会话。"""
        system_msg = (
            [self._messages[0]]
            if self._messages and self._messages[0].get("role") == "system"
            else []
        )
        self._messages = system_msg
        self._current_session_topics = []
