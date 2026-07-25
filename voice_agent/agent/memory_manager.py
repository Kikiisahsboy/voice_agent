# -*- coding: utf-8 -*-
"""双层记忆管理 + 分层长期记忆（事实/偏好/事件）+ 用户画像。

短期：滑动窗口内的对话消息。
长期：ChromaDB 三种类型——
    - fact：客观事实（"用户住在上海"、"用户对青霉素过敏"）
    - preference：用户偏好（"喜欢简洁回答"、"不喜欢被叫昵称"）
    - event：对话事件摘要（"讨论了天气查询"）
用户画像：从对话中持续抽取，注入 system prompt。
"""

import json
import logging
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)

_VALID_MEMORY_TYPES = {"fact", "preference", "event"}

# ── 质量门控配置 ──
_MIN_USEFUL_LEN = 3  # 短于此长度的内容拒绝入库
_REJECT_CHITCHAT = {
    # 寒暄/客套/无关指令 — 直接丢弃，不进入向量库
    "你好", "您好", "再见", "bye", "hi", "hello",
    "哈哈", "呵呵", "嗯嗯", "好的", "收到",
    "测试一下", "测试", "试试",
}
_DEDUP_SCORE_THRESHOLD = 0.95  # 与已有记忆相似度超过此值 → 跳过


class MemoryManager:
    """双层 + 分类 + 画像记忆管理器。"""

    def __init__(
        self,
        chroma_store,
        short_term_max_turns: int = 10,
        long_term_enabled: bool = True,
    ):
        self._chroma = chroma_store
        self._short_term_max_turns = short_term_max_turns
        self._long_term_enabled = long_term_enabled

        self._messages: list[dict] = []
        self._system_prompt: str = ""

        # 用户画像：键值对，例 {"姓名": "小明", "职业": "学生"}
        self._user_profile: dict[str, str] = {}

        # 当前会话的最近 N 个事件，用于异步总结
        self._recent_turns: deque[dict] = deque(maxlen=10)

    # ── 系统提示 ────────────────────────────────────────

    def set_system_prompt(self, prompt: str):
        self._system_prompt = prompt
        if self._messages and self._messages[0].get("role") == "system":
            self._messages[0]["content"] = prompt
        else:
            self._messages.insert(0, {"role": "system", "content": prompt})

    # ── 短期记忆 ────────────────────────────────────────

    def add_user_message(self, text: str):
        self._messages.append({"role": "user", "content": text})
        self._recent_turns.append({"role": "user", "content": text})
        self._trim_short_term()

    def add_assistant_message(self, text: str):
        self._messages.append({"role": "assistant", "content": text})
        self._recent_turns.append({"role": "assistant", "content": text})
        self._trim_short_term()

    def add_tool_message(self, tool_name: str, result: str):
        self._messages.append({
            "role": "tool",
            "content": result,
            "name": tool_name,
        })

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def _trim_short_term(self):
        non_system = [m for m in self._messages if m.get("role") != "system"]
        max_messages = self._short_term_max_turns * 2
        if len(non_system) > max_messages:
            excess = len(non_system) - max_messages
            system_msg = (
                [self._messages[0]]
                if self._messages and self._messages[0].get("role") == "system"
                else []
            )
            self._messages = system_msg + non_system[excess:]

    def reset_session(self):
        system_msg = (
            [self._messages[0]]
            if self._messages and self._messages[0].get("role") == "system"
            else []
        )
        self._messages = system_msg
        self._recent_turns.clear()

    # ── 用户画像 ────────────────────────────────────────

    def update_user_profile(self, key: str, value: str):
        """更新用户画像（key 不存在则新增）。"""
        self._user_profile[key] = value

    def get_user_profile(self) -> dict[str, str]:
        return dict(self._user_profile)

    def _profile_to_text(self) -> str:
        if not self._user_profile:
            return ""
        lines = [f"- {k}: {v}" for k, v in self._user_profile.items()]
        return "用户画像：\n" + "\n".join(lines)

    # ── 长期记忆公共 API（取代之前 MCP Server 直接访问 _chroma）─────

    def store_long_term(
        self,
        text: str,
        memory_type: str = "event",
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """存储一条长期记忆（带质量门控）。

        拒绝：① 长度过短 ② 命中寒暄黑名单 ③ 与已有记忆高度相似。

        Args:
            text: 记忆内容
            memory_type: fact / preference / event
            metadata: 额外元数据

        Returns:
            新写入的 memory_id；若被门控拒绝则返回 None。
        """
        if not self._long_term_enabled:
            return None

        text = (text or "").strip()
        if not text:
            return None

        # ① 长度过滤
        if len(text) < _MIN_USEFUL_LEN:
            logger.debug("记忆过短被丢弃: '%s'", text)
            return None

        # ② 黑名单过滤（仅对非 event 类目生效 — event 可记录具体对话事件）
        if memory_type in ("fact", "preference"):
            for bad in _REJECT_CHITCHAT:
                if bad in text:
                    logger.debug("寒暄/无关内容被丢弃: '%s'", text)
                    return None

        if memory_type not in _VALID_MEMORY_TYPES:
            logger.warning("未知记忆类型 '%s'，回退为 event", memory_type)
            memory_type = "event"

        # ③ 去重：与已有最高相似度对比
        try:
            existing = self._chroma.search(text, top_k=1, threshold=0.0)
            if existing and existing[0].get("score", 0) >= _DEDUP_SCORE_THRESHOLD:
                logger.debug("重复记忆跳过: '%s' ≈ '%s'",
                             text, existing[0].get("text", "")[:30])
                return None
        except Exception as e:
            logger.debug("去重检查失败（继续写入）: %s", e)

        meta = dict(metadata or {})
        meta["memory_type"] = memory_type
        meta.setdefault("timestamp", time.time())
        meta.setdefault("last_accessed", meta["timestamp"])
        return self._chroma.store(text=text, metadata=meta)

    def search_long_term(
        self,
        query: str,
        top_k: int = 3,
        memory_types: Optional[list[str]] = None,
    ) -> list[str]:
        """检索长期记忆。可限定类型。"""
        if not self._long_term_enabled:
            return []
        results = self._chroma.search(query, top_k=top_k * 2 if memory_types else top_k)
        if memory_types:
            results = [r for r in results if r.get("metadata", {}).get("memory_type") in memory_types]
        results = results[:top_k]
        return [r["text"] for r in results]

    def search_long_term_ranked(
        self,
        query: str,
        top_k: int = 3,
        memory_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """返回带分数和 metadata 的检索结果，供 Rerank 使用。"""
        if not self._long_term_enabled:
            return []
        results = self._chroma.search(query, top_k=top_k * 3 if memory_types else top_k)
        if memory_types:
            results = [r for r in results if r.get("metadata", {}).get("memory_type") in memory_types]
        return results[:top_k]

    # ── 上下文组装 ──────────────────────────────────────

    def build_context_messages(
        self,
        user_text: str,
        long_term_memories: Optional[list[str]] = None,
        long_term_ranked: Optional[list[dict]] = None,
    ) -> list[dict]:
        """构建发送给 LLM 的消息列表。

        顺序：
            system（基础提示 + 用户画像 + 长期记忆）
            短期消息
            当前用户输入
        """
        messages = []
        system_parts = []

        if self._system_prompt:
            system_parts.append(self._system_prompt)

        profile_text = self._profile_to_text()
        if profile_text:
            system_parts.append(profile_text)

        if long_term_ranked:
            system_parts.append("以下是与此话题相关的历史对话记忆（按相关度排序）：")
            for i, mem in enumerate(long_term_ranked, 1):
                score = mem.get("score", 0)
                mtype = mem.get("metadata", {}).get("memory_type", "event")
                system_parts.append(f"- [{mtype}|score={score:.2f}] {mem['text']}")
        elif long_term_memories:
            system_parts.append("以下是与此话题相关的历史对话记忆：")
            for i, mem in enumerate(long_term_memories, 1):
                system_parts.append(f"- {mem}")

        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        for msg in self._messages:
            if msg.get("role") != "system":
                messages.append(msg)

        messages.append({"role": "user", "content": user_text})
        return messages

    # ── 异步总结（对话结束后）───────────────────────────

    def generate_and_store_summary(
        self, llm_client, user_text: str, assistant_reply: str
    ):
        """生成结构化摘要并存入长期记忆。

        通过 prompt 让 LLM 输出 JSON：
            {
                "summary": "...",
                "facts": ["用户住在上海", ...],
                "preferences": ["喜欢简洁回答", ...],
                "profile_updates": {"姓名": "小明"}
            }
        """
        if not self._long_term_enabled:
            return

        conversation = f"用户: {user_text}\n助手: {assistant_reply}"
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是信息抽取助手。请从对话中抽取结构化信息。"
                    "严格只输出 JSON：\n"
                    '{"summary": "对话的一句话摘要（≤50字）",'
                    '"facts": ["事实1", ...],'
                    '"preferences": ["用户偏好1", ...],'
                    '"profile_updates": {"key": "value", ...}}\n\n'
                    "质量规则（务必遵守）：\n"
                    "1. 只抽取对未来对话**仍有复用价值**的信息。\n"
                    "2. 以下内容**不要**记入（任何字段都不要写）：\n"
                    "   - 寒暄/问候/客套（你好、再见、测试一下……）\n"
                    "   - 单次性事务（帮我算 3+5、查一下今天天气……）\n"
                    "   - 工具调用原始结果、LLM 自己的输出错误\n"
                    "   - 用户已表达别记/忘掉/清除记忆的内容\n"
                    "3. 不确定的事实宁可漏掉，不要写错的。\n"
                    "4. 如果整段对话都没有值得记的内容，summary 返回空串 \"\" 即可。"
                ),
            },
            {"role": "user", "content": conversation},
        ]

        try:
            raw = ""
            for event in llm_client.stream_chat(
                prompt, on_text_chunk=lambda c: None
            ):
                if event["type"] == "text_chunk":
                    raw += event["content"]
                elif event["type"] == "done":
                    raw = event.get("content", raw)
                    break

            parsed = self._safe_parse_json(raw)
            if not parsed:
                # 退化：保存为 event
                self.store_long_term(
                    raw.strip()[:200], memory_type="event",
                    metadata={"user_text": user_text[:200]},
                )
                return

            if parsed.get("summary"):
                self.store_long_term(
                    parsed["summary"], memory_type="event",
                    metadata={"user_text": user_text[:200]},
                )
            for fact in parsed.get("facts", []):
                self.store_long_term(fact, memory_type="fact")
            for pref in parsed.get("preferences", []):
                self.store_long_term(pref, memory_type="preference")
            for k, v in parsed.get("profile_updates", {}).items():
                self.update_user_profile(k, v)

            logger.info(
                "记忆抽取: 摘要=%s, 事实=%d, 偏好=%d, 画像=%d",
                bool(parsed.get("summary")),
                len(parsed.get("facts", [])),
                len(parsed.get("preferences", [])),
                len(parsed.get("profile_updates", {})),
            )
        except Exception as e:
            logger.warning("生成记忆摘要失败: %s", e)

    @staticmethod
    def _safe_parse_json(raw: str) -> Optional[dict]:
        """尝试从 LLM 输出中提取 JSON。"""
        if not raw:
            return None
        raw = raw.strip()
        # 直接尝试
        try:
            return json.loads(raw)
        except Exception:
            pass
        # 尝试抽取 ```json ... ```
        if "```" in raw:
            try:
                start = raw.find("```")
                end = raw.rfind("```")
                inner = raw[start:end]
                inner = inner.replace("json", "", 1).strip()
                return json.loads(inner)
            except Exception:
                pass
        # 尝试抽取第一个 { 到最后一个 }
        try:
            i = raw.find("{")
            j = raw.rfind("}")
            if i != -1 and j != -1 and j > i:
                return json.loads(raw[i : j + 1])
        except Exception:
            pass
        return None