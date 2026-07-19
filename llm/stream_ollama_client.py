# -*- coding: utf-8 -*-
"""Ollama 流式客户端，支持 tool calling 和流式文本输出。"""

import json
import logging
from typing import Callable, Generator, Optional

import requests

logger = logging.getLogger(__name__)

_SENTENCE_DELIMITERS = "。！？,.!?\n"


class StreamOllamaClient:
    """Ollama 流式 + tool calling 客户端。"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b",
        temperature: float = 0.7,
        num_predict: int = 2048,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                names = [m.get("name", "") for m in models]
                for name in names:
                    if self.model in name or name in self.model:
                        return True
                logger.warning("Ollama 在线但模型 '%s' 未找到", self.model)
                return False
            return False
        except Exception as e:
            logger.warning("Ollama 检查失败: %s", e)
            return False

    def warm_up(self):
        """预热模型，减少首次请求延迟。"""
        logger.info("预热模型: %s ...", self.model)
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "你好"}],
                "stream": False,
                "options": {"num_predict": 1},
            }
            requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=180
            )
            logger.info("模型 %s 预热完成", self.model)
        except Exception as e:
            logger.warning("模型预热失败: %s", e)

    def stream_chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        on_text_chunk: Optional[Callable[[str], None]] = None,
        on_sentence: Optional[Callable[[str], None]] = None,
    ) -> Generator[dict, None, None]:
        """
        流式对话生成器。

        Args:
            messages: 完整消息历史。
            tools: Ollama tool 定义列表。
            on_text_chunk: 每个文本 token 的回调。
            on_sentence: 每个完整句子的回调。

        Yields:
            {'type': 'text_chunk', 'content': str}
            {'type': 'sentence', 'content': str}
            {'type': 'tool_calls', 'calls': [{'name':..., 'args':{}}]}
            {'type': 'done', 'content': str}
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        full_reply = ""
        sentence_buffer = ""
        accumulated_tool_calls: list[dict] = []

        try:
            with requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue

                    msg = chunk.get("message", {})

                    # 处理文本增量
                    if "content" in msg and msg["content"]:
                        delta = msg["content"]
                        full_reply += delta
                        sentence_buffer += delta

                        if on_text_chunk:
                            on_text_chunk(delta)

                        yield {"type": "text_chunk", "content": delta}

                        # 句子切分
                        while True:
                            split_pos = -1
                            for d in _SENTENCE_DELIMITERS:
                                pos = sentence_buffer.find(d)
                                if pos != -1 and (split_pos == -1 or pos < split_pos):
                                    split_pos = pos

                            if split_pos != -1:
                                sentence = sentence_buffer[: split_pos + 1].strip()
                                sentence_buffer = sentence_buffer[split_pos + 1 :]
                                if sentence:
                                    if on_sentence:
                                        on_sentence(sentence)
                                    yield {"type": "sentence", "content": sentence}
                            else:
                                break

                    # 处理 tool_calls（Ollama 流式模式下，tool_calls 可能跨多行累积）
                    if "tool_calls" in msg and msg["tool_calls"]:
                        for tc in msg["tool_calls"]:
                            fn = tc.get("function", {})
                            accumulated_tool_calls.append({
                                "name": fn.get("name", ""),
                                "args": fn.get("arguments", {}),
                            })

                    if chunk.get("done"):
                        # 处理剩余缓冲区
                        if sentence_buffer.strip():
                            sentence = sentence_buffer.strip()
                            full_reply = full_reply.strip()
                            if on_sentence:
                                on_sentence(sentence)
                            yield {"type": "sentence", "content": sentence}

                        # 如果有 tool_calls，先返回
                        if accumulated_tool_calls:
                            yield {
                                "type": "tool_calls",
                                "calls": accumulated_tool_calls,
                            }
                        else:
                            yield {"type": "done", "content": full_reply}
                        return

        except requests.exceptions.RequestException as e:
            logger.error("Ollama 请求失败: %s", e)
            yield {"type": "error", "content": str(e)}

    def build_ollama_tool(
        self, name: str, description: str, parameters: dict
    ) -> dict:
        """将内部 tool schema 转为 Ollama API 格式。"""
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
