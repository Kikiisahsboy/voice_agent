# -*- coding: utf-8 -*-
"""Voice Agent Golden Eval Runner。

跑通 golden_set.json 中的用例，输出 pass/fail 报告。

用法：
    python eval/run_eval.py
    python eval/run_eval.py --case calc_basic

设计：
- 不依赖 Ollama 真实推理（避免跑大模型卡时间）：使用 FakeLLMClient，
  通过观察到的工具调用判断 Agent 是否正确路由。
- 也可以用 --real-llm 强制使用真实 LLM（需要 Ollama 在线）。
"""

import argparse
import json
import os
import sys
import tempfile
from typing import Optional

# UTF-8 输出（Windows 默认 GBK 会让 ✓ ✗ 报错）
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG = os.path.join(_ROOT, "voice_agent")
for p in (_ROOT, _PKG):
    if p not in sys.path:
        sys.path.insert(0, p)


class FakeChromaStore:
    def __init__(self):
        self._docs = []
        self._next = 0

    def store(self, text, metadata=None, memory_id=None):
        self._next += 1
        mid = memory_id or f"mem_{self._next}"
        meta = dict(metadata or {})
        meta.setdefault("timestamp", 0.0)
        meta.setdefault("last_accessed", 0.0)
        self._docs.append({"id": mid, "text": text, "metadata": meta})
        return mid

    def search(self, query, top_k=3, threshold=0.0, where=None):
        out = []
        for d in self._docs:
            meta = d["metadata"]
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            kw = set(query)
            common = sum(1 for c in d["text"] if c in kw)
            if common == 0:
                continue
            score = min(1.0, common / max(len(kw), 1))
            if score >= threshold:
                out.append({"id": d["id"], "text": d["text"],
                            "score": score, "metadata": meta})
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]

    def get_recent(self, limit=10):
        return [d["text"] for d in self._docs[-limit:]]

    def decay(self, *a, **kw):
        pass

    def forget(self, mid):
        self._docs = [d for d in self._docs if d["id"] != mid]

    def count(self):
        return len(self._docs)


class FakeDocumentRAG:
    """极简 DocumentRAG — 用字符重叠做相似度。"""

    def __init__(self):
        self._docs = []

    def add_document(self, text: str, source: str = ""):
        self._docs.append({"text": text, "source": source})

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def search(self, query: str, top_k: int = 3, threshold: float = 0.05):
        kw = set(query)
        out = []
        for i, d in enumerate(self._docs):
            common = sum(1 for c in d["text"] if c in kw)
            if common == 0:
                continue
            score = min(1.0, common / max(len(kw), 1))
            if score >= threshold:
                out.append({
                    "id": str(i),
                    "text": d["text"],
                    "score": score,
                    "metadata": {"source": d["source"]},
                })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:top_k]

    def format_context(self, passages):
        return "\n".join(
            f"[{p['metadata'].get('source', '?')}] {p['text']}"
            for p in passages
        )


class FakeLLMClient:
    """根据工具描述和用户输入"假装"调用工具，再生成简短回答。

    不调真实 LLM。基于关键词匹配决定调用哪个工具。
    """

    def __init__(self):
        self.calls = []

    def stream_chat(self, messages, tools=None, on_text_chunk=None,
                    on_sentence=None):
        self.calls.append({"tools": tools, "messages": messages})

        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        # 优先检查：如果 messages 里已经有 tool 角色，说明上一轮已经调用过工具，
        # 第二轮应直接产出基于工具结果的"回答"，避免无限循环。
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if tool_results:
            last_tool = tool_results[-1]
            tool_name = last_tool.get("name", "tool")
            tool_content = last_tool.get("content", "")
            fake_reply = f"[fake:{tool_name}] {tool_content[:200]}"
            yield {"type": "text_chunk", "content": fake_reply}
            yield {"type": "done", "content": fake_reply}
            return

        # 从 system 消息抽取用户画像（FakeLLM 不理解中文语义，但能匹配 key）
        profile: dict[str, str] = {}
        for m in messages:
            if m.get("role") == "system":
                for line in (m.get("content", "")).split("\n"):
                    if line.startswith("- "):
                        parts = line[2:].split(":", 1)
                        if len(parts) == 2:
                            profile[parts[0].strip()] = parts[1].strip()

        # 询问画像关键词 → 直接基于 profile 答，不再调工具
        key_aliases = {
            "姓名": ["名字", "姓名", "叫什么"],
            "年龄": ["年龄", "多大", "几岁"],
            "职业": ["职业", "工作", "做什么"],
        }
        for k, v in profile.items():
            for alias in key_aliases.get(k, [k]):
                if alias in user_msg:
                    reply = f"[fake] 您的{k}是{v}。"
                    yield {"type": "text_chunk", "content": reply}
                    yield {"type": "done", "content": reply}
                    return

        tool_called = None
        if tools:
            for t in tools:
                fn = t.get("function", {})
                keywords_map = {
                    "calculate": ["算", "计算", "+", "*", "/", "等于"],
                    "get_current_time": ["几点", "时间", "日期", "几号"],
                    "get_weather": ["天气", "气温", "下雨"],
                    "search_memory": ["记得", "之前", "聊过", "回忆"],
                    "search_knowledge": ["政策", "文档", "资料", "公司"],
                    "web_search": ["搜索", "查一下", "最新"],
                }
                for kw in keywords_map.get(fn.get("name", ""), []):
                    if kw in user_msg:
                        tool_called = fn["name"]
                        break
                if tool_called:
                    break

        if tool_called:
            # 根据工具 schema 自动填必填参数，避免 false-positive 触发重试逻辑
            tool_schema = next(
                (t["function"] for t in tools if t["function"]["name"] == tool_called),
                {},
            )
            params = tool_schema.get("parameters", {}).get("properties", {})
            required = tool_schema.get("parameters", {}).get("required", [])
            args = {}
            for pname in required:
                if "query" in pname.lower() or "keyword" in pname.lower():
                    args[pname] = user_msg
                elif "expression" in pname.lower():
                    import re
                    m = re.search(r"[\d+\-*/().\s]+", user_msg)
                    args[pname] = m.group().strip() if m else user_msg
                elif "city" in pname.lower():
                    args[pname] = user_msg
                else:
                    args[pname] = user_msg
            yield {"type": "tool_calls", "calls": [{
                "name": tool_called,
                "args": args,
            }]}
            return

        # 检查是否处于"工具调用后第二轮"（含 tool role 消息）
        tool_results = [m for m in messages if m.get("role") == "tool"]
        if tool_results:
            last_tool = tool_results[-1]
            tool_name = last_tool.get("name", "tool")
            tool_content = last_tool.get("content", "")
            # 把工具结果原样拼成"回复"
            fake_reply = f"[fake:{tool_name}] {tool_content[:200]}"
            yield {"type": "text_chunk", "content": fake_reply}
            yield {"type": "done", "content": fake_reply}
            return

        yield {"type": "text_chunk", "content": f"[fake] 收到: {user_msg[:30]}"}
        yield {"type": "done", "content": f"[fake] 收到: {user_msg[:30]}"}

    def health_check(self):
        return True

    def warm_up(self):
        pass


def build_agent(chroma_store, document_rag=None):
    from voice_agent.agent.memory_manager import MemoryManager
    from voice_agent.agent.skill_manager import SkillManager
    from voice_agent.agent.orchestrator import AgentOrchestrator

    mm = MemoryManager(chroma_store)
    mm.set_system_prompt("你是测试助手。")
    sm = SkillManager()
    from voice_agent.agent.skills_autoloader import autoload_skills
    autoload_skills(sm, [
        "voice_agent.agent.skills.time_skill",
        "voice_agent.agent.skills.calculator_skill",
        "voice_agent.agent.skills.memory_skill",
        "voice_agent.agent.skills.knowledge_skill",
        "voice_agent.agent.skills.weather_skill",
    ])
    return mm, AgentOrchestrator(
        llm_client=FakeLLMClient(),
        asr_service=None, tts_service=None,
        memory_manager=mm, skill_manager=sm,
        mcp_client=None, max_react_rounds=3,
        enable_planning=False, enable_reflection=False,
        tool_retry_limit=0,
        document_rag=document_rag,
    )


def _drive(orch, user_text):
    """跑一次对话，收集 events。"""
    events = []
    for ev in orch.process_text_input(user_text):
        events.append(ev)
        if ev.get("type") in ("done", "error"):
            break
    return events


def _check_expectation(events, expect: dict) -> tuple[bool, str]:
    text = "".join(
        e.get("content", "") for e in events if e.get("type") == "text_chunk"
    )
    final = ""
    for e in reversed(events):
        if e.get("type") == "done":
            final = e.get("content", "")
            break
    full_text = text + final

    # orchestrator 实际 yield 的是 {"type": "tool_call", "name": ..., "args": ...}
    tools_called = []
    for e in events:
        if e.get("type") == "tool_call" and "name" in e:
            tools_called.append(e["name"])
        elif e.get("type") == "tool_calls":
            for c in e.get("calls", []):
                tools_called.append(c.get("name"))

    if "tool_called" in expect:
        if expect["tool_called"] not in tools_called:
            return False, f"期望调用工具 {expect['tool_called']}，实际 {tools_called}"

    if "tools_called_any" in expect:
        if not any(t in tools_called for t in expect["tools_called_any"]):
            return False, f"期望调用任一工具 {expect['tools_called_any']}，实际 {tools_called}"

    if "tools_called_count_max" in expect:
        if len(tools_called) > expect["tools_called_count_max"]:
            return False, f"调用工具数 {len(tools_called)} 超过上限 {expect['tools_called_count_max']}"

    if "answer_contains" in expect:
        if expect["answer_contains"] not in full_text:
            return False, f"回复应包含 '{expect['answer_contains']}'，实际 '{full_text[:80]}'"

    if "answer_contains_any" in expect:
        if not any(s in full_text for s in expect["answer_contains_any"]):
            return False, f"回复应包含任一 {expect['answer_contains_any']}，实际 '{full_text[:80]}'"

    if "answer_min_length" in expect:
        if len(full_text) < expect["answer_min_length"]:
            return False, f"回复长度 {len(full_text)} < {expect['answer_min_length']}"

    return True, "pass"


def run_case(case: dict) -> dict:
    store = FakeChromaStore()
    rag = FakeDocumentRAG()
    mm, orch = build_agent(store, document_rag=rag)

    setup = case.get("setup", {})
    for item in setup.get("preload_long_term", []):
        mm.store_long_term(item["text"], memory_type=item.get("memory_type", "event"))
    for k, v in setup.get("preload_profile", {}).items():
        mm.update_user_profile(k, v)
    for doc in setup.get("preload_rag", []):
        rag.add_document(doc["text"], source=doc.get("source", ""))

    events = _drive(orch, case["input"])
    ok, msg = _check_expectation(events, case.get("expect", {}))
    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "ok": ok,
        "msg": msg,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="只跑指定 id 的用例")
    parser.add_argument("--golden", default=os.path.join(_ROOT, "eval", "golden_set.json"))
    args = parser.parse_args()

    with open(args.golden, "r", encoding="utf-8") as f:
        golden = json.load(f)

    cases = golden["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"未找到 case: {args.case}")
            sys.exit(1)

    results = [run_case(c) for c in cases]

    passed = sum(1 for r in results if r["ok"])
    total = len(results)

    print(f"\n{'='*60}")
    print(f"Voice Agent Golden Eval  |  通过 {passed}/{total}")
    print(f"{'='*60}")
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        print(f"  {mark} [{r['category']:10}] {r['id']:20}  {r['msg']}")
    print(f"{'='*60}\n")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()