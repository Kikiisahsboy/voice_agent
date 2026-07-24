"""MemoryManager 与 ChromaMemoryStore 测试（使用 Fake Chroma Store）。"""

from voice_agent.agent.memory_manager import MemoryManager


class FakeChromaStore:
    """模拟 ChromaMemoryStore，仅用于单元测试。"""

    def __init__(self):
        self._docs: list[dict] = []
        self._next_id = 0

    def store(self, text, metadata=None, memory_id=None):
        self._next_id += 1
        mid = memory_id or f"mem_{self._next_id}"
        meta = dict(metadata or {})
        meta.setdefault("timestamp", 0.0)
        meta.setdefault("last_accessed", meta["timestamp"])
        self._docs.append({"id": mid, "text": text, "metadata": meta})
        return mid

    def search(self, query, top_k=3, threshold=0.0, where=None):
        # 简单模拟：返回包含 query 任一字符的 doc，按"长度近似"模拟 score
        results = []
        keywords = set(query)
        for d in self._docs:
            meta = d["metadata"]
            if where:
                ok = all(meta.get(k) == v for k, v in where.items())
                if not ok:
                    continue
            common = sum(1 for c in d["text"] if c in keywords)
            if common == 0:
                continue
            score = min(1.0, common / max(len(keywords), 1))
            if score >= threshold:
                results.append({
                    "id": d["id"], "text": d["text"],
                    "score": score, "metadata": meta,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_recent(self, limit=10):
        return [d["text"] for d in self._docs[-limit:]]

    def decay(self, *args, **kwargs):
        pass

    def forget(self, mid):
        self._docs = [d for d in self._docs if d["id"] != mid]

    def count(self):
        return len(self._docs)


def test_short_term_window():
    mm = MemoryManager(FakeChromaStore(), short_term_max_turns=2)
    mm.set_system_prompt("system")
    mm.add_user_message("u1")
    mm.add_assistant_message("a1")
    mm.add_user_message("u2")
    mm.add_assistant_message("a2")
    mm.add_user_message("u3")
    mm.add_assistant_message("a3")
    msgs = mm.get_messages()
    # 只保留最近 2 轮（4 条非 system 消息）
    non_sys = [m for m in msgs if m["role"] != "system"]
    assert len(non_sys) == 4
    assert non_sys[0]["content"] == "u2"


def test_long_term_layered_storage():
    store = FakeChromaStore()
    mm = MemoryManager(store)
    mm.store_long_term("用户住在上海", memory_type="fact")
    mm.store_long_term("喜欢简洁回答", memory_type="preference")
    mm.store_long_term("讨论了天气", memory_type="event")
    assert store.count() == 3


def test_long_term_search_by_type():
    store = FakeChromaStore()
    mm = MemoryManager(store)
    mm.store_long_term("住上海", memory_type="fact")
    mm.store_long_term("喜欢简短", memory_type="preference")
    mm.store_long_term("查天气", memory_type="event")

    facts = mm.search_long_term("住", top_k=5, memory_types=["fact"])
    assert "住上海" in facts
    assert all("喜欢" not in f for f in facts)


def test_user_profile():
    mm = MemoryManager(FakeChromaStore())
    assert mm.get_user_profile() == {}
    mm.update_user_profile("姓名", "小明")
    mm.update_user_profile("职业", "学生")
    assert mm.get_user_profile() == {"姓名": "小明", "职业": "学生"}


def test_build_context_includes_profile_and_memory():
    mm = MemoryManager(FakeChromaStore(), short_term_max_turns=5)
    mm.set_system_prompt("base")
    mm.update_user_profile("姓名", "小明")
    mm.store_long_term("住上海", memory_type="fact")

    ranked = mm.search_long_term_ranked("住", top_k=3)
    msgs = mm.build_context_messages("今天天气", long_term_ranked=ranked)
    assert msgs[0]["role"] == "system"
    sys_content = msgs[0]["content"]
    assert "base" in sys_content
    assert "小明" in sys_content
    assert "住上海" in sys_content
    assert msgs[-1] == {"role": "user", "content": "今天天气"}


def test_safe_parse_json_direct():
    assert MemoryManager._safe_parse_json('{"a": 1}') == {"a": 1}


def test_safe_parse_json_with_markdown():
    out = MemoryManager._safe_parse_json("```json\n{\"a\": 2}\n```")
    assert out == {"a": 2}


def test_safe_parse_json_with_garbage():
    assert MemoryManager._safe_parse_json("not json at all") is None


def test_safe_parse_json_extract_braces():
    out = MemoryManager._safe_parse_json("前面的话 {\"k\": \"v\"} 后面的话")
    assert out == {"k": "v"}


def test_reset_session_keeps_system_prompt():
    mm = MemoryManager(FakeChromaStore())
    mm.set_system_prompt("sys")
    mm.add_user_message("u1")
    mm.add_assistant_message("a1")
    mm.reset_session()
    msgs = mm.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"