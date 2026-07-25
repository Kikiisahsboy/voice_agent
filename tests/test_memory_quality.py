"""记忆质量门控 + chroma_store 管理接口测试。"""

from voice_agent.agent.memory_manager import MemoryManager


class FakeChromaStore:
    """复用 tests/test_memory_manager.py 中的最小 Fake。"""

    def __init__(self):
        self._docs = []
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
        kw = set(query)
        results = []
        for d in self._docs:
            meta = d["metadata"]
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            common = sum(1 for c in d["text"] if c in kw)
            if common == 0:
                continue
            score = min(1.0, common / max(len(kw), 1))
            if score >= threshold:
                results.append({"id": d["id"], "text": d["text"],
                                "score": score, "metadata": meta})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_recent(self, limit=10):
        return [d["text"] for d in self._docs[-limit:]]

    def decay(self, *a, **kw):
        pass

    def forget(self, mid):
        self._docs = [d for d in self._docs if d["id"] != mid]

    def list_all(self, limit=200):
        return [
            {"id": d["id"], "text": d["text"], "metadata": dict(d["metadata"])}
            for d in self._docs[:limit]
        ]

    def forget_by_text(self, keyword):
        before = len(self._docs)
        self._docs = [d for d in self._docs if keyword not in (d["text"] or "")]
        return before - len(self._docs)

    def count(self):
        return len(self._docs)


def test_short_text_rejected():
    store = FakeChromaStore()
    mm = MemoryManager(store)
    assert mm.store_long_term("hi", memory_type="fact") is None
    assert store.count() == 0


def test_chitchat_blacklist_rejected():
    store = FakeChromaStore()
    mm = MemoryManager(store)
    # "你好呀" 是 3 字符（过长度门），但命中黑名单
    assert mm.store_long_term("你好呀", memory_type="fact") is None
    assert mm.store_long_term("你好呀", memory_type="preference") is None
    assert store.count() == 0


def test_dedup_skipped():
    store = FakeChromaStore()
    mm = MemoryManager(store)
    mm.store_long_term("用户住在上海", memory_type="fact")
    # 完全重复 — Fake 搜索按字符重叠打分，重复文本 score = 1.0 ≥ 0.95
    mid2 = mm.store_long_term("用户住在上海", memory_type="fact")
    assert mid2 is None
    assert store.count() == 1


def test_list_all_returns_metadata():
    store = FakeChromaStore()
    mm = MemoryManager(store)
    mm.store_long_term("用户住在上海", memory_type="fact")
    mm.store_long_term("喜欢简洁", memory_type="preference")

    items = store.list_all()
    assert len(items) == 2
    types = {it["metadata"]["memory_type"] for it in items}
    assert types == {"fact", "preference"}


def test_forget_by_text_keyword():
    store = FakeChromaStore()
    mm = MemoryManager(store)
    mm.store_long_term("用户住在上海", memory_type="fact")
    mm.store_long_term("今天天气不错", memory_type="event")

    n = store.forget_by_text("上海")
    assert n == 1
    assert store.count() == 1
    assert "天气" in store.list_all()[0]["text"]


def test_forget_by_text_no_match():
    store = FakeChromaStore()
    mm = MemoryManager(store)
    mm.store_long_term("用户住在上海", memory_type="fact")

    assert store.forget_by_text("北京") == 0
    assert store.count() == 1


def test_normal_fact_accepted():
    store = FakeChromaStore()
    mm = MemoryManager(store)
    mid = mm.store_long_term("用户对青霉素过敏", memory_type="fact")
    assert mid is not None
    assert store.count() == 1