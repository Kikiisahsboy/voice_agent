"""BM25 索引、Hybrid Retriever、DocumentChunker 测试。"""

from voice_agent.memory.hybrid_retriever import (
    BM25Index,
    DocumentChunker,
    _char_bigrams,
    _tokenize,
)


def test_tokenize_chinese_english():
    tokens = _tokenize("Hello 世界 2024")
    assert "Hello" in tokens
    # 贪婪匹配中文：连续中文作为一个 token
    assert "世界" in tokens
    assert "2024" in tokens
    assert len(tokens) == 3


def test_char_bigrams():
    bg = _char_bigrams("你好世界")
    assert "你好" in bg
    assert "好世" in bg
    assert "世界" in bg


def test_bm25_basic():
    idx = BM25Index()
    idx.add("d1", "苹果是水果")
    idx.add("d2", "香蕉也是水果")
    idx.add("d3", "汽车是交通工具")

    res = idx.search("水果", top_k=3)
    assert len(res) == 2
    ids = {r["id"] for r in res}
    assert ids == {"d1", "d2"}


def test_bm25_ranking():
    idx = BM25Index()
    idx.add("d1", "苹果")
    idx.add("d2", "苹果 苹果 苹果")
    idx.add("d3", "苹果香蕉")
    res = idx.search("苹果", top_k=3)
    assert res[0]["id"] == "d2"
    assert res[0]["score"] >= res[1]["score"]


def test_bm25_no_match():
    idx = BM25Index()
    idx.add("d1", "hello world")
    res = idx.search("完全不相关", top_k=3)
    assert res == []


def test_chunker_basic():
    chunker = DocumentChunker(chunk_size=200, overlap=20)
    chunks = chunker.split("第一段。\n\n第二段内容。", source="t1")
    assert len(chunks) >= 1
    assert all("id" in c and "text" in c for c in chunks)


def test_chunker_long_paragraph():
    chunker = DocumentChunker(chunk_size=50, overlap=10)
    long_text = "这是一个很长的段落。" * 20
    chunks = chunker.split(long_text, source="t2")
    assert len(chunks) > 1
    assert all(c["text"] for c in chunks)


def test_chunker_metadata():
    chunker = DocumentChunker()
    chunks = chunker.split("A\n\nB\n\nC", source="doc.txt")
    assert all(c["metadata"]["source"] == "doc.txt" for c in chunks)
    assert all("chunk_index" in c["metadata"] for c in chunks)