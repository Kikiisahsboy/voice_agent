# -*- coding: utf-8 -*-
"""Hybrid Retriever — 向量 + BM25 关键词融合，支持 Rerank。

实现：
- 向量检索：复用 ChromaMemoryStore
- BM25：对中文按字符 bigram 分词
- 融合：Reciprocal Rank Fusion (RRF)
- Rerank：基于关键词重叠率 + 向量分加权的简易打分（无 cross-encoder 时使用）

若环境有 sentence-transformers 的 cross-encoder 模型，可启用更精细的 Rerank。
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Optional

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[一-鿿]+|[A-Za-z]+|\d+")


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：连续中文作为一个 token，英文按单词，数字独立。"""
    return _TOKEN_RE.findall(text)


def _char_bigrams(text: str) -> list[str]:
    """中文友好的 bigram 切分。"""
    tokens = _tokenize(text)
    out: list[str] = []
    for t in tokens:
        if re.fullmatch(r"[一-鿿]+", t) and len(t) > 1:
            for i in range(len(t) - 1):
                out.append(t[i : i + 2])
        out.append(t)
    return out


class BM25Index:
    """轻量 BM25 索引（小语料够用）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []  # [{id, text, tokens, length}]
        self.df: Counter = Counter()
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    def add(self, doc_id: str, text: str):
        tokens = _char_bigrams(text)
        self.docs.append({
            "id": doc_id, "text": text, "tokens": tokens,
        })
        self.n_docs += 1
        for t in set(tokens):
            self.df[t] += 1
        self._update_avgdl()

    def _update_avgdl(self):
        if not self.docs:
            self.avgdl = 0.0
            return
        total = sum(len(d["tokens"]) for d in self.docs)
        self.avgdl = total / len(self.docs)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        q_tokens = _char_bigrams(query)
        if not q_tokens or not self.docs:
            return []

        scores = []
        for i, doc in enumerate(self.docs):
            tf = Counter(doc["tokens"])
            score = 0.0
            doc_len = len(doc["tokens"])
            for qt in q_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                df = self.df.get(qt, 0)
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                denom = f + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self.avgdl, 1)
                )
                score += idf * f * (self.k1 + 1) / denom
            scores.append({"idx": i, "score": score, "doc": doc})

        scores.sort(key=lambda x: x["score"], reverse=True)
        out = []
        for s in scores[:top_k]:
            if s["score"] <= 0:
                break
            out.append({
                "id": s["doc"]["id"],
                "text": s["doc"]["text"],
                "score": s["score"],
            })
        return out

    def clear(self):
        self.docs.clear()
        self.df.clear()
        self.n_docs = 0
        self.avgdl = 0.0


class HybridRetriever:
    """向量 + BM25 混合检索器，RRF 融合，可选 Rerank。

    用法：
        hr = HybridRetriever(chroma_store, collection_key="doc")
        hr.add_documents([{"id": ..., "text": ...}, ...])
        results = hr.search("查询", top_k=5)
    """

    def __init__(
        self,
        chroma_store,
        bm25_collection: Optional[Any] = None,
        rrf_k: int = 60,
    ):
        self._chroma = chroma_store
        self._bm25 = bm25_collection or BM25Index()
        self._rrf_k = rrf_k

    def add_documents(self, docs: list[dict]):
        """添加文档到向量库和 BM25 索引。"""
        for doc in docs:
            did = doc["id"]
            text = doc["text"]
            meta = doc.get("metadata", {})
            self._chroma.store(text=text, metadata=meta, memory_id=did)
            self._bm25.add(did, text)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        where: Optional[dict] = None,
        rerank: bool = True,
    ) -> list[dict]:
        """混合检索。

        Args:
            query: 查询文本
            top_k: 返回数量
            threshold: 过滤最低分
            where: ChromaDB metadata 过滤
            rerank: 是否对融合结果做二次排序（关键词 + 向量分加权）

        Returns:
            [{id, text, score, vector_score, bm25_score, metadata}, ...]
        """
        # 1. 向量检索（多取一些用于融合）
        vector_results = self._chroma.search(
            query, top_k=top_k * 2, threshold=0.0, where=where
        )
        # 2. BM25 检索
        bm25_results = self._bm25.search(query, top_k=top_k * 2)
        # 3. RRF 融合
        fused = self._rrf_fusion(vector_results, bm25_results)

        # 4. 可选 Rerank
        if rerank:
            fused = self._rerank(query, fused, vector_results, bm25_results)

        # 过滤 + 截断
        fused = [r for r in fused if r.get("score", 0) >= threshold]
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_k]

    def _rrf_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """Reciprocal Rank Fusion: score = Σ 1/(k + rank)。"""
        scores: dict[str, dict] = {}
        for rank, r in enumerate(vector_results):
            rid = r["id"]
            scores.setdefault(rid, {
                "id": rid, "text": r["text"],
                "score": 0.0, "vector_score": r.get("score", 0.0),
                "bm25_score": 0.0,
                "metadata": r.get("metadata", {}),
            })
            scores[rid]["score"] += 1.0 / (self._rrf_k + rank + 1)

        for rank, r in enumerate(bm25_results):
            rid = r["id"]
            scores.setdefault(rid, {
                "id": rid, "text": r["text"],
                "score": 0.0, "vector_score": 0.0,
                "bm25_score": 0.0,
                "metadata": {},
            })
            scores[rid]["score"] += 1.0 / (self._rrf_k + rank + 1)
            scores[rid]["bm25_score"] = max(
                scores[rid]["bm25_score"], r.get("score", 0.0)
            )

        # 用 metadata 补充（从向量结果）
        for r in vector_results:
            if r["id"] in scores and r.get("metadata"):
                scores[r["id"]]["metadata"] = r["metadata"]

        return list(scores.values())

    def _rerank(
        self,
        query: str,
        fused: list[dict],
        vector_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """简易 Rerank：query token 在 doc 中出现率 × 向量分。

        比 cross-encoder 弱很多，但无需额外模型，作为 fallback。
        """
        q_tokens = set(_tokenize(query)) | set(_char_bigrams(query))
        v_max = max((r["score"] for r in vector_results), default=1.0) or 1.0
        b_max = max((r["score"] for r in bm25_results), default=1.0) or 1.0

        for item in fused:
            doc_tokens = set(_tokenize(item["text"])) | set(
                _char_bigrams(item["text"])
            )
            if not doc_tokens:
                overlap = 0.0
            else:
                overlap = len(q_tokens & doc_tokens) / max(len(q_tokens), 1)

            v_norm = item.get("vector_score", 0.0) / v_max
            b_norm = item.get("bm25_score", 0.0) / b_max
            # 权重：向量 0.6 + 关键词 0.3 + RRF 0.1
            item["score"] = 0.6 * v_norm + 0.3 * b_norm + 0.1 * overlap

        return fused


class DocumentChunker:
    """文档分块器：按段落 + 固定长度回退。"""

    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, source: str = "unknown") -> list[dict]:
        """分块并返回 [{id, text, metadata}]。"""
        chunks = self._paragraph_split(text)
        out = []
        for i, chunk in enumerate(chunks):
            cid = f"{source}::chunk_{i}"
            out.append({
                "id": cid,
                "text": chunk.strip(),
                "metadata": {"source": source, "chunk_index": i},
            })
        return out

    def _paragraph_split(self, text: str) -> list[str]:
        # 先按双换行分段，再按句号分段，最后按 chunk_size 强制切
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(p) <= self.chunk_size:
                chunks.append(p)
                continue
            # 句子切分
            sentences = re.split(r"(?<=[。！？.!?\n])\s*", p)
            current = ""
            for s in sentences:
                if not s:
                    continue
                if len(current) + len(s) <= self.chunk_size:
                    current += s
                else:
                    if current:
                        chunks.append(current)
                    # 单句超过 chunk_size：sliding window 切成多块
                    for piece in self._sliding_window(s):
                        if len(piece) > self.chunk_size:
                            chunks.append(piece[: self.chunk_size])
                        else:
                            chunks.append(piece)
                    current = ""
            if current:
                chunks.append(current)
        return chunks

    def _sliding_window(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        result = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            result.append(text[start:end])
            if end >= len(text):
                break
            start = end - self.overlap
        return result


class DocumentRAG:
    """文档型 RAG：加载 → 分块 → 检索 → 拼装 prompt 上下文。"""

    def __init__(self, persist_dir: str):
        from voice_agent.memory.chroma_store import ChromaMemoryStore
        from voice_agent.memory.hybrid_retriever import (
            DocumentChunker, HybridRetriever,
        )

        self._chroma = ChromaMemoryStore(
            persist_dir=persist_dir,
            collection_name="documents",
        )
        self._chunker = DocumentChunker()
        self._retriever = HybridRetriever(self._chroma)

    def load_file(self, path: str) -> int:
        """加载一个文本文件，返回分块数。"""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.load_text(text, source=path)

    def load_text(self, text: str, source: str = "inline") -> int:
        chunks = self._chunker.split(text, source=source)
        self._retriever.add_documents(chunks)
        logger.info("已加载 %d 个分块 (source=%s)", len(chunks), source)
        return len(chunks)

    def search(
        self, query: str, top_k: int = 3, threshold: float = 0.1
    ) -> list[str]:
        results = self._retriever.search(query, top_k=top_k, threshold=threshold)
        return [r["text"] for r in results]

    def format_context(self, passages: list[str]) -> str:
        if not passages:
            return ""
        lines = ["以下是与问题相关的知识库内容："]
        for i, p in enumerate(passages, 1):
            lines.append(f"[{i}] {p}")
        return "\n".join(lines)

    @property
    def doc_count(self) -> int:
        return self._chroma.count()