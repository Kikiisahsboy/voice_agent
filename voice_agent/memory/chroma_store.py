# -*- coding: utf-8 -*-
"""ChromaDB 长期记忆存储 — 存储、检索、衰减、元数据过滤。"""

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ChromaMemoryStore:
    """ChromaDB 封装的长期记忆存储。

    检索使用 cosine 距离（ChromaDB HNSW 默认）。
    注意：cosine distance ∈ [0, 2]，similarity = 1 - distance ∈ [-1, 1]。
    本类统一以 "score" 输出 ∈ [0, 1]，由调用方根据 threshold 过滤。
    """

    def __init__(
        self,
        persist_dir: str = "./voice_agent/chroma_data",
        collection_name: str = "conversation_memory",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
    ):
        import chromadb
        from sentence_transformers import SentenceTransformer

        os.makedirs(persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedder = SentenceTransformer(embedding_model)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB 就绪: %s (共 %d 条记忆)",
            persist_dir,
            self._collection.count(),
        )

    # ── 写入 ────────────────────────────────────────────

    def store(
        self,
        text: str,
        metadata: Optional[dict] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """存储一条记忆，返回 memory_id。"""
        if metadata is None:
            metadata = {}
        metadata.setdefault("timestamp", time.time())
        metadata.setdefault("last_accessed", metadata["timestamp"])

        embedding = self._embedder.encode(text).tolist()

        if memory_id is None:
            memory_id = f"mem_{int(time.time() * 1000)}"

        self._collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        return memory_id

    # ── 检索 ────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """检索相关记忆。

        Args:
            query: 查询文本
            top_k: 返回数量
            threshold: 相似度阈值，低于此分数的结果被过滤
            where: ChromaDB metadata 过滤条件，例 {"memory_type": "fact"}

        Returns:
            [{id, text, score, metadata}, ...] 按 score 降序
        """
        if self._collection.count() == 0:
            return []

        embedding = self._embedder.encode(query).tolist()

        kwargs = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, self._collection.count()),
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception as e:
            # ChromaDB 在 metadata filter 不存在时可能抛错
            logger.warning("ChromaDB query 失败（尝试无过滤重试）: %s", e)
            kwargs.pop("where", None)
            results = self._collection.query(**kwargs)

        memories: list[dict] = []
        if results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                # cosine distance -> similarity ∈ [-1, 1] -> 归一到 [0, 1]
                distance = (
                    results["distances"][0][i] if results["distances"] else 0.0
                )
                # distance ∈ [0, 2]，对 cosine 距离做 1 - d/2 映射到 [0, 1]
                score = max(0.0, 1.0 - distance / 2.0)

                if score < threshold:
                    continue

                doc = (
                    results["documents"][0][i]
                    if results["documents"]
                    else ""
                )
                meta = dict(
                    results["metadatas"][0][i] if results["metadatas"] else {}
                )
                memories.append({
                    "id": mem_id,
                    "text": doc,
                    "score": score,
                    "metadata": meta,
                })

        memories.sort(key=lambda x: x["score"], reverse=True)

        # 命中后更新 last_accessed
        if memories:
            try:
                ts = time.time()
                ids = [m["id"] for m in memories]
                all_meta = self._collection.get(ids=ids)
                updates = []
                for mem_id, meta in zip(all_meta["ids"], all_meta["metadatas"]):
                    meta["last_accessed"] = ts
                    updates.append(meta)
                self._collection.update(ids=ids, metadatas=updates)
            except Exception as e:
                logger.debug("更新 last_accessed 失败: %s", e)

        return memories

    # ── 删除/衰减 ───────────────────────────────────────

    def forget(self, memory_id: str):
        self._collection.delete(ids=[memory_id])

    def decay(self, max_age_days: int = 30, max_total: int = 1000):
        """删除过旧记忆和超出上限的记忆。"""
        count = self._collection.count()
        if count == 0:
            return

        now = time.time()
        cutoff = now - max_age_days * 86400

        all_data = self._collection.get()
        if not all_data["ids"]:
            return

        to_delete = []
        entries = []

        for i, mem_id in enumerate(all_data["ids"]):
            meta = (
                all_data["metadatas"][i] if all_data["metadatas"] else {}
            )
            ts = meta.get("timestamp", 0)
            la = meta.get("last_accessed", ts)

            if la < cutoff:
                to_delete.append(mem_id)
            else:
                entries.append((mem_id, la))

        if to_delete:
            self._collection.delete(ids=to_delete)
            logger.info("衰减: 删除 %d 条过期记忆", len(to_delete))

        entries.sort(key=lambda x: x[1])
        over_limit = len(entries) - max_total
        if over_limit > 0:
            ids_to_delete = [e[0] for e in entries[:over_limit]]
            self._collection.delete(ids=ids_to_delete)
            logger.info("衰减: 删除 %d 条超出上限的记忆", len(ids_to_delete))

    # ── 其它 ────────────────────────────────────────────

    def get_recent(self, limit: int = 10) -> list[str]:
        all_data = self._collection.get()
        if not all_data["ids"]:
            return []

        entries = []
        for i, mem_id in enumerate(all_data["ids"]):
            doc = all_data["documents"][i] if all_data["documents"] else ""
            meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
            ts = meta.get("timestamp", 0)
            entries.append((ts, doc))

        entries.sort(key=lambda x: x[0], reverse=True)
        return [e[1] for e in entries[:limit]]

    def count(self) -> int:
        return self._collection.count()