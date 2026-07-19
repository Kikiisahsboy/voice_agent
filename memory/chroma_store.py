# -*- coding: utf-8 -*-
"""ChromaDB 长期记忆存储 — 存储、检索、衰减。"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ChromaMemoryStore:
    """ChromaDB 封装的长期记忆存储。"""

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
        metadata.setdefault("last_accessed", time.time())

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

    def search(
        self, query: str, top_k: int = 3, threshold: float = 0.5
    ) -> list[dict]:
        """检索相关记忆。"""
        if self._collection.count() == 0:
            return []

        embedding = self._embedder.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                score = 1.0 - results["distances"][0][i] if results["distances"] else 0
                if score >= threshold:
                    doc = (
                        results["documents"][0][i]
                        if results["documents"]
                        else ""
                    )
                    meta = (
                        results["metadatas"][0][i]
                        if results["metadatas"]
                        else {}
                    )
                    memories.append({
                        "id": mem_id,
                        "text": doc,
                        "score": score,
                        "metadata": meta,
                    })
                    # 更新访问时间
                    meta["last_accessed"] = time.time()
                    self._collection.update(
                        ids=[mem_id], metadatas=[meta]
                    )

        # 按分数降序排列
        memories.sort(key=lambda x: x["score"], reverse=True)
        return memories

    def forget(self, memory_id: str):
        """删除一条记忆。"""
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

        # 删除过期记忆
        if to_delete:
            self._collection.delete(ids=to_delete)
            logger.info("衰减: 删除 %d 条过期记忆", len(to_delete))

        # 按 last_accessed 升序，删除最不活跃的记忆直到不超过上限
        entries.sort(key=lambda x: x[1])
        over_limit = len(entries) - max_total
        if over_limit > 0:
            ids_to_delete = [e[0] for e in entries[:over_limit]]
            self._collection.delete(ids=ids_to_delete)
            logger.info("衰减: 删除 %d 条超出上限的记忆", len(ids_to_delete))

    def get_recent(self, limit: int = 10) -> list[str]:
        """获取最近存储的记忆。"""
        all_data = self._collection.get()
        if not all_data["ids"]:
            return []

        entries = []
        for i, mem_id in enumerate(all_data["ids"]):
            doc = (
                all_data["documents"][i]
                if all_data["documents"]
                else ""
            )
            meta = (
                all_data["metadatas"][i] if all_data["metadatas"] else {}
            )
            ts = meta.get("timestamp", 0)
            entries.append((ts, doc))

        entries.sort(key=lambda x: x[0], reverse=True)
        return [e[1] for e in entries[:limit]]

    def count(self) -> int:
        return self._collection.count()
