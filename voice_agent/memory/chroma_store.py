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

    # ── where 条件构造 ─────────────────────────────────────

    @staticmethod
    def build_where(
        memory_types: Optional[list[str]] = None,
        time_range: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> Optional[dict]:
        """把业务语义的条件转成 ChromaDB 原生 where 语法。

        Args:
            memory_types: 限定 memory_type ∈ 这些值
                - 1 个: {"memory_type": "fact"}
                - 多个: {"memory_type": {"$in": [...]}}
            time_range: 时间窗口
                - {"last_accessed_gte": ts} 或 {"timestamp_gte": ts}
                - {"last_accessed_lte": ts} 或 {"timestamp_lte": ts}
                多个字段可同时给，自动 $and 拼接
            extra: 其它已构造好的 ChromaDB where 片段（会被 $and 合并）

        Returns:
            None / 单条件 / {$and: [...]} 复合条件

        Examples:
            >>> build_where(memory_types=["fact", "preference"])
            {'memory_type': {'$in': ['fact', 'preference']}}
            >>> build_where(time_range={"last_accessed_gte": 1700000000})
            {'last_accessed': {'$gte': 1700000000}}
        """
        conditions: list[dict] = []

        if memory_types:
            if len(memory_types) == 1:
                conditions.append({"memory_type": memory_types[0]})
            else:
                conditions.append({"memory_type": {"$in": memory_types}})

        if time_range:
            for key, op in (
                ("last_accessed_gte", "$gte"),
                ("last_accessed_lte", "$lte"),
                ("timestamp_gte", "$gte"),
                ("timestamp_lte", "$lte"),
            ):
                if key in time_range:
                    field = key.rsplit("_", 1)[0]  # last_accessed_gte -> last_accessed
                    conditions.append({field: {op: time_range[key]}})

        if extra:
            # 兼容已经是 {"key": "value"} 或 {"$and": [...]} 的格式
            if isinstance(extra, dict) and set(extra.keys()) == {"$and"}:
                conditions.extend(extra["$and"])
            else:
                conditions.append(extra)

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.5,
        where: Optional[dict] = None,
        memory_types: Optional[list[str]] = None,
        time_range: Optional[dict] = None,
    ) -> list[dict]:
        """检索相关记忆。

        支持两种过滤方式（可叠加）：
            1. 直接传 where：ChromaDB 原生语法
            2. 传 memory_types / time_range：业务语义，自动 build_where

        Args:
            query: 查询文本
            top_k: 返回数量
            threshold: 相似度阈值，低于此分数的结果被过滤
            where: ChromaDB metadata 过滤条件，例 {"memory_type": "fact"}
            memory_types: 限定记忆类型（自动转 where）
            time_range: 时间窗口（自动转 where）

        Returns:
            [{id, text, score, metadata}, ...] 按 score 降序
        """
        if self._collection.count() == 0:
            return []

        embedding = self._embedder.encode(query).tolist()

        # 业务参数 → 原生 where
        built_where = self.build_where(
            memory_types=memory_types, time_range=time_range
        )
        # 合并：业务构造 + 调用方直接传入的 where
        if where and built_where:
            final_where = {"$and": [where, built_where]}
        else:
            final_where = where or built_where

        # ChromaDB where filter 不能超过 collection 总数，否则会空集
        collection_count = self._collection.count()
        fetch_k = min(top_k, collection_count)

        kwargs = {
            "query_embeddings": [embedding],
            "n_results": fetch_k,
        }
        if final_where:
            kwargs["where"] = final_where

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

    def list_all(self, limit: int = 200) -> list[dict]:
        """列出全部记忆（含 id / text / metadata），按 timestamp 倒序。

        供前端记忆管理面板 / 调试使用。
        """
        if self._collection.count() == 0:
            return []
        data = self._collection.get(limit=limit)
        if not data["ids"]:
            return []
        items: list[dict] = []
        for i, mid in enumerate(data["ids"]):
            items.append({
                "id": mid,
                "text": data["documents"][i] if data["documents"] else "",
                "metadata": dict(data["metadatas"][i] or {}) if data["metadatas"] else {},
            })
        items.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
        return items

    def forget_by_text(self, keyword: str) -> int:
        """按关键字模糊匹配删除（任一记忆文本包含 keyword 即删）。

        Returns:
            删除条数。
        """
        if not keyword or self._collection.count() == 0:
            return 0
        data = self._collection.get()
        if not data["ids"]:
            return 0
        to_delete = [
            mid for i, mid in enumerate(data["ids"])
            if keyword in (data["documents"][i] or "")
        ]
        if to_delete:
            self._collection.delete(ids=to_delete)
            logger.info("按关键字 '%s' 删除 %d 条记忆", keyword, len(to_delete))
        return len(to_delete)


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