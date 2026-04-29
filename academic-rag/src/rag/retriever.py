import asyncio
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from loguru import logger

from src.cache.redis_cache import RedisCache, md5_key, redis_enabled
from src.rag.embedder import Embedder
from src.storage.sqlite_store import SQLiteDocumentStore
from src.utils.cache import TTLCache
from src.utils.pdf_parser import Document


@dataclass
class RetrievedChunk:
    document: Document
    score: float
    rank: int


class FAISSRetriever:
    def __init__(
        self,
        embedder: Embedder,
        dimension: int,
        index_path: str = "./data/faiss_index",
        result_cache_enabled: bool = True,
        result_cache_max_size: int = 5000,
        result_cache_ttl_seconds: int = 900,
        redis_config=None,
    ):
        self.embedder = embedder
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.document_store = SQLiteDocumentStore(self.index_path / "documents.sqlite")

        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Document] = []
        self.faiss_lock = asyncio.Lock()

        self.result_cache_enabled = result_cache_enabled
        self.result_cache = None
        if result_cache_enabled:
            if redis_enabled():
                self.result_cache = RedisCache[str, List[Tuple[int, float, int]]](
                    max_size=result_cache_max_size,
                    ttl_seconds=result_cache_ttl_seconds,
                    redis_config=redis_config,
                    value_codec="json",
                )
            else:
                self.result_cache = TTLCache[str, List[Tuple[int, float, int]]](
                    max_size=result_cache_max_size,
                    ttl_seconds=result_cache_ttl_seconds,
                )
        self._index_epoch = 0

    def _index_version(self) -> int:
        if isinstance(self.result_cache, RedisCache) and self.result_cache.using_redis:
            return self.result_cache.get_counter("rag:index_version", default=0)
        return self._index_epoch

    def _cache_key(self, query: str, top_k: int, score_threshold: float) -> str:
        normalized_query = " ".join(query.strip().split())
        version = self._index_version()
        return (
            f"ret:{version}:{md5_key(normalized_query)}:"
            f"{int(top_k)}:{round(float(score_threshold), 6)}"
        )

    def _serialize_results(self, items: List[Tuple[int, float, int]]) -> List[Tuple[int, float, int]]:
        return [(idx, float(score), int(rank)) for idx, score, rank in items]

    def _build_chunks_from_entries(self, entries: List[Tuple[int, float, int]]) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []
        for idx, score, rank in entries:
            if 0 <= idx < len(self.documents):
                chunks.append(
                    RetrievedChunk(
                        document=self.documents[idx],
                        score=float(score),
                        rank=int(rank),
                    )
                )
        return chunks

    def _invalidate_result_cache(self) -> None:
        if isinstance(self.result_cache, RedisCache) and self.result_cache.using_redis:
            self.result_cache.incr_counter("rag:index_version")
            return

        self._index_epoch += 1
        if self.result_cache is not None:
            self.result_cache.clear()

    def _clear_result_cache(self) -> None:
        if self.result_cache is not None:
            if isinstance(self.result_cache, RedisCache) and self.result_cache.using_redis:
                self.result_cache.clear("ret:*")
            else:
                self.result_cache.clear()

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        logger.info("Embedding {} chunks...", len(documents))
        texts = [doc.content for doc in documents]
        embeddings = self.embedder.embed_documents(texts).astype(np.float32)

        self.index.add(embeddings)
        self.documents.extend(documents)
        self._invalidate_result_cache()

        logger.info("Index size: {}", self.index.ntotal)

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.5) -> List[RetrievedChunk]:
        if self.index.ntotal == 0:
            logger.warning("Index is empty, add documents first")
            return []

        cache_key = self._cache_key(query, top_k, score_threshold)
        if self.result_cache_enabled and self.result_cache is not None:
            cached = self.result_cache.get(cache_key)
            if cached is not None:
                logger.debug("Retriever cache hit for query='{}'", query[:40])
                return self._build_chunks_from_entries(cached)

        query_embedding = self.embedder.embed_query(query).astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)
        scores, indices = scores[0], indices[0]

        entries: List[Tuple[int, float, int]] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx == -1:
                continue
            if score < score_threshold:
                continue
            entries.append((int(idx), float(score), rank))

        if self.result_cache_enabled and self.result_cache is not None:
            self.result_cache.set(cache_key, self._serialize_results(entries))

        results = self._build_chunks_from_entries(entries)
        logger.info("Retrieved {} chunks (threshold={})", len(results), score_threshold)
        return results

    def save(self) -> None:
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        self.document_store.replace_all_documents(self.documents, reason="save")
        logger.info("Saved index and metadata to {}", self.index_path)

    def load(self) -> bool:
        index_file = self.index_path / "index.faiss"
        legacy_docs_file = self.index_path / "documents.pkl"

        if not index_file.exists():
            logger.info("No existing index found, starting fresh")
            return False

        self.index = faiss.read_index(str(index_file))
        if self.document_store.has_chunks():
            self.documents = self.document_store.load_documents()
        elif legacy_docs_file.exists():
            with open(legacy_docs_file, "rb") as f:
                self.documents = pickle.load(f)
            self.document_store.replace_all_documents(self.documents, reason="migrate_pickle")
            logger.info("Migrated {} chunks from documents.pkl to SQLite", len(self.documents))
        else:
            self.documents = []
            logger.warning("Index file exists but no document metadata was found")

        self._clear_result_cache()
        if self.index.ntotal != len(self.documents):
            logger.warning(
                "Index/document count mismatch: vectors={}, documents={}",
                self.index.ntotal,
                len(self.documents),
            )
        logger.info("Loaded index successfully, total vectors={}", self.index.ntotal)
        return True

    def list_documents(self) -> list[dict]:
        return self.document_store.list_documents()
