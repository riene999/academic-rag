from dataclasses import dataclass
from typing import List

from loguru import logger
from sentence_transformers import CrossEncoder

from src.rag.retriever import RetrievedChunk


@dataclass
class RerankResult:
    chunk: RetrievedChunk
    rerank_score: float


class CrossEncoderReranker:
    """Cross-encoder reranker for query-chunk relevance scoring."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        logger.info("Loading reranker model: {}", model_name)
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.document.content) for chunk in chunks]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        results = [
            RerankResult(chunk=chunk, rerank_score=float(score))
            for chunk, score in zip(chunks, scores)
        ]
        results.sort(key=lambda item: item.rerank_score, reverse=True)

        reranked: List[RetrievedChunk] = []
        for rank, item in enumerate(results[:top_k], start=1):
            # Keep score field as the final ranking score (rerank score).
            reranked.append(
                RetrievedChunk(
                    document=item.chunk.document,
                    score=item.rerank_score,
                    rank=rank,
                )
            )
        return reranked

