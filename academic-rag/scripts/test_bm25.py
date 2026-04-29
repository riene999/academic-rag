import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.bm25_retriever import BM25Retriever
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import RetrievedChunk
from src.utils.pdf_parser import Document


class DummyEmbedder:
    dimension = 4

    def embed_query(self, query: str) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(texts), 1))


@dataclass
class DummyBM25Config:
    enabled: bool = True
    top_k: int = 5
    k1: float = 1.5
    b: float = 0.75
    rrf_k: int = 60


@dataclass
class DummyConfig:
    bm25: DummyBM25Config = field(default_factory=DummyBM25Config)


def make_doc(text: str, chunk_id: str) -> Document:
    return Document(content=text, metadata={"source": "test"}, chunk_id=chunk_id)


def test_bm25_keyword_hit() -> None:
    retriever = BM25Retriever()
    retriever.load_documents(
        [
            make_doc("ordinary federated optimization", "doc_1"),
            make_doc("rarekeyword_xyz appears only here", "doc_2"),
        ]
    )
    chunks = retriever.retrieve("rarekeyword_xyz", top_k=2)
    assert chunks, "BM25 returned no chunks"
    assert chunks[0].document.chunk_id == "doc_2", "BM25 did not rank exact keyword first"
    print("OK BM25 keyword hit")


def test_rrf_fusion_keeps_sparse_only_hit() -> None:
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.config = DummyConfig()
    pipeline.bm25_retriever = BM25Retriever()
    pipeline.bm25_retriever.load_documents(
        [
            make_doc("dense result", "dense_doc"),
            make_doc("rarekeyword_xyz sparse result", "sparse_doc"),
        ]
    )

    dense_chunks = [
        RetrievedChunk(
            document=make_doc("dense result", "dense_doc"),
            score=0.9,
            rank=1,
        )
    ]
    fused = pipeline._hybrid_retrieve("rarekeyword_xyz", dense_chunks, retrieve_k=2)
    ids = [chunk.document.chunk_id for chunk in fused]
    assert "dense_doc" in ids, "dense hit missing after fusion"
    assert "sparse_doc" in ids, "BM25-only hit missing after fusion"
    print("OK hybrid fusion keeps sparse-only hit")


def test_bm25_add_documents() -> None:
    retriever = BM25Retriever()
    retriever.load_documents([make_doc("base document", "base")])
    retriever.add_documents([make_doc("newtoken_abc indexed later", "new")])
    chunks = retriever.retrieve("newtoken_abc", top_k=2)
    assert chunks and chunks[0].document.chunk_id == "new", "BM25 add_documents not searchable"
    print("OK BM25 add_documents searchable")


if __name__ == "__main__":
    test_bm25_keyword_hit()
    test_rrf_fusion_keeps_sparse_only_hit()
    test_bm25_add_documents()
