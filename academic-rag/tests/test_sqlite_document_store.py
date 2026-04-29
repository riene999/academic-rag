import numpy as np

from src.rag.retriever import FAISSRetriever
from src.storage.sqlite_store import SQLiteDocumentStore
from src.utils.pdf_parser import Document


class DummyEmbedder:
    dimension = 4

    def embed_documents(self, texts):
        return np.eye(len(texts), self.dimension, dtype=np.float32)

    def embed_query(self, query):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def test_sqlite_store_round_trips_documents(tmp_path):
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite")
    docs = [
        Document(
            content="chunk one",
            metadata={"source": "paper.pdf", "page": 1, "chunk_index": 0},
            chunk_id="paper_chunk_0",
        ),
        Document(
            content="chunk two",
            metadata={"source": "paper.pdf", "page": 2, "chunk_index": 1},
            chunk_id="paper_chunk_1",
        ),
    ]

    store.replace_all_documents(docs)

    loaded = store.load_documents()
    assert [doc.chunk_id for doc in loaded] == ["paper_chunk_0", "paper_chunk_1"]
    assert loaded[1].metadata["page"] == 2
    assert store.current_version() == 1

    listed = store.list_documents()
    assert listed[0]["source_name"] == "paper.pdf"
    assert listed[0]["chunk_count"] == 2


def test_retriever_saves_metadata_to_sqlite(tmp_path):
    retriever = FAISSRetriever(
        embedder=DummyEmbedder(),
        dimension=4,
        index_path=str(tmp_path / "faiss_index"),
        result_cache_enabled=False,
    )
    docs = [
        Document(content="alpha", metadata={"source": "paper.pdf"}, chunk_id="c0"),
        Document(content="beta", metadata={"source": "paper.pdf"}, chunk_id="c1"),
    ]

    retriever.add_documents(docs)
    retriever.save()

    reloaded = FAISSRetriever(
        embedder=DummyEmbedder(),
        dimension=4,
        index_path=str(tmp_path / "faiss_index"),
        result_cache_enabled=False,
    )
    assert reloaded.load()
    assert [doc.content for doc in reloaded.documents] == ["alpha", "beta"]
    assert reloaded.list_documents()[0]["chunk_count"] == 2
