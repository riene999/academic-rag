from fastapi.testclient import TestClient

import main
from src.rag.pipeline import RAGResponse
from src.rag.retriever import RetrievedChunk
from src.utils.pdf_parser import Document


class DummyPipeline:
    def query(self, question: str, top_k: int | None = None):
        assert question == "What is FedAvg?"
        assert top_k == 1
        return RAGResponse(
            answer="FedAvg averages local client updates.",
            query=question,
            retrieved_chunks=[
                RetrievedChunk(
                    document=Document(
                        content="FedAvg averages model updates from clients.",
                        metadata={"source": "fedavg.pdf", "page": 2},
                        chunk_id="fedavg_chunk_0",
                    ),
                    score=0.93,
                    rank=1,
                )
            ],
        )


def test_ask_returns_eval_compatible_payload(monkeypatch):
    monkeypatch.setattr(main, "rag_pipeline", DummyPipeline())
    client = TestClient(main.app)

    response = client.post(
        "/ask",
        json={
            "question": "What is FedAvg?",
            "stream": False,
            "top_k": 1,
            "case_id": "case_demo",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "FedAvg averages local client updates."
    assert payload["citations"] == ["fedavg_chunk_0"]
    assert payload["trace"]["case_id"] == "case_demo"
    assert isinstance(payload["latency_ms"], int)

    chunk = payload["retrieved_chunks"][0]
    assert chunk["chunk_id"] == "fedavg_chunk_0"
    assert chunk["id"] == "fedavg_chunk_0"
    assert chunk["text"] == "FedAvg averages model updates from clients."
    assert chunk["source"] == "fedavg.pdf"
    assert chunk["page"] == 2
    assert chunk["score"] == 0.93
