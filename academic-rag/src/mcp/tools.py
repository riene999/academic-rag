"""Shared structured tool functions for paper retrieval and QA."""

from typing import Any, Dict, List

from src.rag.pipeline import RAGPipeline
from src.rag.retriever import RetrievedChunk


def format_retrieved_chunk(chunk: RetrievedChunk) -> Dict[str, Any]:
    metadata = chunk.document.metadata or {}
    source = metadata.get("source")
    page = metadata.get("page")
    return {
        "chunk_id": chunk.document.chunk_id,
        "id": chunk.document.chunk_id,
        "text": chunk.document.content,
        "content": chunk.document.content,
        "score": round(float(chunk.score), 6),
        "rank": int(chunk.rank),
        "source": source,
        "page": page,
        "metadata": metadata,
    }


def search_papers(
    pipeline: RAGPipeline,
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
    use_reranker: bool = True,
) -> Dict[str, Any]:
    # 输入query和检索参数，调用pipeline中提供的函数检索chunk并返回结果
    chunks = pipeline.retrieve_chunks(
        question=query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranker=use_reranker,
    )
    retrieved_chunks = [format_retrieved_chunk(chunk) for chunk in chunks]
    return {
        "query": query,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "use_reranker": use_reranker,
        "retrieved_chunks": retrieved_chunks,
        "citations": [chunk["chunk_id"] for chunk in retrieved_chunks],
        "retrieved_count": len(retrieved_chunks),
    }


def ask_papers(
    pipeline: RAGPipeline,
    question: str,
    top_k: int | None = None,
) -> Dict[str, Any]:
    # 输入query，调用pipeline中提供的函数生成回答
    response = pipeline.query(question, top_k=top_k)
    retrieved_chunks = [
        format_retrieved_chunk(chunk) for chunk in response.retrieved_chunks
    ]
    return {
        "answer": response.answer,
        "question": question,
        "query": response.query,
        "top_k": top_k,
        "retrieved_chunks": retrieved_chunks,
        "citations": [chunk["chunk_id"] for chunk in retrieved_chunks],
        "trace": {
            "tools": ["ask_papers"],
            "rewrite_query": response.query,
            "steps": [
                {
                    "name": "rag_query",
                    "question": question,
                    "top_k": top_k,
                    "retrieved_count": len(retrieved_chunks),
                }
            ],
        },
    }


def get_index_status(pipeline: RAGPipeline, memory_sessions: int = 0) -> Dict[str, Any]:
    # 获得向量库目前的向量数
    retriever = pipeline.retriever
    index = getattr(retriever, "index", None)
    documents: List[Any] = getattr(retriever, "documents", []) or []
    return {
        "status": "ok",
        "index_size": int(getattr(index, "ntotal", 0) or 0),
        "document_count": len(documents),
        "memory_sessions": int(memory_sessions),
    }
