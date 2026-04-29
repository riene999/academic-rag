import asyncio
import json
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel
from redis.exceptions import RedisError

from src.agent.agent import PaperAgent
from src.jobs.indexing import enqueue_pdf_index_job, get_index_queue, make_redis_connection
from src.rag.pipeline import RAGPipeline, RAGResponse
from src.shared.context import create_pipeline


rag_pipeline: Optional[RAGPipeline] = None
paper_agent: Optional[PaperAgent] = None
index_mtime: Optional[float] = None

UPLOAD_DIR = Path("data/uploads")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, paper_agent, index_mtime
    logger.info("服务启动，加载模型...")
    rag_pipeline = create_pipeline("config.yaml")
    paper_agent = PaperAgent(rag_pipeline, rag_pipeline.config.llm)
    index_mtime = _get_index_mtime()
    logger.info("模型加载完成，服务就绪")
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="学术论文 RAG 问答系统",
    description="基于 RAG 和 Agent 的学术论文智能问答，支持 PDF 后台索引和语义检索。",
    version="1.2.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str
    use_agent: bool = False
    session_id: str = "default"
    use_memory: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: list
    question: str
    session_id: str


class AskRequest(BaseModel):
    question: str
    stream: bool = False
    top_k: Optional[int] = None
    case_id: Optional[str] = None


class IndexJobResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    status_url: str


class ClearMemoryRequest(BaseModel):
    session_id: Optional[str] = None


async def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def _run_with_faiss_lock(func, *args, **kwargs):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")

    if not hasattr(rag_pipeline, "retriever") or not hasattr(rag_pipeline.retriever, "faiss_lock"):
        return await _run_sync(func, *args, **kwargs)

    async with rag_pipeline.retriever.faiss_lock:
        return await _run_sync(func, *args, **kwargs)


def _get_index_mtime() -> Optional[float]:
    if rag_pipeline is None or not hasattr(rag_pipeline, "config"):
        return None
    if hasattr(rag_pipeline.retriever, "document_store"):
        return float(rag_pipeline.retriever.document_store.current_version())

    index_path = Path(rag_pipeline.config.vector_store.index_path)
    version_file = index_path / ".index_version"
    marker_file = version_file if version_file.exists() else index_path / "index.faiss"
    if not marker_file.exists():
        return None
    return marker_file.stat().st_mtime


async def _reload_index_if_changed() -> None:
    global index_mtime
    if rag_pipeline is None:
        return

    current_mtime = _get_index_mtime()
    if current_mtime is None or current_mtime == index_mtime:
        return

    async with rag_pipeline.retriever.faiss_lock:
        current_mtime = _get_index_mtime()
        if current_mtime is None or current_mtime == index_mtime:
            return

        loaded = await _run_sync(rag_pipeline.retriever.load)
        if loaded and rag_pipeline.bm25_retriever is not None:
            rag_pipeline.bm25_retriever.load_documents(rag_pipeline.retriever.documents)
        index_mtime = current_mtime
        logger.info("检测到后台索引更新，已重新加载 FAISS/BM25")


def _build_index_queue():
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")

    try:
        redis_connection = make_redis_connection(rag_pipeline.config.redis)
        redis_connection.ping()
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Redis is unavailable: {exc}") from exc

    return get_index_queue(redis_connection)


def _serialize_job(job) -> dict:
    status = job.get_status(refresh=True)
    result = job.result if status == "finished" and isinstance(job.result, dict) else {}
    return {
        "job_id": job.id,
        "status": status,
        "filename": result.get("filename") or job.meta.get("filename"),
        "chunks_added": result.get("chunks_added"),
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        "error": job.exc_info if status == "failed" else None,
    }


def _build_memory_aware_question(question: str, session_id: str, use_memory: bool) -> str:
    if not use_memory or not paper_agent:
        return question

    history = paper_agent.memory.get_messages(session_id)
    if not history:
        return question

    history_text = "\n".join(
        f"{'用户' if item['role'] == 'user' else '助手'}: {item['content']}"
        for item in history
    )
    return (
        "下面是同一会话的历史问答。请结合历史理解当前问题中的指代，"
        "但回答仍必须基于检索到的论文证据。\n\n"
        f"历史问答：\n{history_text}\n\n"
        f"当前问题：{question}"
    )


def _remember_turn(session_id: str, question: str, answer: str, use_memory: bool) -> None:
    if use_memory and paper_agent:
        paper_agent.memory.add_turn(session_id, question, answer)


def _format_retrieved_chunk(chunk) -> dict:
    metadata = chunk.document.metadata or {}
    return {
        "chunk_id": chunk.document.chunk_id,
        "id": chunk.document.chunk_id,
        "text": chunk.document.content,
        "content": chunk.document.content,
        "score": round(float(chunk.score), 6),
        "rank": chunk.rank,
        "source": metadata.get("source"),
        "page": metadata.get("page"),
        "metadata": metadata,
    }


@app.get("/health")
async def health_check():
    await _reload_index_if_changed()
    index_size = 0
    if rag_pipeline:
        async with rag_pipeline.retriever.faiss_lock:
            index_size = rag_pipeline.retriever.index.ntotal
    return {
        "status": "ok",
        "index_size": index_size,
        "memory_sessions": paper_agent.memory.session_count() if paper_agent else 0,
    }


@app.post("/upload", response_model=IndexJobResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    queue = _build_index_queue()
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF 文件过大")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid4().hex
    filename = Path(file.filename).name
    pdf_path = UPLOAD_DIR / f"{job_id}_{filename}"
    pdf_path.write_bytes(content)

    try:
        job = enqueue_pdf_index_job(
            queue,
            job_id=job_id,
            pdf_path=str(pdf_path),
            source_name=filename,
            config_path="config.yaml",
        )
    except Exception as exc:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail=f"索引任务入队失败: {exc}") from exc

    return IndexJobResponse(
        job_id=job.id,
        status=job.get_status(refresh=True),
        filename=filename,
        status_url=f"/jobs/{job.id}",
    )


@app.get("/jobs/{job_id}")
async def get_index_job(job_id: str):
    queue = _build_index_queue()
    job = queue.fetch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    if job.get_status(refresh=True) == "finished":
        await _reload_index_if_changed()
    return _serialize_job(job)


@app.get("/documents")
async def list_documents():
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")

    await _reload_index_if_changed()
    return {"documents": rag_pipeline.retriever.list_documents()}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    await _reload_index_if_changed()

    if request.use_agent:
        answer = await _run_with_faiss_lock(
            paper_agent.run,
            request.question,
            session_id=request.session_id,
            use_memory=request.use_memory,
        )
        return QueryResponse(
            answer=answer,
            sources=[],
            question=request.question,
            session_id=request.session_id,
        )

    effective_question = _build_memory_aware_question(
        request.question,
        request.session_id,
        request.use_memory,
    )
    chunks = await _run_with_faiss_lock(rag_pipeline.retrieve_chunks, effective_question)
    answer = await _run_sync(rag_pipeline.generator.generate, effective_question, chunks)
    response = RAGResponse(answer=answer, retrieved_chunks=chunks, query=effective_question)
    _remember_turn(request.session_id, request.question, response.answer, request.use_memory)
    sources = [
        {
            "source": chunk.document.metadata.get("source"),
            "page": chunk.document.metadata.get("page"),
            "score": round(chunk.score, 4),
            "content_preview": chunk.document.content[:150] + "...",
        }
        for chunk in response.retrieved_chunks
    ]
    return QueryResponse(
        answer=response.answer,
        sources=sources,
        question=request.question,
        session_id=request.session_id,
    )


@app.post("/ask")
async def ask(request: AskRequest):
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    if request.stream:
        logger.info("/ask received stream=true; returning non-streaming JSON for eval compatibility")

    await _reload_index_if_changed()
    start = time.perf_counter()
    chunks = await _run_with_faiss_lock(
        rag_pipeline.retrieve_chunks,
        request.question,
        top_k=request.top_k,
    )
    answer = await _run_sync(rag_pipeline.generator.generate, request.question, chunks)
    response = RAGResponse(answer=answer, retrieved_chunks=chunks, query=request.question)
    retrieved_chunks = [_format_retrieved_chunk(chunk) for chunk in response.retrieved_chunks]
    citations = [chunk["chunk_id"] for chunk in retrieved_chunks]
    latency_ms = int((time.perf_counter() - start) * 1000)

    return {
        "answer": response.answer,
        "retrieved_chunks": retrieved_chunks,
        "citations": citations,
        "trace": {
            "tools": [],
            "rewrite_query": response.query,
            "steps": [
                {
                    "name": "rag_query",
                    "question": request.question,
                    "top_k": request.top_k,
                    "retrieved_count": len(retrieved_chunks),
                }
            ],
            "case_id": request.case_id,
        },
        "latency_ms": latency_ms,
    }


@app.post("/memory/clear")
async def clear_memory(request: ClearMemoryRequest):
    if not paper_agent:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化")
    paper_agent.clear_memory(request.session_id)
    return {"status": "ok", "cleared_session_id": request.session_id}


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    await _reload_index_if_changed()

    if request.use_agent:
        answer = await _run_with_faiss_lock(
            paper_agent.run,
            request.question,
            session_id=request.session_id,
            use_memory=request.use_memory,
        )

        def generate_agent():
            yield f"data: {json.dumps({'type': 'sources', 'data': []}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'data': answer}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(generate_agent(), media_type="text/event-stream")

    effective_question = _build_memory_aware_question(
        request.question,
        request.session_id,
        request.use_memory,
    )
    chunks = await _run_with_faiss_lock(rag_pipeline.retrieve_chunks, effective_question)
    stream = rag_pipeline.generator.generate_stream(effective_question, chunks)

    def generate():
        sources = [
            {"source": c.document.metadata.get("source"), "score": round(c.score, 4)}
            for c in chunks
        ]
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

        answer_parts = []
        for token in stream:
            answer_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

        _remember_turn(
            request.session_id,
            request.question,
            "".join(answer_parts),
            request.use_memory,
        )
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/index/text")
async def index_text(text: str, source_name: str = "manual"):
    chunks_added = await _run_with_faiss_lock(rag_pipeline.index_text, text, source_name)
    return {"chunks_added": chunks_added}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8011, reload=True)
