from contextlib import asynccontextmanager
from functools import partial
from typing import Optional
import asyncio
import time

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from loguru import logger
import json
import tempfile
import os

from src.rag.pipeline import RAGPipeline
from src.rag.pipeline import RAGResponse
from src.agent.agent import PaperAgent
from src.shared.context import create_pipeline


rag_pipeline: Optional[RAGPipeline] = None
paper_agent: Optional[PaperAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_pipeline, paper_agent
    logger.info("服务启动，加载模型...")
    rag_pipeline = create_pipeline("config.yaml")
    paper_agent = PaperAgent(rag_pipeline, rag_pipeline.config.llm)
    logger.info("模型加载完成，服务就绪")
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="学术论文RAG问答系统",
    description="基于RAG和Agent的学术论文智能问答，支持PDF上传和语义检索",
    version="1.1.2",
    lifespan=lifespan,
)


# ==================== 数据模型 ====================

class QueryRequest(BaseModel):
    question: str
    use_agent: bool = False   # True: 使用Agent多轮检索；False: 直接RAG
    session_id: str = "default"  # Agent记忆会话ID
    use_memory: bool = True      # Agent模式下是否使用会话记忆


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


class IndexResponse(BaseModel):
    message: str
    chunks_added: int


class ClearMemoryRequest(BaseModel):
    session_id: Optional[str] = None


# ==================== API接口 ====================

async def _run_sync(func, *args, **kwargs):
    # 把同步耗时函数丢到线程池，避免阻塞 async 服务
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def _run_with_faiss_lock(func, *args, **kwargs):
    # 先给FAISS操作加锁，再丢到线程池跑
    async with rag_pipeline.retriever.faiss_lock:
        return await _run_sync(func, *args, **kwargs)


def _build_memory_aware_question(question: str, session_id: str, use_memory: bool) -> str:
    # 把历史对话拼进问题，帮助普通RAG理解追问指代
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
        "下面是同一会话的历史问答，请结合历史理解当前问题中的指代，"
        "但回答仍必须基于检索到的论文证据。\n\n"
        f"历史问答：\n{history_text}\n\n"
        f"当前问题：{question}"
    )


def _remember_turn(session_id: str, question: str, answer: str, use_memory: bool) -> None:
    if use_memory and paper_agent:
        paper_agent.memory.add_turn(session_id, question, answer)


def _format_retrieved_chunk(chunk) -> dict:
    metadata = chunk.document.metadata or {}
    source = metadata.get("source")
    page = metadata.get("page")
    return {
        "chunk_id": chunk.document.chunk_id,
        "id": chunk.document.chunk_id,
        "text": chunk.document.content,
        "content": chunk.document.content,
        "score": round(float(chunk.score), 6),
        "rank": chunk.rank,
        "source": source,
        "page": page,
        "metadata": metadata,
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    index_size = 0
    if rag_pipeline:
        async with rag_pipeline.retriever.faiss_lock:
            index_size = rag_pipeline.retriever.index.ntotal
    return {
        "status": "ok",
        "index_size": index_size,
        "memory_sessions": paper_agent.memory.session_count() if paper_agent else 0,
    }


@app.post("/upload", response_model=IndexResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    上传PDF论文并建立索引
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持PDF文件")

    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks_added = await _run_with_faiss_lock(
            rag_pipeline.index_documents_from_pdf,
            tmp_path,
            source_name=file.filename,
        )
        return IndexResponse(
            message=f"成功索引: {file.filename}",
            chunks_added=chunks_added,
        )
    finally:
        os.unlink(tmp_path)  # 清理临时文件


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    问答接口（非流式）
    use_agent=False: 标准RAG
    use_agent=True: Agent多轮检索
    """
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
    chunks = await _run_with_faiss_lock(
        rag_pipeline.retrieve_chunks,
        effective_question,
    )
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
    """给外部测评系统用的 RAG 问答接口"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    if request.stream:
        logger.info("/ask received stream=true; returning non-streaming JSON for eval compatibility")

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
    """清空Agent会话记忆；不传session_id则清空全部。"""
    if not paper_agent:
        raise HTTPException(status_code=503, detail="Agent尚未初始化")
    paper_agent.clear_memory(request.session_id)
    return {"status": "ok", "cleared_session_id": request.session_id}


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    流式问答接口，逐token返回（SSE格式）
    """
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
        # 先返回检索来源信息
        import json
        sources = [
            {"source": c.document.metadata.get("source"), "score": round(c.score, 4)}
            for c in chunks
        ]
        yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

        # 再流式返回回答
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
    """直接索引文本（测试用）"""
    chunks_added = await _run_with_faiss_lock(rag_pipeline.index_text, text, source_name)
    return {"chunks_added": chunks_added}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8011, reload=True)
