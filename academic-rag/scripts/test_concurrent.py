import asyncio
import shutil
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.retriever import FAISSRetriever
from src.utils.pdf_parser import Document


class DummyEmbedder:
    dimension = 4

    def _vector(self, text: str) -> np.ndarray:
        lowered = text.lower()
        if "concurrent_safety_unique" in lowered:
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query).reshape(1, -1)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vector(text) for text in texts]).astype(np.float32)


class FakePipeline:
    def __init__(self, index_path: str):
        self.retriever = FAISSRetriever(
            embedder=DummyEmbedder(),
            dimension=4,
            index_path=index_path,
            result_cache_enabled=False,
        )

    def query(self, question: str):
        return self.retriever.retrieve(question, top_k=3, score_threshold=0.0)

    def index_text(self, text: str, source_name: str = "manual") -> int:
        document = Document(
            content=text,
            metadata={"source": source_name, "chunk_index": len(self.retriever.documents)},
            chunk_id=f"{source_name}_chunk_{len(self.retriever.documents)}",
        )
        self.retriever.add_documents([document])
        self.retriever.save()
        return 1


async def run_with_faiss_lock(pipeline: FakePipeline, func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    async with pipeline.retriever.faiss_lock:
        return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def main() -> None:
    temp_dir = tempfile.mkdtemp(prefix="academic_rag_concurrent_")
    try:
        pipeline = FakePipeline(temp_dir)
        await run_with_faiss_lock(
            pipeline,
            pipeline.index_text,
            "alpha base document for concurrent query safety",
            "base",
        )

        async def query_task(i: int):
            chunks = await run_with_faiss_lock(pipeline, pipeline.query, "alpha")
            if not chunks:
                raise AssertionError(f"query task {i} returned no results")
            return chunks

        async def write_task():
            return await run_with_faiss_lock(
                pipeline,
                pipeline.index_text,
                "concurrent_safety_unique document added while queries are running",
                "concurrent",
            )

        query_results = await asyncio.gather(
            *[query_task(i) for i in range(10)],
            write_task(),
        )

        for i, chunks in enumerate(query_results[:10]):
            if not chunks:
                raise AssertionError(f"query result {i} is empty")

        added = query_results[-1]
        if added != 1:
            raise AssertionError("concurrent write did not add exactly one chunk")

        final_chunks = await run_with_faiss_lock(
            pipeline,
            pipeline.query,
            "concurrent_safety_unique",
        )
        if not any("concurrent_safety_unique" in chunk.document.content for chunk in final_chunks):
            raise AssertionError("newly indexed content was not retrievable after write")

        print("OK no deadlock")
        print("OK 10 concurrent queries returned results")
        print("OK write completed and new content is retrievable")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
