import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.rag.pipeline import RAGResponse
from src.rag.retriever import RetrievedChunk
from src.utils.pdf_parser import Document


class DummyIndex:
    ntotal = 1


class DummyRetriever:
    def __init__(self):
        self.index = DummyIndex()
        self.documents = [
            Document(
                content="FedAvg averages local client updates.",
                metadata={"source": "fedavg.pdf", "page": 2},
                chunk_id="fedavg_chunk_0",
            )
        ]


class DummyPipeline:
    def __init__(self):
        self.retriever = DummyRetriever()

    def retrieve_chunks(
        self,
        question: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
        use_reranker: bool = True,
    ):
        del question, top_k, score_threshold, use_reranker
        return [
            RetrievedChunk(
                document=self.retriever.documents[0],
                score=0.93,
                rank=1,
            )
        ]

    def query(self, question: str, top_k: int | None = None):
        return RAGResponse(
            answer="FedAvg averages local client updates.",
            query=question,
            retrieved_chunks=self.retrieve_chunks(question, top_k=top_k),
        )


def _extract_json(result):
    structured_content = getattr(result, "structuredContent", None)
    if structured_content is None:
        structured_content = getattr(result, "structured_content", None)
    if structured_content is not None:
        return structured_content

    first = result.content[0]
    text = getattr(first, "text", None)
    if text is None:
        return first.model_dump()
    return json.loads(text)


@pytest.mark.anyio
async def test_mcp_tools_are_callable(tmp_path: Path):
    server_script = tmp_path / "dummy_mcp_server.py"
    repo_root = Path(__file__).resolve().parent.parent
    server_script.write_text(
        f"""
import sys
sys.path.insert(0, {str(repo_root)!r})

from mcp.server.fastmcp import FastMCP
import json
from src.mcp.tools import ask_papers as ask_papers_tool
from src.mcp.tools import get_index_status as get_index_status_tool
from src.mcp.tools import search_papers as search_papers_tool
from tests.test_mcp import DummyPipeline

mcp = FastMCP("dummy-academic-rag")
pipeline = DummyPipeline()

@mcp.tool(name="search_papers", description="Search indexed academic papers and return relevant evidence chunks.")
def search_papers(query: str, top_k: int | None = None):
    return json.dumps(search_papers_tool(pipeline, query=query, top_k=top_k), ensure_ascii=False)

@mcp.tool(name="ask_papers", description="Answer a question using the indexed academic paper corpus.")
def ask_papers(question: str, top_k: int | None = None):
    return json.dumps(ask_papers_tool(pipeline, question=question, top_k=top_k), ensure_ascii=False)

@mcp.tool(name="get_index_status", description="Return index size and document count for the loaded paper corpus.")
def get_index_status():
    return json.dumps(get_index_status_tool(pipeline), ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport="stdio")
""",
        encoding="utf-8",
    )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        cwd=str(repo_root),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            listed = await session.list_tools()
            tool_names = {tool.name for tool in listed.tools}
            assert {
                "search_papers",
                "ask_papers",
                "get_index_status",
            }.issubset(tool_names)

            search_result = _extract_json(
                await session.call_tool(
                    "search_papers",
                    {"query": "What is FedAvg?", "top_k": 1},
                )
            )
            assert search_result["retrieved_chunks"][0]["chunk_id"] == "fedavg_chunk_0"

            ask_result = _extract_json(
                await session.call_tool(
                    "ask_papers",
                    {"question": "What is FedAvg?", "top_k": 1},
                )
            )
            assert ask_result["answer"] == "FedAvg averages local client updates."
            assert ask_result["citations"] == ["fedavg_chunk_0"]

            status_result = _extract_json(
                await session.call_tool("get_index_status", {})
            )
            assert status_result["status"] == "ok"
            assert status_result["index_size"] == 1
