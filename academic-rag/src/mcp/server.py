"""FastMCP stdio server for the academic RAG tools."""

import json
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from src.mcp.tools import ask_papers as ask_papers_tool
from src.mcp.tools import get_index_status as get_index_status_tool
from src.mcp.tools import search_papers as search_papers_tool
from src.shared.context import create_pipeline


mcp = FastMCP("academic-rag")


@lru_cache(maxsize=1)
def get_pipeline():
    return create_pipeline("config.yaml")
# 创建pipeline

def _json_result(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool(
    name="search_papers",
    description="Search indexed academic papers and return relevant evidence chunks.",
)
def search_papers(
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
    use_reranker: bool = True,
) -> str:
    return _json_result(
        search_papers_tool(
            get_pipeline(),
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            use_reranker=use_reranker,
        )
    )


@mcp.tool(
    name="ask_papers",
    description="Answer a question using the indexed academic paper corpus.",
)
def ask_papers(question: str, top_k: int | None = None) -> str:
    return _json_result(ask_papers_tool(get_pipeline(), question=question, top_k=top_k))


@mcp.tool(
    name="get_index_status",
    description="Return index size and document count for the loaded paper corpus.",
)
def get_index_status() -> str:
    return _json_result(get_index_status_tool(get_pipeline()))


if __name__ == "__main__":
    mcp.run(transport="stdio")
