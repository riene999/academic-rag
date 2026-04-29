"""
CLI demo for the official src-based RAG pipeline.

Usage:
  python demo.py
"""
from pathlib import Path

from src.rag.pipeline import RAGPipeline
from src.utils.config import load_config


def main() -> None:
    config = load_config("config.yaml")
    pipeline = RAGPipeline(config)

    print("=" * 60)
    print("Academic RAG Demo (src version)")
    print("Commands:")
    print("  ingest <pdf_path>   index a PDF")
    print("  retrieve <query>    retrieve only")
    print("  stats               show current index size")
    print("  quit                exit")
    print("  other text          ask question")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "stats":
            print(f"Index size: {pipeline.retriever.index.ntotal}")
            continue

        if user_input.lower().startswith("ingest "):
            pdf_path = user_input[7:].strip()
            if not Path(pdf_path).exists():
                print(f"File not found: {pdf_path}")
                continue
            chunks = pipeline.index_documents_from_pdf(pdf_path)
            print(f"Indexed {chunks} chunks")
            continue

        if user_input.lower().startswith("retrieve "):
            query = user_input[9:].strip()
            chunks = pipeline.retriever.retrieve(
                query,
                top_k=config.retrieval.top_k,
                score_threshold=config.retrieval.score_threshold,
            )
            print(f"Retrieved {len(chunks)} chunks")
            for i, chunk in enumerate(chunks, start=1):
                source = chunk.document.metadata.get("source", "unknown")
                page = chunk.document.metadata.get("page", "?")
                preview = chunk.document.content[:160].replace("\n", " ")
                print(f"[{i}] {source} p.{page} score={chunk.score:.4f}")
                print(f"    {preview}...")
            continue

        response = pipeline.query(user_input)
        print(f"\nAssistant: {response.answer}")


if __name__ == "__main__":
    main()
