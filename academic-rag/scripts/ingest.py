import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF files into FAISS")
    parser.add_argument("--dir", default="data/papers", help="Directory containing PDFs")
    parser.add_argument("--file", default=None, help="Single PDF path")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(config)

    if args.file:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")
        count = pipeline.index_documents_from_pdf(str(pdf_path))
        print(f"Indexed {count} chunks from {pdf_path.name}")
        return

    pdf_dir = Path(args.dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"Directory not found: {pdf_dir}")

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    total = 0
    for pdf in pdf_files:
        count = pipeline.index_documents_from_pdf(str(pdf))
        total += count
        print(f"{pdf.name}: {count} chunks")

    print(f"Done. Total chunks indexed: {total}")


if __name__ == "__main__":
    main()
