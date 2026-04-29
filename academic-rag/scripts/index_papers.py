"""
批量索引论文脚本
使用方式：python scripts/index_papers.py --pdf_dir ./papers/
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.rag.pipeline import RAGPipeline
from src.utils.config import load_config
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="批量索引PDF论文")
    parser.add_argument("--pdf_dir", type=str, required=True, help="PDF文件夹路径")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(config)

    pdf_dir = Path(args.pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"在 {pdf_dir} 中未找到PDF文件")
        return

    logger.info(f"找到 {len(pdf_files)} 个PDF文件")
    total_chunks = 0

    for pdf_path in pdf_files:
        try:
            chunks = pipeline.index_documents_from_pdf(str(pdf_path))
            total_chunks += chunks
            logger.info(f"✓ {pdf_path.name}: {chunks} chunks")
        except Exception as e:
            logger.error(f"✗ {pdf_path.name}: {e}")

    logger.info(f"索引完成！共 {total_chunks} 个chunks，来自 {len(pdf_files)} 个文件")


if __name__ == "__main__":
    main()
