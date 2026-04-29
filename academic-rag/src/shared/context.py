"""Shared application context factories."""

from src.rag.pipeline import RAGPipeline
from src.utils.config import load_config


def create_pipeline(config_path: str = "config.yaml") -> RAGPipeline:
    """Create and initialize the RAG pipeline from a config file."""
    config = load_config(config_path)
    return RAGPipeline(config)
