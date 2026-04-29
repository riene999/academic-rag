## 目录结构

academic-rag/
├── config.yaml
├── main.py                    # FastAPI 服务入口
├── scripts/
│   └── index_papers.py        # 批量索引 PDF
├── src/
│   ├── rag/
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   ├── generator.py
│   │   └── pipeline.py
│   ├── agent/
│   │   └── agent.py
│   └── utils/
│       ├── config.py
│       └── pdf_parser.py
├── data/                      # FAISS 索引持久化目录
└── tests/
    └── test_rag.py

## 流程

PDF / Text Sources
        |
        v
Ingestion Entrypoints
scripts/ingest.py | scripts/index_papers.py | /upload | /index/text
        |
        v
RAGPipeline
        |
        +--> PDFParser ---> Document chunks + metadata
        |
        +--> Embedder ---> embeddings
        |
        +--> FAISSRetriever ---> FAISS index + documents.pkl
        |
        +--> BM25Retriever optional
        |
        v

Client Query
        |
        v
FastAPI main.py
        |
        +--> Direct RAG Path
        |       |
        |       v
        |   RAGPipeline.retrieve_chunks
        |       |
        |       +--> FAISS dense retrieval
        |       +--> BM25 sparse retrieval optional
        |       +--> RRF fusion optional
        |       +--> CrossEncoder rerank optional
        |       |
        |       v
        |   LLMGenerator ---> Answer + sources
        |
        +--> Agent Path
                |
                v
            PaperAgent
                |
                +--> ConversationMemory
                +--> LLM tool decision loop
                +--> search_papers / get_paper_overview
                |
                v
            RAGPipeline.retrieve_chunks
                |
                v
            Final Answer



