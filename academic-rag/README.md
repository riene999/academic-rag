# 学术论文 RAG 问答系统（正式版）

本项目是一个基于 `RAG + Agent` 的论文问答服务，当前仅保留 **`main.py + src/`** 这一套正式实现。

## 核心能力

- 上传 PDF 并自动切块、向量化、入库
- 基于 FAISS 的语义检索
- 使用 SQLite 持久化论文与 chunk 元数据
- 基于检索上下文的 LLM 问答
- 支持标准 RAG 与 Agent 多轮工具调用两种模式
- Agent 模式支持按 `session_id` 隔离的短期会话记忆
- 支持 SSE 流式返回

## 技术栈

- Web: FastAPI + Uvicorn
- Background Jobs: RQ + Redis
- Metadata Store: SQLite
- Embedding: `BAAI/bge-small-en-v1.5`
- 向量检索: FAISS (`IndexFlatIP`)
- LLM: OpenAI 兼容接口（DeepSeek/OpenAI/Qwen 等）

## 目录结构

```text
academic-rag/
├── config.yaml
├── main.py                    # FastAPI 服务入口
├── scripts/
│   ├── index_papers.py        # 批量索引 PDF
│   └── ingest.py              # 批量/单文件索引
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
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

推荐使用环境变量：

```bash
# Linux/macOS
export DS_API_KEY="your_api_key"

# Windows PowerShell
$env:DS_API_KEY="your_api_key"
```

服务只从环境变量读取 API Key，不要把真实密钥写入 `config.yaml`。

### 3. 启动 Redis

PDF 上传索引使用 RQ 后台任务，需要先启动 Redis。

```bash
redis-server
```

### 4. 启动服务

```bash
python main.py
```

启动后访问：`http://localhost:8011/docs`

### 5. 启动 RQ Worker

另开一个终端，确保同样设置了 `DS_API_KEY`，然后运行：

```bash
python scripts/rq_worker.py
```

## 索引论文

### 方式 A：批量索引目录

```bash
python scripts/index_papers.py --pdf_dir ./papers/
```

### 方式 B：使用 ingest 脚本

```bash
python scripts/ingest.py --dir ./papers/
# 或
python scripts/ingest.py --file ./your_paper.pdf
```

### 方式 C：通过 API 上传

```bash
curl -X POST http://localhost:8011/upload -F "file=@your_paper.pdf"
```

上传接口会立即返回 `job_id`，索引在 RQ Worker 中后台执行：

```bash
curl http://localhost:8011/jobs/<job_id>
```

查看已索引文档：

```bash
curl http://localhost:8011/documents
```

## 问答 API

### 外部评测接口

项目兼容 `agentic-eval-framework` 的黑盒 HTTP 接入协议：

```bash
curl -X POST http://localhost:8011/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "这篇论文的核心贡献是什么？", "stream": false, "top_k": 5, "case_id": "case_demo"}'
```

`/ask` 返回标准 JSON 字段：`answer`、`retrieved_chunks`、`citations`、`trace`、`latency_ms`。其中 `retrieved_chunks` 会包含稳定的 `chunk_id`、完整 `text`、`score`、`source` 和 `page`，用于外部框架计算 Recall@k、MRR 和引用一致性。

在外部评测项目中可这样自测：

```bash
python scripts/test_target_client.py --base-url http://localhost:8011 --question "测试问题"
python scripts/run_eval.py --base-url http://localhost:8011 --case-file examples/sample_cases.jsonl
```

### 标准 RAG

```bash
curl -X POST http://localhost:8011/query \
  -H "Content-Type: application/json" \
  -d '{"question": "这篇论文的核心贡献是什么？", "use_agent": false}'
```

### Agent 模式

```bash
curl -X POST http://localhost:8011/query \
  -H "Content-Type: application/json" \
  -d '{"question": "对比文中提到的不同方法优缺点", "use_agent": true, "session_id": "paper-chat-1"}'
```

同一个 `session_id` 会保留最近多轮用户问题和 Agent 最终回答，适合追问：

```bash
curl -X POST http://localhost:8011/query \
  -H "Content-Type: application/json" \
  -d '{"question": "它和FedAvg相比主要差异是什么？", "use_agent": true, "session_id": "paper-chat-1"}'
```

如需关闭本次记忆：

```bash
curl -X POST http://localhost:8011/query \
  -H "Content-Type: application/json" \
  -d '{"question": "重新总结这篇论文", "use_agent": true, "use_memory": false}'
```

清空某个会话记忆：

```bash
curl -X POST http://localhost:8011/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"session_id": "paper-chat-1"}'
```

### 流式问答

```bash
curl -N -X POST http://localhost:8011/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "总结这篇论文", "use_agent": false}'
```

## 配置说明（与 `src/utils/config.py` 对齐）

`config.yaml` 关键字段：

- `llm.provider / api_key / base_url / model / temperature / max_tokens`
- `embedding.model / device / batch_size`
- `retrieval.top_k / score_threshold / chunk_size / chunk_overlap`
- `vector_store.index_path / dimension`

## 备注

- 旧版 `rag/` 目录已移除，避免双实现并存带来的维护成本。
- 建议生产环境仅使用环境变量注入 `DS_API_KEY`。

## 评测闭环（新增）

项目已提供离线评测脚本：`scripts/evaluate.py`，用于固定评测集下的版本对比。

### 1. 准备评测集

示例文件：`eval/eval_dataset.sample.jsonl`

每行一个 JSON，字段：

- `question`（必填）：评测问题
- `expected_sources`（可选）：期望命中的论文文件名列表
- `expected_answer_keywords`（可选）：期望答案中出现的关键词列表
- `reference_answer`（可选）：参考答案（用于 `answer_f1`）
- `metadata`（可选）：自定义扩展字段

### 2. 运行评测（检索指标）

```bash
python scripts/evaluate.py --dataset eval/eval_dataset.sample.jsonl
```

默认输出到：`data/eval_reports/eval_YYYYMMDD_HHMMSS.json`
评测默认 `score_threshold=0.0`，用于观察召回上限；线上服务可使用更高阈值过滤低质量结果。

### 3. 运行评测（含生成指标）

```bash
python scripts/evaluate.py --dataset eval/eval_dataset.sample.jsonl --with-generation
```

该模式会额外计算答案指标，需可用的 LLM API。

### 4. 核心指标

- `retrieval_hit_rate`：Top-K 内是否命中预期来源
- `retrieval_mrr`：首个相关结果的倒数排名均值
- `avg_relevant_in_top_k`：Top-K 内相关片段平均数量
- `avg_answer_keyword_recall`：答案覆盖预期关键词的平均比例（可选）
- `avg_answer_f1`：答案与参考答案的 token-level F1（可选）

## 缓存测评（新增）

项目提供缓存专用基准测试：`scripts/cache_benchmark.py`。

用于对比三种模式：

- `no_cache`：关闭 query embedding 缓存和检索结果缓存
- `cache_cold`：开启缓存，首次跑工作负载
- `cache_hot`：开启缓存，再跑同一工作负载（热缓存）

运行示例：

```bash
python scripts/cache_benchmark.py \
  --dataset eval/eval_dataset.sample.jsonl \
  --repeats 3 \
  --score-threshold 0.0
```

输出报告：`data/cache_benchmarks/cache_benchmark_*.json`。

重点关注指标：

- 各模式 `latency.avg_ms / p95_ms`
- `embedding_cache.hit/miss` 与 `retrieval_cache.hit/miss`
- `speedup_cached_hot_vs_no_cache`（越高越好）
