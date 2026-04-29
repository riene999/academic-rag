import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

from src.eval.evaluator import load_eval_dataset
from src.rag.pipeline import RAGPipeline
from src.utils.config import AppConfig


@dataclass
class CacheStatsSnapshot:
    hit: int
    miss: int
    evicted: int
    expired: int


@dataclass
class LatencyMetrics:
    count: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class BenchmarkPassResult:
    mode: str
    latency: LatencyMetrics
    embedding_cache: CacheStatsSnapshot
    retrieval_cache: CacheStatsSnapshot


@dataclass
class CacheBenchmarkReport:
    created_at: str
    dataset_path: str
    workload_size: int
    repeats_per_question: int
    with_generation: bool
    top_k: int
    score_threshold: float
    results: List[BenchmarkPassResult]
    speedup_cached_hot_vs_no_cache: float


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round((len(sorted_values) - 1) * q))
    return sorted_values[idx]


def _latency_metrics(latencies_sec: List[float]) -> LatencyMetrics:
    if not latencies_sec:
        return LatencyMetrics(count=0, avg_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0)

    sorted_vals = sorted(latencies_sec)
    avg_ms = sum(latencies_sec) / len(latencies_sec) * 1000
    return LatencyMetrics(
        count=len(latencies_sec),
        avg_ms=round(avg_ms, 3),
        p50_ms=round(_percentile(sorted_vals, 0.50) * 1000, 3),
        p95_ms=round(_percentile(sorted_vals, 0.95) * 1000, 3),
        p99_ms=round(_percentile(sorted_vals, 0.99) * 1000, 3),
    )


def _snapshot_cache_stats(pipeline: RAGPipeline) -> Tuple[CacheStatsSnapshot, CacheStatsSnapshot]:
    embed_cache = getattr(pipeline.embedder, "query_cache", None)
    retr_cache = getattr(pipeline.retriever, "result_cache", None)

    embed_stats = (
        CacheStatsSnapshot(
            hit=embed_cache.stats.hit,
            miss=embed_cache.stats.miss,
            evicted=embed_cache.stats.evicted,
            expired=embed_cache.stats.expired,
        )
        if embed_cache is not None
        else CacheStatsSnapshot(hit=0, miss=0, evicted=0, expired=0)
    )
    retr_stats = (
        CacheStatsSnapshot(
            hit=retr_cache.stats.hit,
            miss=retr_cache.stats.miss,
            evicted=retr_cache.stats.evicted,
            expired=retr_cache.stats.expired,
        )
        if retr_cache is not None
        else CacheStatsSnapshot(hit=0, miss=0, evicted=0, expired=0)
    )
    return embed_stats, retr_stats


def _diff_stats(after: CacheStatsSnapshot, before: CacheStatsSnapshot) -> CacheStatsSnapshot:
    return CacheStatsSnapshot(
        hit=after.hit - before.hit,
        miss=after.miss - before.miss,
        evicted=after.evicted - before.evicted,
        expired=after.expired - before.expired,
    )


def _run_pass(
    pipeline: RAGPipeline,
    queries: List[str],
    top_k: int,
    score_threshold: float,
    with_generation: bool,
    mode: str,
) -> BenchmarkPassResult:
    before_embed, before_retr = _snapshot_cache_stats(pipeline)
    latencies_sec: List[float] = []

    for query in queries:
        start = perf_counter()
        chunks = pipeline.retrieve_chunks(
            question=query,
            top_k=top_k,
            score_threshold=score_threshold,
            use_reranker=True,
        )
        if with_generation:
            _ = pipeline.generator.generate(query, chunks)
        latencies_sec.append(perf_counter() - start)

    after_embed, after_retr = _snapshot_cache_stats(pipeline)
    return BenchmarkPassResult(
        mode=mode,
        latency=_latency_metrics(latencies_sec),
        embedding_cache=_diff_stats(after_embed, before_embed),
        retrieval_cache=_diff_stats(after_retr, before_retr),
    )


def _build_workload(dataset_path: str, repeats_per_question: int, shuffle: bool) -> List[str]:
    samples = load_eval_dataset(dataset_path)
    queries = [s.question for s in samples for _ in range(repeats_per_question)]
    if shuffle:
        random.shuffle(queries)
    return queries


def run_cache_benchmark(
    base_config: AppConfig,
    dataset_path: str,
    repeats_per_question: int,
    top_k: int,
    score_threshold: float,
    with_generation: bool = False,
    shuffle: bool = False,
) -> CacheBenchmarkReport:
    workload = _build_workload(dataset_path, repeats_per_question, shuffle=shuffle)

    # A: no-cache baseline
    cfg_no_cache = deepcopy(base_config)
    cfg_no_cache.embedding.query_cache_enabled = False
    cfg_no_cache.retrieval.result_cache_enabled = False
    pipeline_no_cache = RAGPipeline(cfg_no_cache)
    baseline = _run_pass(
        pipeline=pipeline_no_cache,
        queries=workload,
        top_k=top_k,
        score_threshold=score_threshold,
        with_generation=with_generation,
        mode="no_cache",
    )

    # B/C: cache enabled, cold then hot
    cfg_cache = deepcopy(base_config)
    cfg_cache.embedding.query_cache_enabled = True
    cfg_cache.retrieval.result_cache_enabled = True
    pipeline_cache = RAGPipeline(cfg_cache)
    cached_cold = _run_pass(
        pipeline=pipeline_cache,
        queries=workload,
        top_k=top_k,
        score_threshold=score_threshold,
        with_generation=with_generation,
        mode="cache_cold",
    )
    cached_hot = _run_pass(
        pipeline=pipeline_cache,
        queries=workload,
        top_k=top_k,
        score_threshold=score_threshold,
        with_generation=with_generation,
        mode="cache_hot",
    )

    baseline_avg = baseline.latency.avg_ms if baseline.latency.avg_ms > 0 else 1.0
    speedup = round(baseline_avg / max(cached_hot.latency.avg_ms, 1e-6), 3)

    return CacheBenchmarkReport(
        created_at=datetime.utcnow().isoformat() + "Z",
        dataset_path=str(Path(dataset_path).resolve()),
        workload_size=len(workload),
        repeats_per_question=repeats_per_question,
        with_generation=with_generation,
        top_k=top_k,
        score_threshold=score_threshold,
        results=[baseline, cached_cold, cached_hot],
        speedup_cached_hot_vs_no_cache=speedup,
    )


def save_cache_benchmark_report(report: CacheBenchmarkReport, output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)
    return str(path.resolve())

