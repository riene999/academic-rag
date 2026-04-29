"""
Cache benchmark runner.

Usage:
  python scripts/cache_benchmark.py --dataset eval/eval_dataset.sample.jsonl
  python scripts/cache_benchmark.py --dataset eval/eval_dataset.sample.jsonl --repeats 5
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.eval.cache_benchmark import run_cache_benchmark, save_cache_benchmark_report
from src.utils.config import load_config


def _default_output_path() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"data/cache_benchmarks/cache_benchmark_{ts}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding/retrieval cache effectiveness")
    parser.add_argument("--dataset", required=True, help="Eval dataset path (.json/.jsonl)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--repeats", type=int, default=3, help="Repeat count per question")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Score threshold for retrieval during benchmark",
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Include answer generation in benchmark loop",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle workload queries before running benchmark",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output json path (default: timestamped file under data/cache_benchmarks)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    top_k = args.top_k if args.top_k is not None else config.retrieval.top_k
    output = args.output or _default_output_path()

    report = run_cache_benchmark(
        base_config=config,
        dataset_path=args.dataset,
        repeats_per_question=max(1, args.repeats),
        top_k=top_k,
        score_threshold=args.score_threshold,
        with_generation=args.with_generation,
        shuffle=args.shuffle,
    )
    path = save_cache_benchmark_report(report, output)

    print(
        json.dumps(
            {
                "workload_size": report.workload_size,
                "repeats_per_question": report.repeats_per_question,
                "speedup_cached_hot_vs_no_cache": report.speedup_cached_hot_vs_no_cache,
                "report_file": path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
