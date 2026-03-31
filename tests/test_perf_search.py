"""Search latency benchmark.

Measures p50/p95/p99 latency across a set of representative queries.
Run with:
    uv run python tests/test_perf_search.py [--collection <name>] [--port <port>]

Outputs a summary table to stdout.
"""

import argparse
import json
import statistics
import time
import urllib.request

QUERIES = [
    "动态规划最优子结构",
    "滑动窗口最大值",
    "二分搜索旋转数组",
    "字符串解码嵌套括号",
    "最长公共子序列",
    "图的拓扑排序课程安排",
    "贪心算法跳跃游戏",
    "回溯法全排列组合",
    "前缀和二维矩阵",
    "单调栈下一个更大元素",
]

DEFAULT_COLLECTION = "profession"
DEFAULT_PORT = 8000
WARMUP_ROUNDS = 3
BENCH_ROUNDS = 5  # rounds × len(QUERIES) total requests


def search(query: str, collection: str, port: int) -> float:
    """Run one search and return wall-clock latency in milliseconds."""
    body = json.dumps({
        "query": query,
        "collection": collection,
        "top_k": 5,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/search",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000  # ms


def run_bench(collection: str, port: int, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Collection: {collection}  Port: {port}")
    print(f"{'='*60}")

    # Warm-up
    print(f"  Warming up ({WARMUP_ROUNDS} rounds)...", end="", flush=True)
    for _ in range(WARMUP_ROUNDS):
        for q in QUERIES:
            search(q, collection, port)
    print(" done")

    # Benchmark
    latencies: list[float] = []
    print(f"  Benchmarking ({BENCH_ROUNDS} rounds × {len(QUERIES)} queries)...", end="", flush=True)
    for _ in range(BENCH_ROUNDS):
        for q in QUERIES:
            latencies.append(search(q, collection, port))
    print(" done")

    latencies.sort()
    n = len(latencies)
    print(f"\n  Requests  : {n}")
    print(f"  Mean      : {statistics.mean(latencies):.1f} ms")
    print(f"  Median p50: {statistics.median(latencies):.1f} ms")
    print(f"  p95       : {latencies[int(n * 0.95)]:.1f} ms")
    print(f"  p99       : {latencies[int(n * 0.99)]:.1f} ms")
    print(f"  Min       : {min(latencies):.1f} ms")
    print(f"  Max       : {max(latencies):.1f} ms")
    return latencies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hermit search latency benchmark")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--label", default="benchmark")
    args = parser.parse_args()

    run_bench(args.collection, args.port, args.label)
