# 问题：搜索请求并发执行导致 RSS 峰值爆炸

## 背景

`hermit/app.py` 把搜索请求路由到一个 `ThreadPoolExecutor`，`_SEARCH_WORKERS` 控制
并发上限。历史上默认 `1`（串行），原注释说"raises memory pressure without enough
benefit"。

为支持 deep search agent 的 fan-out 模式（一次发起几十个并行查询），2026-05-16
做了一组实测，把 `HERMIT_SEARCH_WORKERS` 改成环境变量驱动并对比 `1/2/4` 三档。
**结论：单纯调高 WORKERS 是用大幅内存换有限吞吐**，原注释的论断在 LanceDB 切换后
依然成立，但峰值幅度比预想严重得多。

## 现象

环境：M5 Pro 14"，本地两个 collection（`memory` 220/2229 chunks，`Profession`
496/32338 chunks），客户端 40 reqs × concurrency=8 一次 burst。

| 指标 | WORKERS=1 (估) | WORKERS=2 | WORKERS=4 |
|---|---|---|---|
| Baseline RSS | ~1.1 GB | 1.10 GB | 1.18 GB |
| **峰值 RSS** | ~3-4 GB | **10.5 GB** | **15.0 GB** |
| Idle 后 RSS | ~0.8 GB | 1.00 GB | 1.18 GB |
| Wall time (40 reqs) | ~60s | 54.1s | 33.8s |
| 吞吐 | 0.67 req/s | 0.74 req/s | 1.18 req/s |
| p50 latency | ~1.5s | 8.9s | 5.7s |
| p99 latency | — | 17.5s | 11.1s |

峰值大致 `peak ≈ 3.5 × WORKERS` GB，呈线性增长。WORKERS=2 的吞吐只比串行多 10%
就要付出 9 倍内存；WORKERS=4 拿到 1.8 倍吞吐但峰值打到 15 GB——在个人 16-32 GB
机器上几乎稳定 swap。

### RSS 轨迹（WORKERS=4 burst-phase）

```
T+0:    1175 MB   ← baseline，reranker 已 idle-unloaded
T+8:    2696 MB   ← 第一批 4 并发命中 reranker，arena 开始扩
T+14:  12061 MB   ← 大跃迁，4 条并发 forward 同时开活动张量
T+25:  14933 MB   ← 高位坐稳
T+45 ~ T+260s:  ~14966 MB（不归还）
T+266s: 1183 MB   ← reranker idle-unload 把整个 session 析构，arena 跟着回收
```

WORKERS=2 是同样模式，只是峰值低些：从 ~1.1 GB 在 60s 内爬到 10.5 GB，稳态后
等 idle unload 全部归还。

## 根本原因

每多一个并发 worker 跑 reranker，ONNX Runtime 会：

1. **为每条 forward 路径开独立的 activation 张量**——cross-encoder 一次跑 20
   candidates 大约 1-2 GB 中间张量，4 并发就是 4× 同时常驻
2. **arena allocator 高位锁定**——ONNX 默认 `enable_cpu_mem_arena=True`，arena
   涨到峰值后**不归还 OS**，直到整个 `InferenceSession` 析构。所以 burst 结束
   不能立刻降回去，只能等 reranker idle unload 一并清理
3. **per-thread tokenizer / padding buffer 翻倍**——`WORKERS × ONNX_THREADS`
   是 ONNX 实际工作线程数，每条多吃几十 MB

线性增长系数大概是 `arena_peak ≈ activation_per_session × WORKERS + 常驻`。

## 触发路径

```
HERMIT_SEARCH_WORKERS=4 hermit start
   ↓
agent / 客户端 fan-out 一次发 N 个搜索请求
   ↓
ThreadPoolExecutor(4) 同时跑 4 条 search()
   ↓
4 条同时调 reranker.rerank(query, 20 candidates)
   ↓
ONNX session 接到 4 个并发 Run()，分配 4× activation
   ↓
MALLOC_LARGE arena 在 30-60s 内爬到 ~3.5 × WORKERS GB
   ↓
burst 结束，arena 不归还（坐稳直到 idle unload）
   ↓
5 min 后 reranker idle unload，arena 一次性归还 OS
```

## 诊断方法

1. 启动日志看 worker 数：
   ```sh
   grep "Search executor" ~/.hermit/logs/hermit.log
   # parallel (4 workers)  → 非默认
   # serialized (1 worker) → 默认
   ```

2. burst 期间 RSS 采样：
   ```sh
   PID=$(cat ~/.hermit/hermit.pid)
   while true; do
     printf "%.3f %s\n" $(date +%s.%N) $(ps -p $PID -o rss= | tr -d ' ')
     sleep 0.25
   done
   ```

3. arena 区域分布：
   ```sh
   vmmap $(cat ~/.hermit/hermit.pid) | grep -E "^MALLOC_LARGE " | grep -v "empty|reserved" | awk '{print $4}' | sort -rn | head
   ```
   单条 128 MB 的 slab 数量 ≈ reranker 活跃并发数 × 9-15 块。

## 缓解策略

**短期**：保持默认 `HERMIT_SEARCH_WORKERS=1`。该环境变量已开放给愿意承担内存
代价的用户，**不建议生产路径打开**。

**长期推荐路径（按 ROI 排序）**：

1. **agent 侧"宽召回 + 末段精排"**：deep search fan-out 第一波用 `--mode keyword`
   或 `--mode hybrid --no-rerank`，**完全绕过 reranker**——并发 search 不触发
   arena 爆炸（dense embedder 的 arena 比 reranker 小一个数量级）。最后只对收
   敛后的 top 10-20 候选跑一次 rerank。这条路完全不动 hermit 内部，只是 agent
   prompt / skill 文档要补充用法

2. **hermit 加 `POST /search/batch`**：一次收 N 个 query 一起回。内部把所有
   candidates 合并送 reranker **一次 batch**——reranker 跑 100-200 candidates
   一次比跑 N×20 candidates N 次内存更省、CPU 更高效，因为 arena 高位只爬一次

3. **monkey-patch ONNX SessionOptions 关 arena**：`enable_cpu_mem_arena=False`
   + `enable_mem_pattern=False`，让分配走普通 malloc。代价是单次推理慢 10-20%，
   收益是 arena 高位问题彻底消失。需要 fork/patch fastembed（它把 SessionOptions
   写死了）。详细路径见 `design/memory-footprint-analysis.md` 方案 C

## 预防措施

- 默认值留 `1`，env var 仅作为 escape hatch
- `Search executor: parallel (N workers)` 日志能让用户立刻发现自己开了并发模式
- README / SKILL.md 不主动宣传 `HERMIT_SEARCH_WORKERS`——避免新手一上来就开 4 然后内存炸

## 复现脚本

`HERMIT_SEARCH_WORKERS=N` 启动 hermit，然后跑（参考 `/tmp/burst_search.py` 的形态）：

```python
from concurrent.futures import ThreadPoolExecutor
import urllib.request, json, time

def hit(q):
    body = json.dumps({"collection":"memory","query":q,"top_k":5}).encode()
    t0 = time.monotonic()
    urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/search", data=body,
        headers={"Content-Type":"application/json"}, method="POST"
    ), timeout=60).read()
    return time.monotonic() - t0

queries = ["...50 varied queries..."] * 1
t0 = time.monotonic()
with ThreadPoolExecutor(max_workers=8) as ex:
    latencies = list(ex.map(hit, queries[:40]))
wall = time.monotonic() - t0
```

同时后台采样 `ps -p $PID -o rss=`，比较 baseline / 峰值 / idle 后 RSS。
