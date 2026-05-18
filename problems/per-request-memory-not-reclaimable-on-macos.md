# 问题：per-request 增量内存在 macOS 上无法独立归还

## 背景

PR #38 把 ONNX Runtime 的 arena allocator 关掉之后，活动张量本应在每次推理
`free()` 后回到 OS。但实测：

- 单次 hybrid search 仍然让 RSS 增长 ~1-2 GB
- 连续多次 search 把 RSS 推到 7-8 GB 高位后**长期不下降**
- 一旦触发 reranker idle-unload（destroy 整个 `InferenceSession`），RSS 可瞬间跌 ~5 GB，几分钟后接近启动 baseline

直觉上的疑问：为什么"用完一次"不能立刻归还那部分活动张量？非要把模型整个干掉才肯松手？

## 实验：试图 per-request 强制归还

构造的方案：reranker 模型权重常驻（关掉 idle-unload，`HERMIT_RERANKER_IDLE_TIMEOUT=0`），
但在每次 `rerank()` 末尾调用 `malloc_zone_pressure_relief(NULL, 0)`——macOS 暴露的
"请求 libmalloc 把空闲页归还 OS"接口，理论上应该把刚 `free` 的 activation 池立刻
还回去。

代码片段（实验分支 `guoqing/per-request-malloc-trim`，未合并）：

```python
# hermit/retrieval/malloc_trim.py
libc = ctypes.CDLL("libc.dylib")
libc.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t

def trim() -> int:
    return libc.malloc_zone_pressure_relief(None, 0)
```

在 `reranker.rerank()` 末尾插一行 `released = _malloc_trim(); logger.info(...)`。

### 实测结果（4 次 search × 20 candidates rerank）

```
rerank done: 20 candidates, malloc_trim released 0 bytes
rerank done: 20 candidates, malloc_trim released 0 bytes
rerank done: 20 candidates, malloc_trim released 0 bytes
rerank done: 20 candidates, malloc_trim released 0 bytes
```

**每次都返回 0**。同期 RSS：

| | RSS | Physical footprint |
|---|---|---|
| 启动 baseline（reranker resident，未推理） | 1097 MB | 955 MB |
| 4 次 rerank 后 | 3224 MB | **3.0 GB** |
| 60 次 rerank 后（之前的大 probe） | 7832 MB | 7.4 GB |

`pressure_relief` 完全没拦住。

## 根本原因

macOS libmalloc 的归还策略**基于"释放量 / 持有量"的相对比例 + 未来分配预测**，
不是"调用了 API 就释放"。两条路径对比：

### 路径 A：单次 rerank 完成 + `pressure_relief`

- ONNX `free()` 了 ~500 MB activation 张量
- 但 reranker session 还在，挂着：
  - 700 MB 模型权重（mmap）
  - tokenizer / kernel registry / thread pool / op buffer 累计 ~几百 MB
- libmalloc 视角："总持有 7 GB，刚 free 500 MB，下次 rerank 大概率还要这块"
- 结论：留着复用，**`pressure_relief` 返回 0**

### 路径 B：`_reranker = None; gc.collect()`

- 不只 free activation——是把整个 `InferenceSession` 析构掉
- ONNX 析构链：
  1. `munmap()` 卸掉 mmap 的模型权重文件（~700 MB），**直接还给 kernel，绕过 libmalloc**
  2. `free()` 所有内部 buffer / tokenizer / kernel registry / thread pool 状态，累计几 GB
  3. activation 池一并 free
- libmalloc 视角："瞬间 free 了 ~6 GB，剩下工作集 ~500 MB，再留这么大空池无意义"
- 阈值翻转，**真还给 OS**

### 关键 takeaway

| 机制 | 能否触发 macOS 归还 |
|---|---|
| `malloc_zone_pressure_relief(NULL, 0)` | ❌ 几乎从不（API 是"请求"，libmalloc 可无视） |
| `malloc_trim(0)` | ⚠️ macOS 上是 glibc API，**不存在**；Linux 上有效 |
| Python `gc.collect()` | ❌ 不触发 C 层归还 |
| ONNX session 析构 | ✅ 触发 `munmap` + 大量 `free` 同时发生，libmalloc 阈值翻转 |
| 子进程退出 | ✅ 绝对干净（所有页面归 kernel） |

**hermit 现状下唯一能让 RSS 真实下降的机制是 destroy 整个 session**——这是
[reranker idle-unload](../hermit/retrieval/reranker.py) 设计上能成立的根本原因，
不是"等 5 min"那个表面行为，而是"找一个不影响 latency 的时机做 destruction"。

## 引申：dense embedder 同样的问题，量级不同

dense embedder 也走 ONNX session + libmalloc，没有 idle-unload 机制——所以
理论上 `free()` 出来的 activation 也滞留。但量级差一个数量级：

| 路径 | activation 量级 | 触发频率 | 累计上限 |
|---|---|---|---|
| Reranker forward（20 candidates） | 5-6 GB | 每次 hybrid/semantic search | 5-6 GB |
| Dense query embed（1 句） | 10-30 MB | 每次 search | <100 MB |
| Dense indexing batch（64 chunks） | 200-500 MB | 每次 reindex 文件 | **~1-2 GB** |

dense 的累计上限（~1-2 GB）被 reranker 那 5-6 GB 完全盖住。在"再压就影响召回质量"
的内存优化边界下（见 `project_memory_footprint_target.md`），dense 这部分是个
可接受的 tax，不主动处理。

## 可行的缓解

按 ROI 排序：

1. **维持现状**：reranker idle-unload 解决大头（5-6 GB），dense 累计 1-2 GB 接受
2. **dense 加 idle-unload，阈值 30 min**：减 dense 那 ~1 GB 累计，但 query path
   要付偶发冷启 0.2-0.5s。ROI 边界，不强推
3. **indexing 跑子进程**：dense 大头是 indexing 时积累的。改 task_queue 用
   subprocess 消费索引任务，task done 时进程退出，dense arena 跟进程一起归还。
   工程量较大（LanceDB 多进程写、IPC、metadata 协调）。1M chunks 的 deep-search
   目标场景下值得考虑
4. **per-request malloc trim**（本文实验）：**已验证无效**，不要再走这条

## 不要再做的事

- **不要写"per-request 释放" feature**：macOS 没接口能做，Linux 也只能做 partial
- **不要相信 vmmap "MALLOC_LARGE (empty)" 是"白占着"**：那些页面在内存压力下
  kernel 能回收，只是 ps/vmmap 视角看不见——属于 macOS 内存会计的视觉误导
  
## 验证脚本

复现实验：

```sh
# 1. 起 daemon，关 idle-unload
HERMIT_RERANKER_IDLE_TIMEOUT=0 hermit start

# 2. 在 reranker.rerank() 末尾加 trim + log（实验分支已写好）
# git checkout guoqing/per-request-malloc-trim

# 3. 跑几个 hybrid search
for q in "memory" "lancedb" "deep search" "ONNX"; do
  curl -s -X POST http://127.0.0.1:8000/search \
    -H 'Content-Type: application/json' \
    -d "{\"collection\":\"memory\",\"query\":\"$q\",\"top_k\":5}" > /dev/null
done

# 4. 看日志，预期每行都是 "released 0 bytes"
grep "malloc_trim released" ~/.hermit/logs/hermit.log

# 5. 看 RSS，预期 1 GB → 3 GB+，trim 没拦住
ps -p $(cat ~/.hermit/hermit.pid) -o rss=
```
