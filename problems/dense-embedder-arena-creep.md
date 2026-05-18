# 问题：dense embedder ONNX arena 随 indexing 累积，无 idle unload 回收

## 现象

观察自 2026-05-18 一台连续运行 ~38 小时的 hermit daemon（PID 45517，2026-05-16
19:51:30 启动，`HERMIT_SEARCH_WORKERS=2`）。本意是观察 burst 过后的稳态内存，
却发现：

- Reranker **5/16 19:58:31 之后再没被加载过**（日志零条 `Reranker model loaded`）
- 期间只有 **3 次后台 re-index 同一个文件**（`0041_Probability.md`，294 chunks）
- **没有 hybrid/semantic 类的搜索请求触发过 reranker**
- 然而 RSS 从 5/16 19:58 的 999 MB 一路爬到 5/18 09:38 的 **5.07 GB**

`vmmap --summary` 视角：

| | 当前 (5/18 09:38) | 5/16 19:58 idle 后 |
|---|---|---|
| Physical footprint | **4.5 GB** | ~1.0 GB |
| Peak footprint | 10.1 GB（5/16 burst 留下） | 10.1 GB |
| MALLOC_LARGE resident | **3.3 GB**（67 regions，其中 **19 块 128 MB super-block**） | ~300 MB |
| MALLOC_LARGE (empty) | 836 MB（allocator 已 free 未还 OS） | small |
| MALLOC_SMALL | 436 MB | ~400 MB |
| shared libs (TEXT+LINKEDIT+OBJC) | ~545 MB | ~545 MB |

整 ~4 GB 增量来自 MALLOC_LARGE 累积，**不是来自 reranker**——reranker session 在
内存里压根不存在。

## 根本原因

[hermit/retrieval/reranker.py](../hermit/retrieval/reranker.py) 实现了 idle-unload
机制（默认 5 min 无活动就 `del _reranker; gc.collect()`，把整个 ONNX session
析构，arena 一并归还 OS）。这条路径在该 daemon 上观测到生效了 8 次。

[hermit/retrieval/embedder.py](../hermit/retrieval/embedder.py) 的 dense embedder
**没有对应机制**。`_dense_model: TextEmbedding | None = None` 单例一旦在
`warmup()` 时加载就常驻整个 daemon 生命周期。每次新的推理（indexing 一批 chunks
或 query embed）都会让 ONNX Runtime 的 arena allocator 高位上探，**arena 涨上去
不归还**——这是 ONNX 的 `enable_cpu_mem_arena=True` 默认行为。

具体到这次观察：
- 单次 indexing 294 chunks（jina-embeddings-v2-base-zh，hidden=768，token 序列
  256）按 batch_size=64 跑 ~5 batches
- 每个 batch 的活动张量在 transformer 12 层间累积，arena 高位大致 ~200-500 MB
- 但 dense session 是常驻共享的，3 次 indexing 跑下来 arena 会被推到几个并发
  forward 路径的 max 值之和，最终高位可能锁定在 **2-4 GB**

reranker 不在场也能涨到 4 GB，是因为：
- dense embedder 的活动张量比 reranker 的小，但仍是百 MB 级
- 3 次 indexing 之间没有清空机制
- 19 块 128 MB super-block 是 ONNX/malloc 分配大块的典型形态

## 触发路径

```
hermit start                              # warmup() 加载 dense embedder
   ↓
indexing 任务（startup scan 或 watcher poll）→ embed_dense(batch)
   ↓
ONNX dense session.Run() 分配活动张量
   ↓
MALLOC_LARGE arena 涨高位（不归还）
   ↓
任务结束，session 仍持有；下次 indexing 高位再上探
   ↓
N 次 indexing 后，arena 锁定在 max(activation_set) × 各路径并发因子
   ↓
（reranker 即便 idle-unload 把自己那 ~1 GB 还了，dense 这部分仍占着）
```

## 实际暴露的影响

按这次速率，**3 次 294-chunk 重建 → 累计涨 ~4 GB**。外推：

- 百万级 chunk 全量初次索引（hermit 当前目标），dense arena 可能锁定在 **6-10 GB**
- 与单次 burst 测试时的"reranker arena 爆炸"是同等量级问题，但触发面更广：
  reranker 只在 hybrid/semantic 模式才触发；dense 在 indexing 和所有 query embed
  路径都触发
- 不像 reranker 有 5 min idle-unload 兜底，dense 的高位**只能 daemon 重启清零**
- 配合现在的 100k+ chunks 用户库已经能看到效应；deep-search 的 1M chunks 目标下
  这将是最大的内存"暗债"

## 诊断方法

1. 看 reranker 是否在场（应该不在）：
   ```sh
   grep "Reranker model loaded" ~/.hermit/logs/hermit.log | tail -5
   # 若最近一条远早于现在，且 RSS 仍很高 → dense arena 累积问题
   ```

2. MALLOC_LARGE 高位区域计数：
   ```sh
   vmmap $(cat ~/.hermit/hermit.pid) | grep "^MALLOC_LARGE " | grep -v "empty\|reserved" \
     | awk '$4=="128.0M"' | wc -l
   # 几个 128 MB super-block 数量大致反映了 ONNX arena 的当前形态
   ```

3. 关联到 indexing 频率：
   ```sh
   grep "Indexed .*chunks)" ~/.hermit/logs/hermit.log | wc -l
   # 配合 daemon uptime 看每次 indexing 的平均 chunk 数与 RSS 增长趋势
   ```

## 缓解策略

按 ROI 排序：

1. **给 dense embedder 加 idle unload**（最对症，工作量最小）——直接复用
   `hermit/retrieval/reranker.py` 现有的 `start_idle_unloader` 模式。注意 dense
   用得比 reranker 频繁，阈值得更大（建议 15-30 min，单独 env var 控制），冷
   启动延迟也更明显（dense + tokenizer + ONNX 编译大概 1-3s）。配套：query path
   要小心，频繁单次 query 会反复触发卸载/重载

2. **关 ONNX arena allocator**（治根，但代价大）——`enable_cpu_mem_arena=False`
   + `enable_mem_pattern=False`，分配走普通 malloc，arena 高位问题彻底消失。
   单次推理慢 10-20%。需 fork/patch fastembed（同 `concurrent-search-rss-blowup.md`
   方案 C）。**优势**：同一改动顺手解决 reranker 并发 burst 内存爆炸问题

3. **限制 indexing 批大小**（无效）——arena 高位取决于单批活动张量峰值，更小
   批只会让吞吐变差不会让高位变低。不要走这条

## 附录：3 次 re-index 是合预期的——用户在编辑该文件

观察期里 `0041_Probability.md` 被 indexing 三次（chunk 数 `294 → 294 → 293`）
**不是 bug**，是用户当时正在编辑该文件，SHA256 hash 每次都变了，scanner 按设计
重新切片+嵌入。hermit 的 mtime 跳过 + hash 比较逻辑工作正常：

```
2026-05-16 20:06:57  Indexed (294 chunks)    ← 用户编辑保存 → hash 变 → re-index
2026-05-16 21:06:56  Indexed (294 chunks)    ← 又编辑（chunk 数恰好相同）
2026-05-18 09:38:05  Indexed (293 chunks)    ← 36h 后再编辑（chunk 数减 1）
```

所以这条路径对内存观察的意义是：**正常使用中"被频繁编辑的大文件"会持续给
dense embedder 投活，加速 arena 高位攀升**。换句话说，dense 缺 idle-unload 这个
窟窿在真实使用场景里是被合理的用户行为踩出来的，不需要构造极端 workload。
