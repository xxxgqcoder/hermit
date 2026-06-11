# 检索路径设计

查询编码 → 召回 → 融合 → 精排 → 返回。主模块 `hermit/retrieval/searcher.py`，
底层存储/索引机制见 [`lancedb.md`](./lancedb.md)，模型服务见 [`inference.md`](./inference.md)。

## 搜索模式

`searcher.py` 暴露四种模式（`SEARCH_MODES`，默认 `hybrid`）：

| 模式 | 召回方式 | 融合/排序 |
|---|---|---|
| `hybrid` | dense 向量 + 原生 FTS 双路 | LanceDB `RRFReranker()` 融合 |
| `semantic` | 仅 dense 向量 | 向量距离 |
| `keyword` | 仅原生 FTS | BM25/FTS 分 |
| `fuzzy` | `text`/`filename` 上 SQL `contains()` 子串扫描 | 子串命中 |

模式正交于 [filename 过滤](#filename-过滤)，两者可叠加。

## 融合（RRF）

`hybrid` 内部并行跑向量召回与 FTS 召回，由 LanceDB 原生 `RRFReranker()` 按倒数排名融合：

```python
tbl.search((dense_vec, query_text), query_type="hybrid")
   .where(filter_sql, prefilter=True)       # filename 过滤，可选
   .limit(DEFAULT_RERANK_CANDIDATES)        # 默认 20
   .rerank(RRFReranker())
   .to_list()
```

`RRFReranker` 仅按排序融合，**不接受 per-query 权重**——相对 Qdrant 时代的 `w_dense`/`w_sparse` 是一处显式收窄（`SearchRequest` 已去掉这两个字段）。需要加权融合只能各自跑一遍再在 Python 侧拼接。

## filename 过滤

正交的文件名过滤（`filename` 列 = 去扩展名、小写化的 basename stem）：

- **普通子串** → SQL `contains(lower(filename), '<needle>')`，作为 prefilter 下推到 LanceDB。
- **glob 模式**（含 `*?[`）→ 全表扫描 + Python `fnmatch` 后过滤。

## 精排（Cross-Encoder）

融合/召回得到 Top-N（`DEFAULT_RERANK_CANDIDATES=20`）候选后，送 cross-encoder reranker 重排：

- reranker 返回排序后的索引列表，按其顺序重组候选。
- 最终 `score` 字段优先取 LanceDB 的 `_relevance_score`（hybrid+rerank 后的 RRF 分），其次 `_distance` 或 `_score`。
- reranker 的加载/量化/idle-unload 见 [`inference.md`](./inference.md)。

## 端到端检索流程

1. fastembed 编码 query → dense 向量（先查 embed cache，命中则跳过推理）。
2. 按模式召回：`hybrid` 走向量 + FTS 并 RRF 融合；其余走单路。可叠加 filename 过滤。
3. 取 Top-N 候选送 cross-encoder 精排。
4. 返回 Top-K（`DEFAULT_TOP_K=5`）结果：切片原文、来源文件路径、切片位置（`chunk_index`/`total_chunks`）、融合分数。

## 并发

搜索请求经 `app.py` 内 `ThreadPoolExecutor` 执行，池大小 `HERMIT_SEARCH_WORKERS`（**默认 1，串行**）。ONNX session 线程安全且推理时释放 GIL，agent 式 deep-search 可调高并发，但每个并发 run 会抬升 arena 峰值——需配合 reranker idle-unload 回收。代价测量见 `problems/concurrent-search-rss-blowup.md`。
