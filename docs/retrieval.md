# Retrieval

> 检索路径：双路召回 → RRF 融合 → Cross-Encoder 精排

---

## 混合召回（Hybrid Recall）

使用 Qdrant 原生的 **Reciprocal Rank Fusion (RRF)**：

1. 使用 Qdrant `prefetch` + `query` API 进行双路召回
2. Dense prefetch 和 Sparse prefetch 各取 Top-N 候选（N = `rerank_candidates`，默认 50）
3. Qdrant 服务端执行 RRF 融合，合并去重
4. 融合后取 Top-N 候选送入 Reranker

> API 中 `w_dense`/`w_sparse` 参数保留，当前未使用。

---

## 精排（Reranker）

| 维度 | 说明 |
|---|---|
| 模型 | `jinaai/jina-reranker-v2-base-multilingual` (~0.7GB) |
| 架构 | Cross-Encoder |
| 接口 | fastembed `TextCrossEncoder` |
| 语言 | 多语言（中英均适用） |

### 实现细节

- Reranker 对候选集重新排序，返回排序后的**索引列表**
- 最终返回的 `score` 字段为 Qdrant RRF 融合分数，非 Reranker 分数

---

## 检索流程

```
query
  │
  ├─ fastembed → dense 向量
  └─ fastembed → sparse 向量
          │
          ▼
  Qdrant prefetch（dense Top-50 + sparse Top-50）
          │
          ▼
  RRF 融合（服务端）→ Top-50 候选
          │
          ▼
  Cross-Encoder 精排
          │
          ▼
  返回 Top-K 结果
```

结果包含：切片原文、来源文件路径、切片位置（`chunk_index`/`total_chunks`）、RRF 融合分数。

---

## 相关模块

| 模块 | 说明 |
|---|---|
| `hermit/retrieval/embedder.py` | Dense (`TextEmbedding`) + Sparse (`SparseTextEmbedding`) |
| `hermit/retrieval/searcher.py` | Qdrant prefetch + RRF 融合 + rerank 编排 |
| `hermit/retrieval/reranker.py` | `TextCrossEncoder` 封装 |
