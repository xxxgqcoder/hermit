# Models

> 模型选型、下载管理与内存预算。

---

## 模型清单

| 模型 | 用途 | 大小 | 说明 |
|---|---|---|---|
| `jinaai/jina-embeddings-v2-base-zh` | Dense Embedding (768 维) | ~0.64GB | 中英双语语义向量 |
| `Qdrant/bm25` | Sparse Embedding | <50MB | BM25 稀疏词权重 |
| `jinaai/jina-reranker-v2-base-multilingual` | Reranker (Cross-Encoder) | ~0.7GB | 多语言 reranker |

所有模型通过 **fastembed**（基于 ONNX Runtime）加载，CPU 推理，无需 PyTorch 或 GPU 依赖。

**约束**：模型选择受限于 fastembed 支持的 ONNX 预转换模型列表。

---

## 路径管理

由 `config.py` 统一定义，所有模块基于此路径，不硬编码：

| 配置项 | 默认值 | 环境变量覆盖 |
|---|---|---|
| `HERMIT_HOME` | `~/.hermit/` | `HERMIT_HOME` |
| `MODEL_ROOT` | `~/.hermit/models/` | — |
| `DATA_ROOT` | `~/.hermit/data/` | — |

fastembed 的 `cache_dir` 参数指向 `MODEL_ROOT`。

模型文件不随项目代码分发，通过 `.gitignore` 排除，卸载即删目录，不污染全局缓存（如 `~/.cache/huggingface`）。

---

## 下载策略

- **服务启动时自动检测**：`app.py` lifespan 中调用 `warmup()` 预加载 embedding 和 reranker 模型，缺失则通过 fastembed 自动下载
- **提前下载**：`hermit download` 命令

---

## 模型变更检测

`~/.hermit/data/model_signature.json` 记录上次使用的 embedding 模型。若模型发生变更，启动时自动触发所有 collection 的全量重建索引。

**模块**：`hermit/storage/model_signature.py`

---

## 内存预算（参考：64GB 机器）

| 组件 | 实际占用 |
|---|---|
| jina-embeddings-v2-base-zh (Dense) | ~640MB |
| Qdrant/bm25 (Sparse) | <50MB |
| jina-reranker-v2-base-multilingual (Reranker) | ~700MB |
| Qdrant（取决于数据量） | ~100–500MB |
| 系统 + FastAPI + ONNX Runtime | ~500MB |
| **总计** | **~2–2.5GB** |

> **注意**：ONNX Runtime 在高并发搜索时（`SEARCH_THREADS` 个并发会话 × `ONNX_THREADS`）会放大内存占用。建议在内存受限环境下设置 `HERMIT_ONNX_THREADS=2 SEARCH_THREADS=2`。

---

## 相关模块

| 模块 | 说明 |
|---|---|
| `hermit/config.py` | `MODEL_ROOT`、`DATA_ROOT`、模型名、线程数等配置 |
| `hermit/models.py` | 模型下载与校验 |
| `hermit/storage/model_signature.py` | 模型变更检测 |
