# 推理后端与模型服务设计

dense embedding 与 reranker 的加载、量化、内存治理与下载。涉及模块：
`hermit/retrieval/embedder.py`、`reranker.py`、`fastembed_patch.py`、
`hermit/storage/quantizer.py`、`hermit/models.py`。

## 后端选型

使用 **fastembed**（基于 ONNX Runtime）作为 dense embedding 与 reranker 的推理后端，CPU 推理，无需 PyTorch 或 GPU。关键词检索由 LanceDB 原生 FTS 提供，**不加载 sparse embedding 模型**。

选型理由：
- 依赖轻量，免装 PyTorch（省数 GB 磁盘与安装时间）
- ONNX Runtime 在 Apple Silicon 上 CPU 推理性能良好
- 自动管理模型下载与缓存

**约束**：模型选择受限于 fastembed 支持的 ONNX 预转换模型列表。

## 模型清单

| 模型 | 用途 | 源大小 | 接口 |
|---|---|---|---|
| `jinaai/jina-embeddings-v2-base-zh` | Dense Embedding（768 维，中英双语） | ~0.64GB | fastembed `TextEmbedding` |
| `jinaai/jina-reranker-v2-base-multilingual` | Reranker（Cross-Encoder，多语言） | ~0.7GB | fastembed `TextCrossEncoder` |

均以 **INT8 量化 ONNX** 服务。模型名/维度定义在 `config.py`（`DENSE_MODEL`/`DENSE_DIM`/`RERANKER_MODEL`）。

## INT8 动态量化

`storage/quantizer.py` 在启动时用 `onnxruntime.quantization.quantize_dynamic`（`weight_type=QInt8`）对两个模型做**动态量化**，产物落 `~/.hermit/models/quantized/{repo_slug}/onnx/model.onnx`。`embedder.py` / `reranker.py` 加载时优先指向量化权重（fastembed 的 `specific_model_path`）；精度几乎无损，权重内存与磁盘显著降低。`app.py` lifespan 调 `ensure_quantized_models()` 确保产物就绪。

## ONNX Runtime 调优

ONNX Runtime 的 arena allocator 会把 `MALLOC_LARGE` 高水位逐步顶高且**不归还 OS**（只有销毁整个 InferenceSession 才释放——这正是 [idle-unload](#idle-unload双模型) 的依据）。因此默认关闭 arena。

| 配置 | 默认 | 说明 |
|---|---|---|
| `HERMIT_ONNX_ARENA` | `false` | arena 关闭；每次 `Run()` 走 plain malloc：单次慢几 %，但 RSS 在突发/累积索引下保持平直。设 `true` 恢复旧 fastembed 默认 |
| `HERMIT_ONNX_THREADS` | `2` | 每个 session 的 intra/inter-op 线程数。ORT 保留 per-thread arena，多线程按数十 MB 抬升常驻内存而延迟收益有限 |

arena/mem-pattern 开关由 `fastembed_patch.py` 给 fastembed 暴露 `enable_cpu_mem_arena` / `enable_mem_pattern` 注入点实现。背景测量见 `problems/concurrent-search-rss-blowup.md`、`problems/dense-embedder-arena-creep.md`。

## Idle-Unload（双模型）

两个 ONNX session 各有后台 daemon 线程，闲置超阈值后销毁实例并 `gc.collect()`，把 session 持有的内存整块还给 OS（arena 关闭前提下，这是回收 `MALLOC_LARGE` 高水位的主要手段）。下次请求懒重载。

| 模型 | 闲置阈值 env | 默认 | 检查频率 env | 默认 | 冷启动 |
|---|---|---|---|---|---|
| Reranker | `HERMIT_RERANKER_IDLE_TIMEOUT` | `300`s（5min） | `HERMIT_RERANKER_IDLE_CHECK_INTERVAL` | `60`s | ~1-3s |
| Dense | `HERMIT_DENSE_IDLE_TIMEOUT` | `1800`s（30min） | `HERMIT_DENSE_IDLE_CHECK_INTERVAL` | `120`s | ~0.2-0.5s |

阈值差异：dense 被每次搜索（query 编码）和每个索引批次触碰，工作时段基本常热，30 分钟阈值只在真正安静的间隙（隔夜、突发之间）触发以回收索引累积的激活池；reranker 仅搜索时用，5 分钟即可快速回收。任一阈值设 `0`（或负）关闭对应卸载。整体内存预算见 [`design.md`](./design.md#内存预算)。

## 精排（Reranker）作为查询步骤

- 架构 Cross-Encoder，对候选集重排，返回排序后的**索引列表**。
- 候选数 `DEFAULT_RERANK_CANDIDATES=20`。
- 在检索路径中的位置与打分见 [`retrieval.md`](./retrieval.md)。

## 模型管理与下载

设计原则：模型存 `~/.hermit/models/`，不随代码分发（`.gitignore` 排除）。好处——不污染用户全局环境（如 `~/.cache/huggingface`）、卸载即删目录、多实例互不干扰。

- 路径由 `config.py` 统一定义：`MODEL_ROOT = HERMIT_HOME/"models"`（含原始权重 + `quantized/`），fastembed `cache_dir` 指向它。
- 下载：`models.py` 经 `huggingface_hub.snapshot_download` 拉取；启动时自动检测缺失并下载，随后 `ensure_quantized_models()` 产出 INT8 版本。
- `app.py` lifespan 调 `warmup()` 预加载两个模型；也可 `hermit download` 提前下载。
- **模型变更检测**：`~/.hermit/data/model_signature.json` 记录上次 dense 模型，变更则启动时触发所有 collection 全量重建。
