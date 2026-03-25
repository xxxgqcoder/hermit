
## 分层结构：

1. **Ingestion 层**：文件夹扫描 + 定期轮询 → 变更检测（SQLite）→ 文本切片 → 向量化 → 入库
2. **检索层 (Recall)**：Dense + Sparse 双路召回，Qdrant RRF 融合
3. **精排层 (Rerank)**：Cross-Encoder 处理 Top 50 候选集
4. **服务层**：FastAPI 暴露 OpenAPI 接口

支持**多知识库**：每个文件夹 = 一个独立的 Qdrant collection。

### 推理后端

使用 **fastembed**（基于 ONNX Runtime）作为统一推理后端。所有模型（embedding、sparse、reranker）均通过 fastembed 加载 ONNX 格式权重，CPU 推理，无需 PyTorch 或 GPU 依赖。

选型理由：
- 依赖轻量，无需安装 PyTorch（节省数 GB 磁盘和安装时间）
- ONNX Runtime 在 Apple Silicon 上具有良好的 CPU 推理性能
- 与 Qdrant 同生态，API 设计简洁
- 自动管理模型下载和缓存

**约束**：模型选择受限于 fastembed 支持的 ONNX 预转换模型列表。

---

## Ingestion Pipeline（写入路径）

### 数据源

- 指定文件夹作为知识库唯一来源
- 文件夹内文件统一视为文本文件处理（UTF-8, errors=replace）
- 多模态文件（PDF、图片等）的解析为文本格式由外部流程负责，不在本服务范围内
- 跳过隐藏文件（路径中含 `.` 开头的部分）

### 触发方式

- **启动时**：全量扫描，逐文件对比 hash 与 SQLite 记录，增量更新索引
- **运行时**：定期轮询扫描（默认每 15 分钟），检测文件变更并触发增量更新

### 变更检测（SQLite 元数据库）

每个知识库维护一个 SQLite 元数据库（存储在 `~/.hermit/data/metadata/{collection}.db`），记录已索引文件状态：

| 字段 | 类型 | 说明 |
|---|---|---|
| `file_path` (PK) | TEXT | 文件绝对路径 |
| `file_hash` | TEXT | 文件内容 SHA256 |
| `file_mtime` | REAL | 修改时间 |
| `chunk_count` | INTEGER | 该文件切片数 |
| `last_indexed_at` | REAL | 上次索引时间 |

变更检测仅对比 `file_hash`（SHA256），`file_mtime` 作为记录字段保留但不参与判定。

**选用 SQLite 的理由**：启动时对每个文件查 Qdrant payload 效率低；SQLite 单次全表扫描即可完成对比，天然适合存储关系型元数据。

### 文本切片

使用 embedding 模型自带的 tokenizer 按 token 数切片 + 滑动窗口重叠：

- 默认 `chunk_tokens=256` tokens，`overlap_tokens=32` tokens
- 使用模型 tokenizer 计数，消除中英文字符密度差异
- 短文本（≤ chunk_tokens）不做切分
- 空文本跳过
- **向量化增强**：Embedding 时会将文件名作为标题拼接到切片内容前，格式为 `[{title}]\n{chunk}`，以增强语义召回

### 索引流程

1. 递归扫描文件夹（`rglob("*")`），逐文件对比 SHA256 与 SQLite 记录
2. **新增/修改文件**：切片 → 向量化 → 按 `source_file` 删除旧 chunks → 插入新 chunks（UUID 作为 point ID）→ 更新 SQLite
3. 后台线程定期轮询执行扫描逻辑（Scanning Watcher中对应 chunks → 删除 SQLite 记录
4. watchdog 监听到文件事件时执行全量 rescan 逻辑（防抖后）

---

## 存储 Schema（Qdrant）

每个知识库 = 1 个 collection，使用 **named vectors**。Qdrant 运行在嵌入式模式（embedded），数据存储在 `~/.hermit/data/qdrant/`。

### 向量配置

| 向量名 | 维度 | 距离 | 说明 |
|---|---|---|---|
| `dense` | 768 | Cosine | jina-embeddings-v2-base-zh 语义向量 |
| `sparse` | 可变 | Dot | BM25 稀疏词权重向量 |

### Payload 结构

| 字段 | 类型 | 说明 |
|---|---|---|
| `title` | string | 文件名（不含扩展名） |
| `text` | string | 切片原文 |
| `source_file` | string | 源文件绝对路径 |
| `chunk_index` | int | 该切片在源文件中的位置（0-based） |
| `total_chunks` | int | 该源文件的总切片数 |

在 `source_file` 上建 **payload index**（KEYWORD 类型），确保按文件过滤/删除高效。

### 查询文件全部切片

通过 `source_file` payload filter 查询，配合 `chunk_index` 排序还原原文顺序。`total_chunks` 字段让调用方判断是否获取完整。

---

## 混合召回融合策略

使用 Qdrant 原生的 **Reciprocal Rank Fusion (RRF)**：

1. 使用 Qdrant `prefetch` + `query` API 进行双路召回
2. Dense prefetch 和 Sparse prefetch 各取 Top-N 候选（N = rerank_candidates，默认 50）
3. Qdrant 服务端执行 RRF 融合，合并去重
4. 融合后取 Top-N 候选送入 Reranker

API 中 `w_dense`/`w_sparse` 参数保留，当前未使用。

---

## 模型选型

### 模型清单

| 模型 | 用途 | 大小 | 说明 |
|---|---|---|---|
| `jinaai/jina-embeddings-v2-base-zh` | Dense Embedding (768 维) | ~0.64GB | 中英双语语义向量 |
| `Qdrant/bm25` | Sparse Embedding | <50MB | BM25 稀疏词权重 |
| `jinaai/jina-reranker-v2-base-multilingual` | Reranker (Cross-Encoder) | ~0.7GB | 多语言 reranker |

---

## 精排（Reranker）

| 维度 | 说明 |
|---|---|
| 模型 | `jinaai/jina-reranker-v2-base-multilingual` (~0.7GB) |
| 架构 | Cross-Encoder |
| 接口 | fastembed `TextCrossEncoder` |
| 语言 | 多语言（中英均适用） |

### 实现细节

- Reranker 对候选集重新排序，返回排序后的 **索引列表**
- 最终返回的 `score` 字段为 Qdrant RRF 融合分数，非 Reranker 分数

---

## 模型管理

### 设计原则

模型文件存放在 `~/.hermit/models/`，不随项目代码分发，通过 `.gitignore` 排除。

优势：
- 不污染用户全局环境（如 `~/.cache/huggingface`）
- 卸载即删目录，干净
- 多实例可以共存不互相干扰

### 路径管理

- 由 `config.py` 统一定义 `HERMIT_HOME`（默认 `~/.hermit/`，可通过 `HERMIT_HOME` 环境变量覆盖）
- `MODEL_ROOT = HERMIT_HOME / "models"`
- `DATA_ROOT = HERMIT_HOME / "data"`（含 Qdrant + SQLite 元数据）
- 所有模型加载均基于此路径，不在各模块中硬编码
- fastembed 的 `cache_dir` 参数指向 `MODEL_ROOT`

### 下载策略

- **服务启动时自动检测**：通过 fastembed 内部机制，模型缺失则自动下载
- `app.py` lifespan 中调用 `warmup()` 预加载 embedding 和 reranker 模型
- 也可通过 `hermit download` 命令提前下载

---

## 服务层（FastAPI）

单进程，模型启动时预加载。`app.py` 使用 `asynccontextmanager` lifespan 管理启动/关闭。知识库管理（注册、删除、更新）通过 CLI 完成，不暴露写操作 HTTP 端点。

### 检索流程

1. fastembed 编码 query → dense 向量 + sparse 向量
2. Qdrant prefetch 双路召回 → RRF 融合 → Top 50 候选
3. Cross-Encoder 对候选集精排
4. 返回 Top-K 结果（含切片原文、来源文件路径、切片位置、融合分数）

### 持久化与启动恢复

- **Collection 注册表持久化**：`~/.hermit/data/collections.json` 记录所有已注册知识库（folder_path、ignore_patterns、ignore_extensions）。服务启动时自动加载并恢复所有 collection，无需重新注册。
- **Watchdog 自动恢复**：服务启动时为每个已注册 collection 自动启动文件监听。
- **模型变更检测**：`~/.hermit/data/model_signature.json` 记录上次使用的 embedding 模型。若模型发生变更，启动时自动触发所有 collection 的全量重建索引。

---

## 内存预算（64GB）

| 组件 | 实际占用 |
|---|---|
| jina-embeddings-v2-base-zh (Dense) | ~640MB |
| Qdrant/bm25 (Sparse) | <50MB |
| jina-reranker-v2-base-multilingual (Reranker) | ~700MB |
| Qdrant（嵌入式） | ~100-500MB（取决于数据量） |
| 系统 + FastAPI + ONNX Runtime | ~500MB |
| **总计** | **~2-2.5GB** |

远低于 64GB 上限，无内存压力。

---

## 目录结构

```text
hermit/
├── main.py                    # 开发模式入口（uvicorn 直接运行）
├── pyproject.toml
├── hermit/
│   ├── app.py                 # FastAPI 应用 + lifespan（模型预加载、collection 恢复）
│   ├── cli.py                 # CLI 入口（hermit start/stop/kb/search/...）
│   ├── config.py              # 配置管理（HERMIT_HOME、MODEL_ROOT、DATA_ROOT、模型名、默认参数）
│   ├── models.py              # 模型下载与校验（huggingface_hub）
│   ├── ingestion/
│   │   ├── scanner.py         # 文件夹扫描 + 变更检测 + 索引
│   │   ├── watcher.py         # watchdog 实时监听（2s 防抖）
│   │   ├── chunker.py         # token 级文本切片 + overlap
│   │   └── task_queue.py      # 后台索引任务队列（线程池）
│   ├── retrieval/
│   │   ├── embedder.py        # Dense (TextEmbedding) + Sparse (SparseTextEmbedding)
│   │   ├── searcher.py        # Qdrant prefetch + RRF 融合 + rerank
│   │   └── reranker.py        # TextCrossEncoder
│   ├── storage/
│   │   ├── qdrant.py          # Qdrant 嵌入式客户端 + collection 管理
│   │   ├── metadata.py        # SQLite 元数据管理
│   │   ├── registry.py        # 知识库注册表（~/.hermit/data/collections.json）
│   │   └── model_signature.py # 模型变更检测（~/.hermit/data/model_signature.json）
│   └── api/
│       ├── routes.py          # API 路由
│       └── schemas.py         # Pydantic 请求/响应模型
├── models/                    # 开发时模型缓存（生产在 ~/.hermit/models/）
└── docs/
    └── design.md

~/.hermit/                     # 运行时数据（HERMIT_HOME，可通过环境变量覆盖）
├── models/                    # 模型文件（fastembed ONNX cache）
├── data/
│   ├── qdrant/                # Qdrant 嵌入式存储
│   ├── metadata/              # SQLite 元数据库（{collection}.db）
│   ├── collections.json       # 知识库注册表
│   └── model_signature.json   # 模型签名（变更检测）
├── logs/
│   └── hermit.log             # 服务日志
└── hermit.pid                 # 进程 PID 文件
```