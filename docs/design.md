
分层结构：

1. **Ingestion 层**：文件夹扫描 + watchdog 监听 → 变更检测（SQLite）→ 文本切片 → 向量化 → 入库
2. **检索层 (Recall)**：Dense + Sparse 双路召回，Qdrant RRF 融合
3. **精排层 (Rerank)**：Cross-Encoder 处理 Top 20-30 候选集
4. **服务层**：FastAPI 暴露 OpenAPI 接口

支持**多知识库**：每个文件夹 = 一个独立的 Qdrant collection。

### 推理后端

使用 **fastembed**（基于 ONNX Runtime）作为统一推理后端。所有模型（embedding、sparse、reranker）均通过 fastembed 加载 ONNX 格式权重，CPU 推理，无需 PyTorch 或 GPU 依赖。

选型理由：
- 依赖轻量，无需安装 PyTorch（节省数 GB 磁盘和安装时间）
- ONNX Runtime 在 Apple Silicon 上具有良好的 CPU 推理性能
- 与 Qdrant 同生态，API 设计简洁
- 自动管理模型下载和缓存

**约束**：模型选择受限于 fastembed 支持的 ONNX 预转换模型列表。原设计中 BGE-M3 和 bge-reranker-v2-m3 均不在 fastembed 0.7.x 支持范围内，因此需要替换。

---

## Ingestion Pipeline（写入路径）

### 数据源

- 指定文件夹作为知识库唯一来源
- 文件夹内文件统一视为文本文件处理（UTF-8, errors=replace）
- 多模态文件（PDF、图片等）的解析为文本格式由外部流程负责，不在本服务范围内
- 跳过隐藏文件（路径中含 `.` 开头的部分）

### 触发方式

- **启动时**：全量扫描，逐文件对比 hash 与 SQLite 记录，增量更新索引
- **运行时**：watchdog 监听文件夹变更事件（2 秒防抖），触发全量 rescan

### 变更检测（SQLite 元数据库）

每个知识库维护一个 SQLite 元数据库（存储在 `data/metadata/{collection}.db`），记录已索引文件状态：

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

### 索引流程

1. 递归扫描文件夹（`rglob("*")`），逐文件对比 SHA256 与 SQLite 记录
2. **新增/修改文件**：切片 → 向量化 → 按 `source_file` 删除旧 chunks → 插入新 chunks（UUID 作为 point ID）→ 更新 SQLite
3. **删除文件**：按 `source_file` 删除 Qdrant 中对应 chunks → 删除 SQLite 记录
4. watchdog 监听到文件事件时执行全量 rescan 逻辑（防抖后）

---

## 存储 Schema（Qdrant）

每个知识库 = 1 个 collection，使用 **named vectors**。Qdrant 运行在嵌入式模式（embedded），数据存储在 `data/qdrant/`。

### 向量配置

| 向量名 | 维度 | 距离 | 说明 |
|---|---|---|---|
| `dense` | 512 | Cosine | bge-small-zh-v1.5 语义向量 |
| `sparse` | 可变 | Dot | BM25 稀疏词权重向量 |

### Payload 结构

| 字段 | 类型 | 说明 |
|---|---|---|
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
2. Dense prefetch 和 Sparse prefetch 各取 Top-N 候选（N = rerank_candidates，默认 30）
3. Qdrant 服务端执行 RRF 融合，合并去重
4. 融合后取 Top-N 候选送入 Reranker

> **与原设计的差异**：原设计使用加权分数融合（Weighted Score Fusion, `w_dense * dense_score + w_sparse * sparse_score`）。实现中改用 Qdrant 内置 RRF，因为 RRF 是 rank-based 融合，不依赖异构分数的归一化，实现更简洁且鲁棒。API 中 `w_dense`/`w_sparse` 参数保留但当前未使用，待后续按需接入。

---

## 模型选型

### 推理后端约束

所有模型必须在 **fastembed 0.7.x** 的支持列表内（需预转换为 ONNX 格式）。这排除了原设计中的 BGE-M3 和 bge-reranker-v2-m3。

### 当前模型清单

| 模型 | 用途 | 大小 | 说明 |
|---|---|---|---|
| `BAAI/bge-small-zh-v1.5` | Dense Embedding (512 维) | ~90MB | 中文语义向量 |
| `Qdrant/bm25` | Sparse Embedding | ~小 | BM25 稀疏词权重 |
| `Xenova/ms-marco-MiniLM-L-12-v2` | Reranker (Cross-Encoder) | ~120MB | 英文 reranker |

### 与原设计的差异

| 原设计 | 实际选型 | 原因 |
|---|---|---|
| `BAAI/bge-m3` (Dense+Sparse 统一模型, 1024 维, ~2GB) | `bge-small-zh-v1.5` (Dense) + `Qdrant/bm25` (Sparse) | fastembed 不支持 BGE-M3 |
| `BAAI/bge-reranker-v2-m3` (~1.2GB, 多语言) | `Xenova/ms-marco-MiniLM-L-12-v2` (~120MB, 英文) | fastembed 不支持 bge-reranker-v2-m3；较大替代模型下载失败 |

### 已验证可用的 fastembed 多语言 Dense 模型

| 模型 | 维度 | 大小 | 语言 |
|---|---|---|---|
| `jinaai/jina-embeddings-v2-base-zh` | 768 | 0.64GB | 中英 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 0.22GB | 50+ 语言 |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | 1.0GB | 50+ 语言 |
| `intfloat/multilingual-e5-large` | 1024 | 2.24GB | 100+ 语言 |
| `jinaai/jina-embeddings-v3` | 1024 | 2.29GB | ~100 语言 |

### 升级路径

- **Dense Embedding**：推荐升级到 `jinaai/jina-embeddings-v2-base-zh`（768 维，中英双语，0.64GB）
- **Reranker**：推荐升级到 `jinaai/jina-reranker-v2-base-multilingual`（多语言，fastembed 支持）
- 如 fastembed 后续版本支持 BGE-M3，可回到原设计方案

---

## 精排（Reranker）

### 当前选型：Xenova/ms-marco-MiniLM-L-12-v2

| 维度 | 说明 |
|---|---|
| 模型 | `Xenova/ms-marco-MiniLM-L-12-v2` (~120MB) |
| 架构 | Cross-Encoder |
| 接口 | fastembed `TextCrossEncoder` |
| 语言 | 英文（对中文效果有限） |

### 实现细节

- Reranker 对候选集重新排序，返回排序后的 **索引列表**
- 最终返回的 `score` 字段为 Qdrant RRF 融合分数，非 Reranker 分数

---

## 模型管理

### 设计原则

模型文件存放在项目目录内（`./models/`），通过 `.gitignore` 排除，不作为 git 项目的一部分。

优势：
- 不污染用户全局环境（如 `~/.cache/huggingface`）
- 卸载即删目录，干净
- 多实例可以共存不互相干扰

### 路径管理

- 由 `config.py` 统一定义 `MODEL_ROOT`（默认 `./models`）
- 所有模型加载均基于此路径，不在各模块中硬编码
- fastembed 的 `cache_dir` 参数指向 `MODEL_ROOT`

### 下载策略

- **服务启动时自动检测**：通过 fastembed 内部机制，模型缺失则自动下载
- `main.py` lifespan 中调用 `warmup()` 预加载 embedding 和 reranker 模型

> **与原设计的差异**：未实现 `bootstrap.sh` 预下载脚本、模型版本锁定（revision pinning）和启动完整性校验。模型的下载和校验完全委托给 fastembed 内部管理。

---

## 服务层（FastAPI）

单进程，模型启动时预加载。`main.py` 作为入口，使用 `asynccontextmanager` lifespan 管理启动/关闭。

### API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/search` | 语义检索（query, top_k, w_dense, w_sparse, collection, rerank_candidates） |
| `POST` | `/collections` | 创建知识库（name, folder_path, chunk_size, chunk_overlap） |
| `DELETE` | `/collections/{name}` | 删除知识库（含 Qdrant collection + SQLite 元数据） |
| `POST` | `/collections/{name}/sync` | 手动触发同步 |
| `GET` | `/collections/{name}/status` | 查看索引状态（indexed_files, total_chunks, watching） |

### 检索流程

1. fastembed 编码 query → dense 向量 + sparse 向量
2. Qdrant prefetch 双路召回 → RRF 融合 → Top 20-30 候选
3. Cross-Encoder 对候选集精排
4. 返回 Top-K 结果（含切片原文、来源文件路径、切片位置、融合分数）

### 已知限制

- **Collection 注册表不持久化**：`_collections` dict 存于内存，服务重启后丢失。Qdrant 数据和 SQLite 元数据持久化在磁盘，但 collection→folder 映射需重新创建。
- **Watchdog 状态不持久化**：重启后需重新调用 `POST /collections` 恢复监听。

---

## 内存预算（64GB）

| 组件 | 实际占用 |
|---|---|
| bge-small-zh-v1.5 (Dense) | ~90MB |
| Qdrant/bm25 (Sparse) | <50MB |
| ms-marco-MiniLM-L-12-v2 (Reranker) | ~120MB |
| Qdrant（嵌入式） | ~100-500MB（取决于数据量） |
| 系统 + FastAPI + ONNX Runtime | ~500MB |
| **总计** | **<1.5GB** |

远低于 64GB 上限，无内存压力。如升级到更大模型（如 jina-embeddings-v2-base-zh + jina-reranker），预计增加到 ~2-3GB。

---

## 目录结构

```text
hermit/
├── main.py                    # FastAPI 入口 + lifespan（模型预加载）
├── pyproject.toml
├── hermit/
│   ├── config.py              # 配置管理（MODEL_ROOT、模型名、默认参数）
│   ├── ingestion/
│   │   ├── scanner.py         # 文件夹扫描 + 变更检测 + 索引
│   │   ├── watcher.py         # watchdog 实时监听（2s 防抖）
│   │   └── chunker.py         # 固定大小文本切片 + overlap
│   ├── retrieval/
│   │   ├── embedder.py        # Dense (TextEmbedding) + Sparse (SparseTextEmbedding)
│   │   ├── searcher.py        # Qdrant prefetch + RRF 融合 + rerank
│   │   └── reranker.py        # TextCrossEncoder
│   ├── storage/
│   │   ├── qdrant.py          # Qdrant 嵌入式客户端 + collection 管理
│   │   └── metadata.py        # SQLite 元数据管理
│   └── api/
│       ├── routes.py          # API 路由
│       └── schemas.py         # Pydantic 请求/响应模型
├── models/                    # 模型文件（.gitignore 排除）
├── data/
│   ├── qdrant/                # Qdrant 嵌入式存储
│   └── metadata/              # SQLite 元数据库 ({collection}.db)
└── docs/
    └── design.md