# Hermit — 本地语义检索服务设计文档

面向 **M4 Pro (64GB RAM)** 的 Self-contained 语义检索 Skill。集成 **BGE-M3** 混合召回与 **bge-reranker-v2-m3** 精排，单机部署，个人使用场景。

---

## 架构总览

```
写入路径: 文件夹扫描 → 变更检测 → 文本切片 → 向量化(Dense+Sparse) → Qdrant 入库
查询路径: Query → 混合召回(加权融合) → 精排(Cross-Encoder) → 返回结果
```

分层结构：

1. **Ingestion 层**：文件夹扫描 + watchdog 监听 → 变更检测（SQLite）→ 文本切片 → 向量化 → 入库
2. **检索层 (Recall)**：BGE-M3 Dense (1024维) + Sparse 双路召回，加权分数融合
3. **精排层 (Rerank)**：bge-reranker-v2-m3 Cross-Encoder，处理 Top 20-30 候选集
4. **服务层**：FastAPI 暴露 OpenAPI 接口

支持**多知识库**：每个文件夹 = 一个独立的 Qdrant collection。

---

## Ingestion Pipeline（写入路径）

### 数据源

- 指定文件夹作为知识库唯一来源
- 文件夹内文件统一视为文本文件处理
- 多模态文件（PDF、图片等）的解析为文本格式由外部流程负责，不在本服务范围内

### 触发方式

- **启动时**：全量扫描，逐文件对比 hash/mtime 与 SQLite 记录，增量更新索引
- **运行时**：watchdog 监听文件夹变更事件，实时触发索引更新

### 变更检测（SQLite 元数据库）

每个知识库维护一个 SQLite 元数据库，记录已索引文件状态：

| 字段 | 类型 | 说明 |
|---|---|---|
| `file_path` (PK) | TEXT | 文件绝对路径 |
| `file_hash` | TEXT | 文件内容 SHA256 |
| `file_mtime` | REAL | 修改时间 |
| `chunk_count` | INTEGER | 该文件切片数 |
| `last_indexed_at` | REAL | 上次索引时间 |

**选用 SQLite 的理由**：启动时对每个文件查 Qdrant payload 效率低；SQLite 单次全表扫描即可完成对比，天然适合存储关系型元数据。

### 索引流程

1. 扫描文件夹，逐文件对比 hash/mtime 与 SQLite 记录
2. **新增/修改文件**：切片 → 向量化 → 按 `source_file` 删除旧 chunks → 插入新 chunks → 更新 SQLite
3. **删除文件**：按 `source_file` 删除 Qdrant 中对应 chunks → 删除 SQLite 记录
4. watchdog 监听到文件事件时执行同样逻辑

---

## 存储 Schema（Qdrant）

每个知识库 = 1 个 collection，使用 **named vectors**。

### 向量配置

| 向量名 | 维度 | 距离 | 说明 |
|---|---|---|---|
| `dense` | 1024 | Cosine | BGE-M3 语义向量 |
| `sparse` | 可变 | Dot | BGE-M3 稀疏词权重向量 |

### Payload 结构

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | string | 切片原文 |
| `source_file` | string | 源文件路径 |
| `chunk_index` | int | 该切片在源文件中的位置（0-based） |
| `total_chunks` | int | 该源文件的总切片数 |

在 `source_file` 上建 **payload index**，确保按文件过滤/删除高效。

### 查询文件全部切片

通过 `source_file` payload filter 查询，配合 `chunk_index` 排序还原原文顺序。`total_chunks` 字段让调用方判断是否获取完整。

---

## 混合召回融合策略

使用**加权分数融合（Weighted Score Fusion）**：

1. 分别进行 dense search 和 sparse search，各取 Top-K 候选
2. 归一化两路分数到 [0, 1]
3. `final_score = w_dense * dense_score + w_sparse * sparse_score`
4. 权重通过 API 参数传入，默认 `w_dense=0.7, w_sparse=0.3`
5. 合并去重后取 Top-N 送入 Reranker

实现方式：使用 Qdrant 原生的 `prefetch` + `query` API（支持 named vectors 的多路召回 + 服务端融合），避免在应用层做两次查询。

---

## 精排（Reranker）

### 选型：bge-reranker-v2-m3

| 维度 | 说明 |
|---|---|
| 模型 | `BAAI/bge-reranker-v2-m3` (~568M 参数) |
| 架构 | 专用 Cross-Encoder，非 generative model |
| 接口 | fastembed `TextCrossEncoder`，一行调用 |
| 延迟 | Rerank 20-30 条 ≈ 100-200ms (Apple Silicon) |
| 语言 | 多语言支持，与 BGE-M3 同生态 |

### 选型理由

- 与 BGE-M3 同生态，模型能力互补
- 专用 cross-encoder 架构，不需要自定义 prompt + logit 提取
- fastembed 直接支持，实现复杂度极低
- 相比 Gemma-2B 裸模型：体积更小（568M vs 4GB）、延迟更低、rerank 效果经过充分验证

### 升级路径

如果后续发现 v2-m3 质量不够，可换 `bge-reranker-v2-gemma`（同 BGE 生态，Gemma-2B 底座但经过 rerank 微调）。

---

## 模型管理

### 设计原则

模型文件存放在项目目录内（`./models/`），通过 `.gitignore` 排除，不作为 git 项目的一部分。分发时由 bootstrap 脚本或服务启动时自动下载。

优势：
- 不污染用户全局环境（如 `~/.cache/huggingface`）
- 卸载即删目录，干净
- 多实例可以共存不互相干扰

### 模型清单

| 模型 | 用途 | 大小 |
|---|---|---|
| `BAAI/bge-m3` | Embedding (Dense + Sparse) | ~2GB |
| `BAAI/bge-reranker-v2-m3` | Reranker (Cross-Encoder) | ~1.2GB |

### 路径管理

- 由 `config.py` 统一定义 `MODEL_ROOT`（默认 `./models`）
- 所有模型加载均基于此路径，不在各模块中硬编码
- fastembed 的 `cache_dir` 参数指向 `MODEL_ROOT`

### 版本锁定

配置中固定模型 `revision`（HuggingFace commit hash），确保所有用户下载同一版本，保证可复现性。

### 下载策略

- **服务启动时自动检测**：模型缺失则自动下载（带进度提示），用户直接 `python main.py` 即可使用
- **bootstrap.sh 可选预下载**：作为预安装手段，适合网络受限环境提前准备

### 完整性校验

启动时轻量校验：检查关键文件/目录是否存在 + 文件大小是否匹配。校验失败提示重新下载。

---

## 服务层（FastAPI）

单进程，模型启动时预加载。

### API 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/search` | 语义检索（query, top_k, w_dense, w_sparse, collection） |
| `POST` | `/collections` | 创建知识库（name, folder_path, chunk_size, overlap） |
| `DELETE` | `/collections/{name}` | 删除知识库 |
| `POST` | `/collections/{name}/sync` | 手动触发同步 |
| `GET` | `/collections/{name}/status` | 查看索引状态 |

### 检索流程

1. BGE-M3 编码 query → dense + sparse 向量
2. Qdrant prefetch 双路召回 → 加权融合 → Top 20-30 候选
3. bge-reranker-v2-m3 对候选集精排
4. 返回 Top-K 结果（含切片原文、来源文件路径、切片位置）

---

## 内存预算（64GB）

| 组件 | 预估占用 |
|---|---|
| BGE-M3 | ~2GB |
| bge-reranker-v2-m3 | ~1.2GB |
| Qdrant（嵌入式，10 万条） | ~1-2GB |
| 系统 + FastAPI | ~1GB |
| **总计** | **~5-6GB** |

远低于 64GB 上限，无内存压力。

---

## 目录结构

```text
hermit/
├── main.py                    # FastAPI 入口
├── pyproject.toml
├── bootstrap.sh               # 可选的一键安装脚本
├── hermit/
│   ├── config.py              # 配置管理（MODEL_ROOT、默认参数等）
│   ├── ingestion/
│   │   ├── scanner.py         # 文件夹扫描 + 变更检测
│   │   ├── watcher.py         # watchdog 实时监听
│   │   └── chunker.py         # 文本切片
│   ├── retrieval/
│   │   ├── embedder.py        # BGE-M3 dense + sparse embedding
│   │   ├── searcher.py        # 混合召回 + 加权融合
│   │   └── reranker.py        # bge-reranker-v2-m3
│   ├── storage/
│   │   ├── qdrant.py          # Qdrant collection 管理
│   │   └── metadata.py        # SQLite 元数据管理
│   └── api/
│       ├── routes.py          # API 路由
│       └── schemas.py         # Pydantic 模型
├── models/                    # 模型文件（.gitignore 排除）
├── data/                      # Qdrant + SQLite 数据（.gitignore 排除）
└── docs/
    └── design.md
```

### .gitignore 规则

```
/models/
/data/
```

---

## 性能调优参数

| 参数 | 推荐设置 | 理由 |
|---|---|---|
| **Embedding Batch Size** | 16 - 32 | 充分利用 M4 Pro 的 273GB/s 内存带宽 |
| **Qdrant Threading** | 12 Threads | 匹配 M4 Pro 核心数，加速 HNSW 索引构建 |
| **Quantization** | Binary (BQ) + rescore | 减少内存占用，rescore 补偿精度 |
| **Rerank 候选数** | 20 - 30 | 在延迟和精度间平衡（100 条串行推理太慢） |

---

## 决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 变更检测 | SQLite + 启动扫描 + watchdog | 全覆盖，SQLite 做 diff 高效 |
| 多知识库 | 每个文件夹 = 独立 collection | 隔离性好，独立管理 |
| 融合策略 | 加权分数融合，API 可配 | 简单有效，灵活度够 |
| Reranker | bge-reranker-v2-m3 | 同生态、低延迟、专用架构 |
| 模型存储 | 项目目录内 + .gitignore 排除 | Self-contained，分发简单 |
| 模型版本 | 锁定 revision (commit hash) | 可复现性 |
| 模型下载 | 启动时自动检测 + bootstrap 可选预下载 | 开箱即用 |
| 部署 | 单进程 | 个人场景，后续再优化并发 |
| 文件解析 | 仅文本 | 多模态解析由外部负责 |
| 依赖管理 | pyproject.toml | 与项目结构统一 |
