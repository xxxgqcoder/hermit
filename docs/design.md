# Hermit 设计总览（索引）

Hermit 是一个自包含的本地语义搜索服务：监听知识库文件夹，增量索引文本文件，
提供 dense + 关键词混合检索 + cross-encoder 精排，经 FastAPI 暴露 OpenAPI 接口。

本文为**索引**，各子模块的详细设计拆分到独立文档；这里只保留分层结构、模块地图、
跨模块的内存预算与目录结构。

## 分层结构

1. **Ingestion 层**：文件夹扫描 + 定期轮询 → 变更检测（SQLite）→ 文本切片（token 级 / markdown 语义级）→ dense 向量化（带磁盘缓存）→ 入库
2. **检索层 (Recall)**：Dense 向量 + LanceDB 原生 FTS 双路召回 + RRF 融合；另支持纯语义 / 纯关键词 / 子串 fuzzy 三种单路模式
3. **精排层 (Rerank)**：Cross-Encoder 处理 Top 20 候选集（`DEFAULT_RERANK_CANDIDATES`）
4. **服务层**：FastAPI 单进程，模型启动预加载，lifespan 管理收发

支持**多知识库**（上限 `MAX_COLLECTIONS=4`）：每个文件夹 = 一张独立的 LanceDB 表 + 一个 SQLite 元数据库。

## 模块文档索引

| 子模块 | 文档 | 内容 |
|---|---|---|
| Ingestion 写入路径 | [`ingestion.md`](./ingestion.md) | 数据源/扩展名 allowlist、轮询 watcher、变更检测 + **SQLite schema**、切片分派、标题前缀、索引流程、task_queue、嵌入缓存 |
| Markdown 语义切片 | [`markdown-chunking.md`](./markdown-chunking.md) | `parse_md_blocks` 状态机 + `chunk_markdown` heading-aware 滑窗 |
| 存储（LanceDB） | [`lancedb.md`](./lancedb.md) | 磁盘布局、表 schema、索引策略、FTS 行为、压缩/版本回收、并发 |
| 检索路径 | [`retrieval.md`](./retrieval.md) | 四种搜索模式、RRF 融合、filename 过滤、精排、端到端流程 |
| 推理后端与模型服务 | [`inference.md`](./inference.md) | fastembed、INT8 量化、ONNX arena/线程、双模型 idle-unload、模型下载/管理 |
| Skill 分发 | [`skill-distribution.md`](./skill-distribution.md) | hermit-search skill 的打包与安装链路 |

## 内存预算

稳态内存由三项设计共同压低：**ONNX arena 默认关闭**（RSS 不随突发/累积索引爬升）+
**dense / reranker 双 idle-unload**（安静期把整块 session 内存还给 OS）+
**INT8 量化**（权重减半）。机制细节见 [`inference.md`](./inference.md)。

| 状态 | 典型 RSS | 说明 |
|---|---|---|
| 闲置（两模型均已 unload） | **~320MB** | 仅 Python + FastAPI + LanceDB + 常驻 |
| 活跃峰值 | **~7GB** | 重并发搜索 + 索引同时进行；单 worker 串行下远低于此 |

并发缩放代价：搜索峰值大致随 worker 线性增长（`peak ≈ 3.5 × HERMIT_SEARCH_WORKERS` GB，见
`problems/concurrent-search-rss-blowup.md`），故默认 `HERMIT_SEARCH_WORKERS=1`。历史上 arena 开启 +
无 dense unload 时，守护进程曾在数十小时索引累积下爬到 5–10GB（见
`problems/dense-embedder-arena-creep.md`），现已由上述三项设计收敛。

## 服务与持久化

- **管理接口**：知识库注册/删除/改忽略规则既可 CLI（`hermit kb add/remove/update`）也可 HTTP（`POST /collections`、`DELETE /collections/{name}`）。CLI 在服务运行时转发到 HTTP，未运行时直接操作本地注册表。
- **注册表持久化**：`~/.hermit/data/collections.json` 记录所有 collection（folder_path、ignore_patterns、ignore_extensions），启动时自动恢复并为每个 collection 起轮询 watcher。
- **模型变更检测**：`~/.hermit/data/model_signature.json` 记录上次 dense 模型，变更则启动时全量重建。
- **端口**：`~/.hermit/port.json` 持久化端口，`resolve_port()` 启动时择优可用端口。

## 目录结构

```text
hermit/
├── main.py                    # 开发模式入口（uvicorn 直接运行）
├── pyproject.toml
├── hermit/
│   ├── app.py                 # FastAPI 应用 + lifespan（预加载、量化、collection 恢复、search executor）
│   ├── cli.py                 # CLI 入口（hermit start/stop/kb/search/collection/...）
│   ├── config.py              # 配置（HERMIT_HOME、模型名、idle-timeout、worker 数、ONNX arena/threads 等）
│   ├── models.py              # 模型下载与校验（huggingface_hub）
│   ├── ingestion/             # → docs/ingestion.md
│   │   ├── scanner.py         # 文件夹扫描 + 变更检测 + 索引
│   │   ├── watcher.py         # 后台轮询 watcher（_PollingWatcher，按 POLL_INTERVAL_SECONDS）
│   │   ├── chunker.py         # token 级文本切片 + markdown 语义切片（→ docs/markdown-chunking.md）
│   │   └── task_queue.py      # 后台索引任务队列（线程池，HERMIT_INDEX_WORKERS）
│   ├── retrieval/             # → docs/retrieval.md, docs/inference.md
│   │   ├── embedder.py        # Dense (TextEmbedding)，含 idle unload
│   │   ├── embed_cache.py     # diskcache 落地的 dense 嵌入缓存（sha256 keyed，TTL 30 天）
│   │   ├── searcher.py        # hybrid/semantic/keyword/fuzzy + filename filter + rerank
│   │   ├── reranker.py        # TextCrossEncoder，含 idle unload
│   │   └── fastembed_patch.py # 给 fastembed 暴露 ONNX arena/mem-pattern 注入点
│   ├── storage/               # → docs/lancedb.md
│   │   ├── lance.py           # LanceDB 表管理 + replace_file_chunks + 压缩/版本回收
│   │   ├── quantizer.py       # dense embedder / reranker 的 INT8 量化加载
│   │   ├── metadata.py        # SQLite 元数据管理（files 表）
│   │   ├── registry.py        # 知识库注册表（~/.hermit/data/collections.json）
│   │   └── model_signature.py # 模型变更检测（~/.hermit/data/model_signature.json）
│   └── api/
│       ├── routes.py          # API 路由
│       └── schemas.py         # Pydantic 请求/响应模型
├── docs/                      # 本设计文档集（design.md 为索引）
├── problems/                  # 故障/调查记录（内存爬升、端口冲突等）
└── tests/

~/.hermit/                     # 运行时数据（HERMIT_HOME，可通过环境变量覆盖）
├── models/                    # 模型文件（fastembed ONNX cache）
│   └── quantized/             # INT8 量化产物（{repo_slug}/onnx/model.onnx）
├── cache/
│   └── dense/                 # dense 嵌入向量磁盘缓存（sha256 keyed，TTL 30 天）
├── data/
│   ├── lance/                 # LanceDB 表，每个 collection 一张
│   ├── metadata/              # SQLite 元数据库（{collection}.db）
│   ├── collections.json       # 知识库注册表
│   └── model_signature.json   # 模型签名（变更检测）
├── logs/
│   └── hermit.log             # 服务日志
├── hermit.pid                 # 进程 PID 文件
└── port.json                  # 持久化的服务端口
```
