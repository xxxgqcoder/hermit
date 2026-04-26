
# Hermit 设计文档索引

本地语义搜索服务：文件夹 → 向量索引 → 混合召回 + 精排。

---

## 分层架构

```
┌─────────────────────────────────────────────────────┐
│  Ingestion 层  文件夹扫描 + 变更检测 → 切片 → 向量化 → 入库  │
├─────────────────────────────────────────────────────┤
│  Recall 层     Dense + Sparse 双路召回，Qdrant RRF 融合     │
├─────────────────────────────────────────────────────┤
│  Rerank 层     Cross-Encoder 处理 Top-50 候选集            │
├─────────────────────────────────────────────────────┤
│  Service 层    FastAPI + CLI，多知识库管理                  │
└─────────────────────────────────────────────────────┘
```

支持**多知识库**：每个文件夹 = 一个独立的 Qdrant collection。

**推理后端**：fastembed（ONNX Runtime），CPU 推理，无需 PyTorch / GPU。

---

## 模块文档

| 文档 | 内容 |
|---|---|
| [ingestion.md](ingestion.md) | 数据源白名单、变更检测（SQLite）、文本切片、索引流程、批量写入 |
| [storage.md](storage.md) | Qdrant Local/Standalone 双模式、Docker 容器管理、collection schema |
| [retrieval.md](retrieval.md) | 混合召回（RRF）、Cross-Encoder 精排、检索流程 |
| [models.md](models.md) | 模型选型、路径管理、下载策略、内存预算 |
| [api.md](api.md) | HTTP 端点、CLI、持久化恢复 |

其他文档：

| 文档 | 内容 |
|---|---|
| [qdrant_standalone.md](qdrant_standalone.md) | Standalone 模式详细部署说明 |
| [markdown-chunking.md](markdown-chunking.md) | Markdown 切片策略详解 |
| [skill-distribution.md](skill-distribution.md) | hermit-search Agent Skill 分发 |

---

## 目录结构

```text
hermit/
├── main.py                    # 开发模式入口（uvicorn 直接运行）
├── pyproject.toml
├── hermit/
│   ├── app.py                 # FastAPI 应用 + lifespan
│   ├── cli.py                 # CLI 入口（hermit start/stop/kb/search/...）
│   ├── config.py              # 配置（HERMIT_HOME、MODEL_ROOT、DATA_ROOT、模型名）
│   ├── models.py              # 模型下载与校验
│   ├── ingestion/
│   │   ├── scanner.py         # 文件夹扫描 + 白名单过滤 + 变更检测 + 索引
│   │   ├── watcher.py         # watchdog 实时监听（2s 防抖）
│   │   ├── chunker.py         # token 级文本切片 + overlap
│   │   └── task_queue.py      # 后台索引任务队列（线程池）
│   ├── retrieval/
│   │   ├── embedder.py        # Dense + Sparse embedding
│   │   ├── searcher.py        # Qdrant prefetch + RRF 融合 + rerank 编排
│   │   └── reranker.py        # TextCrossEncoder
│   ├── storage/
│   │   ├── qdrant.py          # Qdrant 客户端（Local/Standalone 双模式）
│   │   ├── qdrant_docker.py   # Docker 容器生命周期管理（Standalone 模式）
│   │   ├── metadata.py        # SQLite 元数据读写
│   │   ├── registry.py        # 知识库注册表
│   │   └── model_signature.py # 模型变更检测
│   └── api/
│       ├── routes.py          # API 路由
│       └── schemas.py         # Pydantic 请求/响应模型
└── docs/
    ├── design.md              # 本文件（索引）
    ├── ingestion.md
    ├── storage.md
    ├── retrieval.md
    ├── models.md
    ├── api.md
    ├── qdrant_standalone.md
    ├── markdown-chunking.md
    └── skill-distribution.md

~/.hermit/                     # 运行时数据（可通过 HERMIT_HOME 覆盖）
├── models/                    # 模型文件（fastembed ONNX cache）
├── data/
│   ├── qdrant/                # Qdrant 数据
│   ├── metadata/              # SQLite 元数据库（{collection}.db）
│   ├── collections.json       # 知识库注册表
│   └── model_signature.json   # 模型签名
├── logs/
│   └── hermit.log
└── hermit.pid
```