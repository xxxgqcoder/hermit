# Hermit

Hermit 是一个**本地运行、开箱即用的语义检索服务**，适合把一组文档目录注册成知识库，然后通过 HTTP API 或命令行进行管理与搜索。

它的核心特点是：

- **完全本地运行**：模型、向量库、元数据都保存在项目目录中
- **多知识库支持**：一个目录对应一个 collection
- **混合检索**：Dense + Sparse 双路召回
- **精排**：使用 reranker 对候选结果再次排序
- **自动增量同步**：启动时扫描，运行时监听文件变更
- **无 GPU 依赖**：基于 `fastembed` + ONNX Runtime，默认 CPU 运行

## 适合用来做什么

Hermit 适合这类场景：

- 给个人笔记、技术文档、题解目录做本地语义搜索
- 为本地工具或 Agent 提供一个轻量的检索后端
- 在不依赖云服务的前提下快速搭建 RAG 检索层

当前实现会把知识库目录中的文件统一按文本读取（`UTF-8`，失败时容错替换），因此特别适合已经整理成 `.md`、`.txt` 等纯文本内容的资料目录。

## 核心能力

### 检索链路

Hermit 采用以下检索流程：

1. Query 同时生成 dense / sparse 表示
2. Qdrant 执行混合召回
3. 使用 RRF 融合候选结果
4. reranker 对候选结果重新排序
5. 返回 Top-K chunk 结果

### 索引链路

每个知识库目录都会经过：

1. 启动扫描
2. SQLite 元数据比对（增、删、改）
3. 文本分块
4. 向量化
5. 写入 Qdrant
6. watchdog 持续监听目录变更

### 默认参数

- Chunk size: `512`
- Chunk overlap: `64`
- Search top_k: `5`
- Dense weight 参数默认值: `0.7`
- Sparse weight 参数默认值: `0.3`
- Rerank candidate 数量: `30`
- 最大 collection 数量: `4`
- Collection 名称最大长度: `64`
- 默认服务端口: `8000`

## 技术栈

- **服务框架**: FastAPI
- **向量数据库**: Qdrant embedded
- **推理后端**: fastembed
- **文件监听**: watchdog
- **元数据存储**: SQLite

当前使用的模型：

- Dense embedding: `jinaai/jina-embeddings-v2-base-zh`
- Sparse embedding: `Qdrant/bm25`
- Reranker: `jinaai/jina-reranker-v2-base-multilingual`

## 目录结构

```text
.
├── main.py                  # FastAPI 入口
├── pyproject.toml
├── README.md
├── docs/
│   └── design.md            # 设计说明
├── hermit/
│   ├── cli.py               # 命令行入口
│   ├── config.py            # 全局配置
│   ├── api/
│   │   ├── routes.py        # HTTP 路由
│   │   └── schemas.py       # API 模型定义
│   ├── ingestion/
│   │   ├── chunker.py       # 文本分块
│   │   ├── scanner.py       # 启动扫描 / 增量同步
│   │   ├── task_queue.py    # 后台索引任务
│   │   └── watcher.py       # 文件系统监听
│   ├── retrieval/
│   │   ├── embedder.py      # dense / sparse 编码
│   │   ├── reranker.py      # reranker 封装
│   │   └── searcher.py      # 混合搜索
│   └── storage/
│       ├── metadata.py      # SQLite 元数据
│       ├── model_signature.py
│       ├── qdrant.py        # Qdrant 操作
│       └── registry.py      # collection 注册表
├── data/
│   ├── collections.json     # collection 持久化配置
│   ├── metadata/            # SQLite 元数据库
│   └── qdrant/              # Qdrant embedded 数据
└── models/                  # 本地模型缓存
```

## 安装

### 运行环境

- Python `3.12+`
- macOS / Linux 均可运行
- 建议使用虚拟环境

### 安装依赖

如果你在项目根目录：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

如果你计划使用 `hermit download` 预下载模型，建议额外确认已安装 `huggingface_hub`；该命令依赖它来拉取模型快照。

## 快速开始

### 1. 下载模型（可选但推荐）

```bash
hermit download
```

可选参数：

```bash
hermit download --force
hermit download --skip-verify
```

说明：

- 首次运行服务时，如果模型不存在，也会在预热阶段自动下载
- 手动下载的好处是首次启动更可控，不会把等待时间藏在服务启动里

### 2. 注册一个知识库目录

```bash
hermit kb add my_docs ./documents
```

带自定义分块参数：

```bash
hermit kb add my_docs ./documents --chunk-size 512 --chunk-overlap 64
```

查看已注册的知识库：

```bash
hermit kb list
```

删除一个知识库：

```bash
hermit kb remove my_docs
```

Collection 名称规则：

- 必须以字母或数字开头
- 仅允许字母、数字、下划线 `_`、连字符 `-`
- 不能重复

### 3. 启动服务

```bash
python main.py
```

服务启动后会做这些事：

- 预热 embedding / reranker 模型
- 启动后台索引任务 worker
- 恢复 `data/collections.json` 中保存的 collection 配置
- 对每个 collection 做启动扫描
- 启动文件系统监听

默认监听地址：

- Host: `0.0.0.0`
- Port: `8000`

### 4. 发起搜索

```bash
curl -X POST http://127.0.0.1:8000/search \
	-H 'Content-Type: application/json' \
	-d '{
		"query": "two sum 的思路",
		"collection": "my_docs",
		"top_k": 5,
		"w_dense": 0.7,
		"w_sparse": 0.3,
		"rerank_candidates": 30
	}'
```

## 命令行说明

Hermit 提供 `hermit` 命令：

### `hermit download`

下载所有必需模型，并可选执行简单校验。

```bash
hermit download
```

参数：

- `--force`: 强制重新下载
- `--skip-verify`: 跳过下载后的模型验证

### `hermit kb add <name> <dir>`

注册一个知识库目录。

```bash
hermit kb add notes ./documents
```

可选参数：

- `--chunk-size`
- `--chunk-overlap`

### `hermit kb remove <name>`

移除一个知识库，并删除对应的元数据记录。

```bash
hermit kb remove notes
```

### `hermit kb list`

列出所有已注册知识库。

```bash
hermit kb list
```

## HTTP API

当前代码中已经实现以下接口。

### `POST /search`

执行混合语义检索。

请求体示例：

```json
{
	"query": "滑动窗口最大值",
	"collection": "my_docs",
	"top_k": 5,
	"w_dense": 0.7,
	"w_sparse": 0.3,
	"rerank_candidates": 30
}
```

返回示例：

```json
{
	"results": [
		{
			"text": "...",
			"source_file": "/abs/path/to/file.md",
			"chunk_index": 0,
			"total_chunks": 3,
			"score": 0.82
		}
	]
}
```

### `POST /collections/{name}/sync`

对指定 collection 手动触发一次扫描同步。

返回字段：

- `added`
- `updated`
- `deleted`

示例：

```bash
curl -X POST http://127.0.0.1:8000/collections/my_docs/sync
```

### `GET /collections/{name}/status`

查看 collection 当前状态。

返回字段：

- `name`
- `folder_path`
- `indexed_files`
- `total_chunks`
- `watching`

示例：

```bash
curl http://127.0.0.1:8000/collections/my_docs/status
```

### `GET /collections/{name}/tasks`

查看 collection 的后台索引任务状态。

返回字段：

- `collection`
- `pending_tasks`
- `queued_tasks`
- `in_progress_tasks`
- `worker_alive`

示例：

```bash
curl http://127.0.0.1:8000/collections/my_docs/tasks
```

## 数据存储说明

Hermit 默认把运行数据保存在项目目录下：

- `models/`: 模型缓存
- `data/qdrant/`: Qdrant embedded 数据
- `data/metadata/`: 每个 collection 的 SQLite 元数据库
- `data/collections.json`: 已注册 collection 的持久化配置

这意味着它非常适合本地单机使用，也意味着你可以直接备份整个项目目录来保留检索状态与模型缓存。

## 索引行为说明

### 文件读取规则

- 递归扫描目录下所有**非隐藏文件**
- 跳过路径中任一段以 `.` 开头的文件/目录
- 按文本文件读取，编码使用 `utf-8`，读取失败时使用 `errors="replace"`

### 变更检测规则

Hermit 通过 SQLite 维护每个已索引文件的记录，并使用 **SHA256** 判断文件内容是否变化。

扫描时会处理三类变化：

- **新增文件**：加入索引任务
- **修改文件**：重新切块、重新向量化、覆盖旧数据
- **删除文件**：从 Qdrant 和 SQLite 中清除

### 分块规则

- 默认每 `512` 个字符一个 chunk
- 相邻 chunk 重叠 `64` 个字符
- 空白文本不会被索引
- 短文本不会被强制切成多个块

## 已知限制

当前实现已经很好用，但也有几处值得提前知道：

- 目前 API **没有**提供新增 / 删除 collection 的接口，这部分由 CLI 管理
- `w_dense` / `w_sparse` 参数目前在接口中保留，但搜索实现实际使用的是 **RRF 融合**，不是显式加权打分
- 所有文件都按文本读取，不负责 PDF、图片、Office 文档等格式解析
- 最大 collection 数量当前限制为 `4`
- 首次下载模型可能较慢，且占用一定磁盘空间

## 开发与测试

项目当前包含 pytest 测试，主要覆盖：

- CLI 注册与参数校验
- scanner 增删改逻辑
- task queue 状态统计
- 部分 API 路由行为

运行测试：

```bash
pytest
```

## 设计文档

更详细的实现思路见：

- `docs/design.md`

## 一句话总结

如果你想要一个**纯本地、可持续监听目录、支持多知识库的轻量语义搜索服务**，Hermit 就是那种“不吵不闹，但很能干”的工具型选手。