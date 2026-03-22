# Hermit

[English version](./README.md)

Hermit 是一个**自包含、纯本地运行的语义检索服务**，用于把一个或多个文档目录注册成可搜索的知识库 collection。

它适合本地优先的工作流，也很适合作为笔记库、技术文档库和小型 RAG 应用的轻量检索后端。

## 特性

- **完全本地运行**：模型、向量数据和元数据都保存在项目目录中
- **多 collection 支持**：一个目录对应一个 collection
- **混合检索**：dense + sparse 双路召回
- **重排**：使用 cross-encoder 对融合后的候选结果进行 rerank
- **增量同步**：启动时扫描，运行时监听文件变化
- **CPU 友好**：基于 `fastembed` + ONNX Runtime，无需 GPU

## 适用场景

Hermit 很适合这些场景：

- 对本地笔记仓库或 Markdown 文档做语义搜索
- 为本地工具或 Agent 提供简单的检索 API
- 在不依赖云服务的情况下搭建轻量、私有的 RAG 检索层

当前实现会把文件按文本读取，使用 `UTF-8` 解码，解码失败时进行容错替换，因此最适合 `.md`、`.txt` 等纯文本内容。

## 工作方式

### 检索流程

Hermit 的搜索流程如下：

1. 将 query 编码为 dense 和 sparse 两种表示
2. 在 Qdrant 中执行混合召回
3. 使用 RRF 融合候选结果
4. 对候选结果执行 rerank
5. 返回最相关的 chunk

### 索引流程

每个已注册目录都会经历：

1. 启动扫描
2. SQLite 元数据对比
3. 文本分块
4. 生成向量
5. 写入 Qdrant
6. 持续文件监听

### 默认参数

- Chunk size: `512`
- Chunk overlap: `64`
- 搜索 `top_k`: `5`
- 默认 `w_dense`: `0.7`
- 默认 `w_sparse`: `0.3`
- 默认 rerank candidates: `30`
- collection 数量上限: `4`
- collection 名称最大长度: `64`
- 默认端口: `8000`

## 技术栈

- **API 框架**: FastAPI
- **向量数据库**: Qdrant embedded mode
- **推理后端**: fastembed
- **元数据存储**: SQLite
- **文件监听**: watchdog

当前使用的模型：

- Dense embedding: `jinaai/jina-embeddings-v2-base-zh`
- Sparse embedding: `Qdrant/bm25`
- Reranker: `jinaai/jina-reranker-v2-base-multilingual`

## 项目结构

```text
.
├── main.py
├── pyproject.toml
├── README.md
├── README_cn.md
├── docs/
│   └── design.md
├── hermit/
│   ├── cli.py
│   ├── config.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── scanner.py
│   │   ├── task_queue.py
│   │   └── watcher.py
│   ├── retrieval/
│   │   ├── embedder.py
│   │   ├── reranker.py
│   │   └── searcher.py
│   └── storage/
│       ├── metadata.py
│       ├── model_signature.py
│       ├── qdrant.py
│       └── registry.py
├── data/
│   ├── collections.json
│   ├── metadata/
│   └── qdrant/
└── models/
```

## 安装

### 环境要求

- Python `3.12+`
- macOS 或 Linux

建议使用虚拟环境。

### 从源码安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

如果你打算使用 `hermit download`，请确认环境中可用 `huggingface_hub`，因为该命令依赖它下载模型快照。

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

- 首次启动服务时，若模型缺失，也会自动下载
- 提前下载可以让首次启动更可控，避免启动过程“边开机边搬家”

### 2. 注册知识库目录

```bash
hermit kb add my_docs ./documents
```

自定义分块参数：

```bash
hermit kb add my_docs ./documents --chunk-size 512 --chunk-overlap 64
```

查看 collection 列表：

```bash
hermit kb list
```

删除 collection：

```bash
hermit kb remove my_docs
```

collection 名称规则：

- 必须以字母或数字开头
- 只能包含字母、数字、下划线和连字符
- 名称必须唯一

### 3. 启动服务

```bash
python main.py
```

启动时，Hermit 会：

- 预热 embedding 和 reranker 模型
- 启动后台索引 worker
- 从 `data/collections.json` 恢复已持久化的 collection
- 扫描每个 collection 目录
- 启动目录监听

默认监听地址：

- Host: `0.0.0.0`
- Port: `8000`

### 4. 搜索

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

## CLI

Hermit 当前提供以下命令。

### `hermit download`

下载所有所需模型，并可选执行基础验证。

```bash
hermit download
```

参数：

- `--force`: 强制重新下载
- `--skip-verify`: 跳过下载后的验证

### `hermit kb add <name> <dir>`

将目录注册为 collection。

```bash
hermit kb add notes ./documents
```

可选参数：

- `--chunk-size`
- `--chunk-overlap`

### `hermit kb remove <name>`

删除 collection 及其元数据。

```bash
hermit kb remove notes
```

### `hermit kb list`

列出所有已注册 collection。

```bash
hermit kb list
```

## HTTP API

当前代码实现了以下接口。

### `POST /search`

执行混合语义检索。

请求示例：

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

手动触发某个 collection 的扫描同步。

返回字段：

- `added`
- `updated`
- `deleted`

示例：

```bash
curl -X POST http://127.0.0.1:8000/collections/my_docs/sync
```

### `GET /collections/{name}/status`

查看 collection 状态。

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

查看某个 collection 的后台索引任务状态。

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

## 数据存储

默认情况下，Hermit 将运行数据保存在项目目录中：

- `models/`: 本地模型缓存
- `data/qdrant/`: Qdrant embedded 数据
- `data/metadata/`: 每个 collection 一个 SQLite 数据库
- `data/collections.json`: collection 持久化配置

因此它很容易备份、迁移和清理，不会悄悄在用户目录里挖地道。

## 索引行为

### 文件处理规则

- 递归扫描所有非隐藏文件
- 跳过任一路径片段以 `.` 开头的文件或目录
- 按文本读取，使用 `utf-8` 和 `errors="replace"`

### 变更检测

Hermit 通过 SQLite 跟踪已索引文件，并使用 **SHA256** 检测内容变化。

扫描时会处理：

- **新增文件**：入队或直接索引
- **修改文件**：重新切块、重建向量并替换旧数据
- **删除文件**：从 Qdrant 和 SQLite 中移除

### 分块规则

- 默认 chunk 大小为 `512` 字符
- 相邻 chunk 重叠 `64` 字符
- 空白文本会被跳过
- 短文本保持单 chunk

## 已知限制

- 当前 **没有 API** 用于创建或删除 collection；请使用 CLI
- API 接受 `w_dense` 和 `w_sparse` 参数，但当前实现使用的是 **RRF 融合**，并非显式加权分数融合
- 所有文件均按文本处理；PDF、图片和 Office 文档解析不在当前范围内
- collection 数量上限目前是 `4`
- 首次模型下载可能较慢，并会占用一定磁盘空间

## 开发与测试

当前测试覆盖：

- CLI 参数校验与 collection 管理
- scanner 的新增 / 更新 / 删除逻辑
- task queue 状态统计
- 部分 API 路由行为

运行测试：

```bash
pytest
```

## 设计说明

更多实现细节请见：

- `docs/design.md`

## 一句话总结

如果你需要一个小巧、纯本地、支持多 collection 的语义检索服务，Hermit 是个安静但靠谱的工具选手。
