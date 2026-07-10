# Hermit

[English version](./README.md)

Hermit 是一个**自包含、纯本地运行的语义检索服务**，用于把一个或多个文档目录注册成可搜索的知识库 collection。

它适合本地优先的工作流，也很适合作为笔记库、技术文档库和小型 RAG 应用的轻量检索后端。

## 特性

- **完全本地运行**：模型、向量数据和元数据都保存在项目目录中
- **Markdown 语义切片**：状态机解析器将 `.md` 文件拆分为 11 种语义完整的 block 类型（标题、代码块、数学公式、表格、列表……），代码块和表格不会被从中间截断
- **Heading-Aware 滑动窗口**：chunk 尽量从 section 标题开始，每个 chunk 无需外部上下文即可自我表达所属 section，检索命中质量更高
- **嵌入式向量存储**：LanceDB 列存，无 Docker、无外部进程
- **多 collection 支持**：一个目录对应一个 collection
- **混合检索**：dense 向量 + tantivy FTS（BM25），通过 RRF 融合
- **重排**：使用 cross-encoder 对融合后的候选结果进行 rerank
- **增量同步**：启动时扫描，运行时定期轮询检测变化
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

1. 将 query 编码为 dense 向量
2. 在 LanceDB 中执行混合召回（向量 + tantivy FTS），用 RRF 融合
3. 用 cross-encoder 对候选结果做 rerank
4. 返回最相关的 chunk

### 索引流程

每个已注册目录都会经历：

1. 启动扫描
2. SQLite 元数据对比
3. 文本分块（见下文）
4. 生成 dense 向量
5. 写入 LanceDB（FTS 索引随写入自动更新）
6. 定期轮询检测

### Markdown 语义切片

对于 `.md` 文件，Hermit 采用两阶段策略，而非简单的 token 滑动窗口：

**第一阶段 — 块解析（`parse_md_blocks`）**：状态机解析器逐行扫描文件，将内容归类为 11 种语义完整的 block 类型：

| block 类型 | 示例 |
|---|---|
| YAML frontmatter | 文件首部 `---` … `---` |
| 代码围栏块 | ` ``` ` … ` ``` ` 或 `~~~` … `~~~` |
| 数学块 | `$$` … `$$` |
| ATX 标题 | `# H1` … `###### H6` |
| Setext 标题 | 文本 + `===` 或 `---` 下划线 |
| 表格 | 管道符分隔的连续行 |
| 引用块 | `>` 前缀行 |
| 分隔线 | `---` / `***` / `___` |
| 列表 | 整个列表，含嵌套子项 |
| 独立图片 | `![alt](url)` 或 Obsidian `![[path]]` |
| 段落 | 其他连续非空行 |

代码块、数学公式、表格始终作为完整单元保留，不会被从中间截断。

**第二阶段 — Heading-Aware 滑动窗口（`chunk_markdown`）**：将 block 组合为 chunk（默认每个 chunk 包含 4 个 block），遵循两条结构性规则：

- **Rule 1 — 不孤立标题**：若当前 chunk 末尾是标题，自动多包一个 body block，确保标题不会单独出现在 chunk 末尾。
- **Rule 2 — 标题锚定起点**：下一个 chunk 从最近的前置标题开始，而不是从随机段落开始，使每个 chunk 都携带所属 section 的上下文。

其他文件类型继续使用基于 token 的滑动窗口（`chunk_text`）。

详细设计见 [docs/markdown-chunking.md](docs/markdown-chunking.md)。

### 默认参数

- Chunk size: `256` tokens（使用 embedding 模型的 tokenizer）
- Chunk overlap: `32` tokens
- 搜索 `top_k`: `5`
- 默认 rerank candidates: `20`
- collection 数量上限: `4`
- collection 名称最大长度: `64`
- 默认端口: `8000`

## 技术栈

- **API 框架**: FastAPI
- **向量数据库**: LanceDB（嵌入式列存，自带 tantivy FTS）
- **推理后端**: fastembed (基于 ONNX, 支持 `ThreadPoolExecutor` 并行化)
- **元数据存储**: SQLite
- **文件监听**: 定期轮询 (Polling)

关键词检索由 LanceDB 自带的 tantivy FTS 提供，不再单独加载稀疏 embedding
模型。存储布局、索引策略、查询语义见 [docs/lancedb.md](docs/lancedb.md)。

## 性能与内存

Hermit 针对本地搜索的稳定内存占用进行了优化：
- **串行搜索请求**：搜索请求通过单 worker executor 串行执行，避免多个 reranker 请求同时占用共享 ONNX session。
- **限制 ONNX 推理线程**：默认使用 `HERMIT_ONNX_THREADS=2`，避免 ONNX Runtime 每线程 arena 累积带来的常驻内存膨胀；仅在测得单请求延迟收益足够时再调大。
- **缩小重排候选池**：默认每次查询使用 20 个候选，并保留 Cross-Encoder reranker 精排。
- **嵌入缓存**：索引时若某个 chunk 的模型输入文本之前已经算过（sha256 keyed by `model_name::input_text`），直接复用磁盘缓存里的向量，跳过 ONNX 推理。缓存路径 `HERMIT_HOME/cache`，TTL 7 天硬编码；命中时校验向量维度，不合法当未命中处理（模型升级 / 旧脏数据自愈）。设计上默认开启、不提供关闭开关。

## 项目结构

```text
.
├── main.py
├── pyproject.toml
├── README.md / README_cn.md
├── docs/
│   ├── design.md
│   ├── markdown-chunking.md
│   ├── lancedb.md
│   └── skill-distribution.md
├── hermit/
│   ├── app.py                 # FastAPI 应用 + lifespan
│   ├── cli.py                 # CLI 入口（JSON 输出）
│   ├── config.py              # 路径、默认值、环境变量
│   ├── models.py              # 模型下载与校验
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── ingestion/
│   │   ├── chunker.py         # markdown / token 切片
│   │   ├── scanner.py
│   │   ├── task_queue.py
│   │   └── watcher.py
│   ├── retrieval/
│   │   ├── embedder.py
│   │   ├── embed_cache.py     # diskcache 落地的 per-model 向量缓存
│   │   ├── reranker.py
│   │   └── searcher.py
│   └── storage/
│       ├── lance.py           # LanceDB 向量存储
│       ├── metadata.py        # 每 collection 一个 SQLite
│       ├── model_signature.py
│       ├── quantizer.py       # dense embedder 的 INT8 量化
│       └── registry.py
└── tests/

~/.hermit/                     # 运行时数据（可用 HERMIT_HOME 覆盖）
├── models/                    # ONNX 权重（fastembed cache）
├── cache/dense/               # dense 嵌入向量缓存（sha256 key，TTL 7 天）
├── data/
│   ├── lance/                 # LanceDB 表（每个 collection 一张）
│   ├── metadata/              # 每 collection 一个 SQLite
│   ├── collections.json       # 注册表
│   └── model_signature.json
├── logs/hermit.log
└── hermit.pid
```

## 安装

### 环境要求

- Python `3.12 ~ 3.13`
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

可选参数：

```bash
hermit kb add my_docs ./documents \
  --ignore "build/**" --ignore "*.tmp" \
  --ignore-ext .pdf --ignore-ext .png
```

查看 collection 列表：

```bash
hermit kb list
```

更新忽略规则：

```bash
hermit kb update my_docs --ignore "dist/**" --ignore-ext .log
hermit kb update my_docs --clear-ignore --clear-ignore-ext
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
hermit start
```

启动时，Hermit 会：

- 预热 embedding 和 reranker 模型
- 启动后台索引 worker
- 从 `~/.hermit/data/collections.json` 恢复已持久化的 collection
- 扫描每个 collection 目录
- 启动目录监听

默认监听地址：

- Host: `0.0.0.0`
- Port: `8000`

其他服务命令：

```bash
hermit status     # 健康 JSON（模式、collection、待索引任务等）
hermit logs       # 流式查看服务日志
hermit stop       # 优雅停服
```

### 4. 搜索

推荐用 CLI（JSON 输出，免去 curl 拼装）：

```bash
hermit search my_docs "two sum 的思路"
hermit search my_docs "lancedb" --mode keyword          # FTS only，更快
hermit search my_docs "二分"   --mode fuzzy             # 子串扫描
hermit search my_docs "design" --mode fuzzy --filename "*.md"
hermit search my_docs "embedding" --no-rerank --top-k 3
```

HTTP 等价调用：

```bash
curl -X POST http://127.0.0.1:8000/search \
	-H 'Content-Type: application/json' \
	-d '{
		"query": "two sum 的思路",
		"collection": "my_docs",
		"top_k": 5,
		"rerank_candidates": 20,
		"mode": "hybrid"
	}'
```

`mode` 取值：`hybrid`（默认）、`semantic`、`keyword`、`fuzzy`。

## CLI

所有命令默认输出 JSON 到 stdout。加 `--pretty` 可得到缩进格式。错误以 `{"error": "message"}` 形式输出，退出码非零。

### 服务生命周期

| 命令 | 用途 |
|---|---|
| `hermit start` | 后台启动服务（uvicorn daemon）。 |
| `hermit stop` | 优雅停止（SIGTERM，10 秒后退化为 SIGKILL）。 |
| `hermit status` | 健康 JSON：模式、uptime、collections、待处理任务等。 |
| `hermit logs` | 流式查看 `~/.hermit/logs/hermit.log`（非 JSON）。 |

### `hermit download`

下载所有所需模型，并可选执行基础验证。

参数：

- `--force`：强制重新下载
- `--skip-verify`：跳过下载后的验证

### `hermit search <collection> [<query>]`

语义 / 关键词 / 模糊检索，JSON 输出。

参数：

- `--mode {hybrid,semantic,keyword,fuzzy}` — 默认 `hybrid`
- `--top-k N` — 返回数量（默认 5）
- `--rerank-candidates N` — 精排前的召回候选池大小（默认 20）
- `--filename PATTERN` — 文件名过滤（子串或 glob），正交参数
- `--rerank` / `--no-rerank` — 显式覆盖各模式默认的 rerank 行为

`fuzzy` 模式下，当 `--filename` 给出时，`query` 可省略。

### `hermit kb add <name> <dir>`

将目录注册为 collection。

参数：

- `--ignore PATTERN` — glob 形式路径忽略（可重复）
- `--ignore-ext EXT` — 后缀忽略，例如 `.pdf`（可重复）

### `hermit kb update <name>`

替换某个 collection 的忽略规则（替换语义，非追加）。

参数：

- `--ignore PATTERN` — 新的路径忽略列表（替换已有配置）
- `--ignore-ext EXT` — 新的后缀忽略列表（替换已有配置）
- `--clear-ignore` — 清空所有路径忽略模式
- `--clear-ignore-ext` — 清空所有后缀忽略规则

### `hermit kb remove <name>`

删除 collection 及其元数据。

### `hermit kb list`

列出所有已注册 collection。

### `hermit collection <subcommand> <name>`

查询正在运行的服务的实时 collection 状态（需要 `hermit start` 已经启动）。

- `hermit collection status <name>` — 索引状态（`indexed_files` / `total_chunks` / `watching`）
- `hermit collection sync <name>` — 手动触发同步扫描
- `hermit collection tasks <name>` — 后台任务队列状态

### `hermit install-skills`

将内置的 agent skill 安装到 `~/.agents/skills/`。

- `--uninstall` 反向卸载已安装的 skill

## HTTP API

当前代码实现了以下接口。

### `POST /search`

执行混合 / 语义 / 关键词 / 模糊检索。

请求示例：

```json
{
	"query": "滑动窗口最大值",
	"collection": "my_docs",
	"top_k": 5,
	"rerank_candidates": 20,
	"mode": "hybrid",
	"filename": null,
	"rerank": null
}
```

`mode`：`hybrid`（默认） / `semantic` / `keyword` / `fuzzy`。`filename` 支持子串或 glob（例如 `*.md`）。`rerank` 用来覆盖该模式的默认 rerank 行为（`null` 走模式默认；`false` 跳过；`true` 强制开启）。

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

### `POST /collections`

注册一个新的 collection。服务端异步扫描该目录并持久化到注册表，重启后自动恢复。

请求：

```json
{
	"name": "my_docs",
	"folder_path": "/abs/path/to/docs",
	"ignore_patterns": ["build/**"],
	"ignore_extensions": [".pdf"]
}
```

冲突时返回 `409`，非法名称 / 路径返回 `400`。

### `DELETE /collections/{name}`

删除 collection：停掉文件监听 → 等后台索引队列清空 → 删 LanceDB 表 → 删 SQLite 元数据 → 从注册表移除。

若 30 秒内后台索引任务无法清空，返回 `409`，稍后重试。

### `POST /collections/{name}/sync`

手动触发某个 collection 的扫描同步。返回 `{ "added": N, "updated": M, "deleted": K }`。

### `GET /collections/{name}/status`

返回 `{ name, folder_path, indexed_files, total_chunks, watching }`。

### `GET /collections/{name}/tasks`

返回 `{ collection, pending_tasks, queued_tasks, in_progress_tasks, worker_alive }`。

### `GET /health`

服务健康状态与运行信息。

返回字段：

- `status` — `"ready"` 或 `"starting"`
- `uptime` — 服务启动后经过的秒数
- `models_loaded` — 模型是否已加载完成
- `collections` — 各 collection 的 `{name, indexed_files, total_chunks}`
- `pending_index_tasks` — 全部 collection 待处理的后台索引任务总数
- `storage` — 固定为 `"lance"`（LanceDB 嵌入式存储）

## 数据存储

Hermit 默认把所有运行时数据存到 `~/.hermit/`（可用 `HERMIT_HOME` 环境变量覆盖）：

- `~/.hermit/models/`: 本地模型缓存（fastembed ONNX 权重）
- `~/.hermit/data/lance/`: LanceDB 表，每个 collection 一张
- `~/.hermit/data/metadata/`: 每个 collection 一个 SQLite 数据库
- `~/.hermit/data/collections.json`: collection 持久化配置
- `~/.hermit/cache/dense/`: dense 嵌入向量缓存（sha256 keyed，TTL 7 天）
- `~/.hermit/logs/hermit.log`：服务日志，达到 256 MiB 时轮转，并保留一个备份（`hermit.log.1`）
- `~/.hermit/hermit.pid` 和 `~/.hermit/port.json`: daemon 状态文件

集中在单一目录，便于备份、迁移和清理。

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
- **删除文件**：从 LanceDB 和 SQLite 中移除

### 分块规则

- 默认 chunk 大小为 `256` tokens（使用 embedding 模型自带的 tokenizer 计数）
- 相邻 chunk 重叠 `32` tokens
- 空白文本会被跳过
- 短文本保持单 chunk

## 已知限制

- 混合检索使用 LanceDB 内置的 RRF 融合，不再暴露 `w_dense`/`w_sparse` 显式权重
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
