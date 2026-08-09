---
name: hermit-search
description: 'Install, set up, upgrade, configure, troubleshoot, and use Hermit for local knowledge-base search. Supports four search modes (hybrid, semantic, keyword, fuzzy), filename filtering, collection management, and file indexing. Use when: installing Hermit or the hermit-search skill, searching or querying local documents, managing collections, adding a knowledge base, or indexing files.'
---

# Skill: hermit-search

Hermit 本地知识库检索服务使用指南。Hermit 提供四种搜索模式（hybrid / semantic / keyword / fuzzy）+ 正交的文件名过滤，支持多知识库管理。

## 搜索语义一览

| 模式 | 召回 | Rerank（默认） | 适用场景 |
|---|---|---|---|
| `hybrid`（默认） | dense + FTS（BM25），LanceDB RRF 融合 | ✅ | 通用查询，最强相关性 |
| `semantic` | dense only | ✅ | "我大概记得意思但不记得用词" |
| `keyword` | FTS（tantivy BM25）only | ❌ | 关键词精确命中、最快 |
| `fuzzy` | SQL `contains` 子串扫描 | ❌ | 找包含某子串的文件或段落、按文件名查找 |

正交过滤参数：

- `--filename PATTERN` — 子串（`design`）或 glob（`*.md`、`*notes*`），匹配文件名 stem 或完整路径
- `--rerank` / `--no-rerank` — 显式覆盖各模式默认的 rerank 行为

## 平台支持

- **macOS**：已测试通过
- **Linux**：即将支持（短期内增加测试）

## 前置条件

### 环境要求

- Python 3.12 ~ 3.13

### Agent 安装流程

先根据当前上下文选择安装来源：

- **Agent 正在 Hermit repo 内工作**：严格遵循根目录 `AGENTS.md` 的“CLI 与 Skill 安装”，安装并核验当前 checkout，不执行下面的远端安装命令。
- **Agent 在其他工作区**：从 Hermit GitHub 仓库安装：

```sh
uv tool install git+https://github.com/xxxgqcoder/hermit.git --force
```

随后部署全局 Skill：

```sh
hermit install-skills
```

- 在 Hermit repo 内，`.agents/skills/hermit-search/` 已可被当前项目上下文发现，全局部署可选。
- 要让其他工作区发现 `hermit-search`，必须执行 `hermit install-skills`。

安装后验证：

```sh
command -v hermit
test -f ~/.agents/skills/hermit-search/SKILL.md
```

`hermit install-skills` 应返回包含 `"status": "installed"` 和 `"hermit-search"` 的 JSON。若当前 Agent 不会动态刷新 Skill 列表，告知用户新开任务或重启会话后再使用。

仅安装 CLI 和 Skill 时，不要自动执行模型下载或启动服务；这两步只在用户准备实际检索或明确要求时执行。

### 模型下载

首次使用前需下载模型（约 1.3GB，dense + reranker）：

```sh
hermit download
```

## 使用流程

### 1. 启动服务

```sh
hermit start
```

Hermit 使用嵌入式 **LanceDB** 作为向量存储，无需 Docker、无外部进程。启动时会预加载 dense embedding 和 reranker 模型；两者都带闲置自动卸载机制以释放 ONNX arena（实测 reranker 可回收 ~1.1 GB、dense ~0.6 GB 常驻），下次请求懒重载（冷启 ~0.3-1s）。默认阈值：reranker 5 分钟、dense 30 分钟。

输出示例：`{"status": "started", "pid": 12345, "port": 8000}`

通过 `GET /health`（或 `hermit status`）的 `storage` 字段固定返回 `"lance"`，表明存储后端。

### 2. 内存与并发策略

Hermit 的搜索请求**默认串行执行**——多个请求共享同一份 ONNX session 和 cross-encoder reranker，串行化能压住本地常驻和峰值内存。如需 deep-search 类工作流（一次 query 扇出多次召回），可通过 `HERMIT_SEARCH_WORKERS` 调大并发：

```sh
HERMIT_SEARCH_WORKERS=4 hermit start   # 默认 1（串行）；调大会按比例放大峰值内存
```

单次 ONNX 推理的内部线程数可通过 `HERMIT_ONNX_THREADS` 调整，默认 `2`（ONNX Runtime 每线程会保留独立 arena，调大主要换来内存膨胀，仅在测得延迟收益时再调大）：

```sh
HERMIT_ONNX_THREADS=2 hermit start
```

模型闲置卸载阈值可通过环境变量微调（设为 `0` 关闭对应模型的闲置卸载）：

```sh
HERMIT_RERANKER_IDLE_TIMEOUT=300        # 默认 300s
HERMIT_RERANKER_IDLE_CHECK_INTERVAL=60  # 后台检查频率，默认 60s
HERMIT_DENSE_IDLE_TIMEOUT=1800          # 默认 1800s（dense 调用更频繁，阈值更长）
HERMIT_DENSE_IDLE_CHECK_INTERVAL=120    # 后台检查频率，默认 120s
```

### 3. 添加知识库

将一个文件夹注册为知识库 collection：

```sh
hermit kb add <name> <directory> [--ignore <glob>]... [--ignore-ext <ext>]...
```

- `name`：collection 别名（字母数字 + 下划线/连字符，以字母或数字开头）
- `directory`：文件夹路径，Hermit 会递归扫描并索引其中的文本文件
- `--ignore`：glob 模式，匹配的**相对路径**将被忽略（可重复指定多个）
- `--ignore-ext`：文件后缀名，匹配的文件将被忽略（大小写不敏感，可重复指定多个）

示例：

```sh
# 基本用法
hermit kb add my-notes ~/Documents/notes

# 忽略特定路径和后缀
hermit kb add my-project ~/code/project \
  --ignore "build/**" \
  --ignore "*.tmp" \
  --ignore "node_modules/*" \
  --ignore-ext .pdf \
  --ignore-ext .png
```

### 4. 更新知识库忽略规则

修改已有知识库的忽略配置（替换模式，非追加）：

```sh
hermit kb update <name> [--ignore <glob>]... [--ignore-ext <ext>]... [--clear-ignore] [--clear-ignore-ext]
```

- `--ignore`：设置新的路径忽略 glob 模式（替换已有配置）
- `--ignore-ext`：设置新的后缀忽略规则（替换已有配置）
- `--clear-ignore`：清除所有路径忽略模式
- `--clear-ignore-ext`：清除所有后缀忽略规则

示例：

```sh
# 更新忽略模式
hermit kb update my-project --ignore "dist/**" --ignore "*.log"

# 清除所有忽略规则
hermit kb update my-project --clear-ignore --clear-ignore-ext
```

### 5. 搜索

```sh
hermit search <collection> [<query>] [--mode ...] [--filename PATTERN] [--top-k N]
```

参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `collection` | str | — | Collection 名称 |
| `query` | str | "" | 搜索查询；`fuzzy` 模式下若有 `--filename` 则可省略 |
| `--mode` | str | `hybrid` | `hybrid` / `semantic` / `keyword` / `fuzzy` |
| `--filename` | str | — | 文件名子串或 glob（`*?[]`），匹配 basename / 完整路径 |
| `--rerank` | flag | — | 强制开启 cross-encoder 精排（覆盖模式默认） |
| `--no-rerank` | flag | — | 跳过精排（覆盖模式默认） |
| `--top-k` | int | 5 | 返回结果数 |
| `--rerank-candidates` | int | 20 | 精排前的召回候选池大小 |

#### 模式选择建议

- 默认场景或没把握时 → `hybrid`
- 查询是完整自然语言描述、希望命中"意思接近"的内容 → `semantic`
- 查询是产品名、人名、技术术语、API 名等专有名词 → `keyword`（更快）
- 想找"哪些文件/段落里出现过 X 子串"、或按文件名查找文件 → `fuzzy`

#### 示例

```sh
# 默认混合检索
hermit search my-notes "如何实现二分查找"

# 纯语义检索（"模糊语义"，去掉 BM25 的精确词限制）
hermit search my-notes "存储层架构演进" --mode semantic

# 关键词模式（更快，跳过 rerank）
hermit search my-notes "lancedb" --mode keyword

# Fuzzy 子串：找所有提到 "二分" 的段落
hermit search my-notes "二分" --mode fuzzy

# 仅按文件名查找文件（fuzzy 下 query 可省略）
hermit search my-notes --mode fuzzy --filename design

# 文件名过滤 + 任意模式（正交叠加）
hermit search my-notes "embedding" --filename "docs/*.md"
hermit search my-notes "向量数据库" --mode hybrid --filename design

# 显式开/关精排
hermit search my-notes "lancedb" --mode keyword --rerank      # 关键词模式也走 rerank
hermit search my-notes "向量数据库" --no-rerank                # 默认 hybrid 不走 rerank
```

### 6. 其他管理命令

```sh
# 查看所有知识库
hermit kb list

# 删除知识库
hermit kb remove <name>

# 触发同步（需要服务运行中）
hermit collection sync <name>

# 查看索引状态
hermit collection status <name>

# 查看索引任务队列
hermit collection tasks <name>
```

### 7. 服务生命周期

```sh
hermit status    # 查看服务状态
hermit stop      # 停止服务
hermit logs      # 查看日志（流式输出）
```

## 输出格式

所有命令输出 JSON。添加 `--pretty` 获取格式化输出：

```sh
hermit --pretty search my-notes "query"
```

错误输出格式：`{"error": "message"}`

## 忽略规则说明

- 路径模式（`--ignore`）使用 glob 语法（与 `.gitignore` 类似），匹配相对于知识库根目录的路径
  - `*.log` — 忽略根目录下所有 `.log` 文件
  - `build/**` — 忽略 `build/` 目录下所有文件
  - `**/temp/*` — 忽略任意层级下的 `temp/` 目录内容
- 后缀模式（`--ignore-ext`）大小写不敏感：`.PDF` 和 `.pdf` 等价
- 隐藏文件（以 `.` 开头的目录或文件）始终被忽略，无需额外配置
- `hermit kb list` 可查看每个知识库当前的忽略配置

## 技术细节

- **Embedding 模型**：jinaai/jina-embeddings-v2-base-zh（768 维 dense）
- **Reranker**：jinaai/jina-reranker-v2-base-multilingual（cross-encoder，闲置 5 分钟自动卸载）
- **向量数据库**：LanceDB（嵌入式列存，自带 tantivy FTS 提供 BM25 关键词召回）
- **推理后端**：fastembed（ONNX Runtime，纯 CPU）
- **数据目录**：`~/.hermit/`（可通过 `HERMIT_HOME` 环境变量覆盖）
  - `data/lance/` — 每个 collection 一张 LanceDB 表
  - `data/metadata/` — 每个 collection 一个 SQLite 元数据库
- **存储后端查询**：`GET /health` 的 `storage` 字段固定返回 `"lance"`

## 搜索模式实现细节

- **hybrid**：LanceDB 内置 `query_type="hybrid"`，dense 向量召回 + FTS 召回各自取 `rerank_candidates` 个候选，`RRFReranker` 按倒数排名融合，cross-encoder 精排取 top-k
- **semantic**：仅 dense 向量召回 → cross-encoder 精排
- **keyword**：仅 FTS（tantivy BM25）召回，默认跳过精排（追求速度）。需要时用 `--rerank` 显式启用
- **fuzzy**：不走任何向量或 BM25 索引；通过 LanceDB SQL `contains(lower(text), '<needle>')` 谓词扫描，DataFusion 下推到列文件，扫描成本与命中数线性相关。按 `(source_file, chunk_index)` 排序后截 top-k。结果不带分数（`score=null`）
- **filename 过滤**：纯子串走 LanceDB SQL `contains(lower(filename), '<needle>')` 作为 prefilter 下推；含 `*?[` 的 glob 走 Python `fnmatch` 后过滤，可同时匹配 basename 或完整路径
