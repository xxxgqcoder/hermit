
## 分层结构：

1. **Ingestion 层**：文件夹扫描 + 定期轮询 → 变更检测（SQLite）→ 文本切片（token 级 / markdown 语义级）→ dense 向量化（带磁盘缓存）→ 入库
2. **检索层 (Recall)**：Dense 向量 + LanceDB 原生 FTS 双路召回，LanceDB 内置 RRF 融合；另支持纯语义 / 纯关键词 / 子串 fuzzy 三种单路模式
3. **精排层 (Rerank)**：Cross-Encoder 处理 Top 20 候选集（`DEFAULT_RERANK_CANDIDATES`）
4. **服务层**：FastAPI 暴露 OpenAPI 接口

支持**多知识库**（上限 `MAX_COLLECTIONS=4`）：每个文件夹 = 一个独立的 LanceDB 表。

### 推理后端

使用 **fastembed**（基于 ONNX Runtime）作为 dense embedding 与 reranker 的推理后端。模型均以 ONNX 格式 CPU 推理，无需 PyTorch 或 GPU 依赖。关键词检索由 LanceDB 自带的原生 FTS（inverted index）提供，不再单独加载 sparse embedding 模型。

选型理由：
- 依赖轻量，无需安装 PyTorch（节省数 GB 磁盘和安装时间）
- ONNX Runtime 在 Apple Silicon 上具有良好的 CPU 推理性能
- 自动管理模型下载和缓存

**约束**：模型选择受限于 fastembed 支持的 ONNX 预转换模型列表。

#### INT8 动态量化

两个模型在首次启动时由 `hermit/storage/quantizer.py` 用 `onnxruntime.quantization.quantize_dynamic`（`weight_type=QInt8`）做**动态量化**，量化产物落在 `~/.hermit/models/quantized/{repo_slug}/onnx/model.onnx`。`embedder.py` / `reranker.py` 加载时优先指向量化权重（`specific_model_path`），在精度几乎无损的前提下显著降低权重内存与磁盘占用。启动时 `app.py` lifespan 调用 `ensure_quantized_models()` 确保量化产物就绪。

#### ONNX Runtime arena 与线程控制

ONNX Runtime 的 arena allocator 会把 `MALLOC_LARGE` 高水位逐步顶高且**不归还 OS**（只有销毁整个 InferenceSession 才释放——这正是 idle-unload 的依据）。因此默认 **关闭 arena**：

| 配置 | 默认 | 说明 |
|---|---|---|
| `HERMIT_ONNX_ARENA` | `false` | arena 关闭，每次 `Run()` 走 plain malloc：单次慢几 %，但 RSS 在突发与累积索引下保持平直；设 `true` 恢复旧 fastembed 默认 |
| `HERMIT_ONNX_THREADS` | `2` | 每个 ONNX session 的 intra/inter-op 线程数。ORT 保留 per-thread arena，多线程会按数十 MB 抬升常驻内存而延迟收益有限 |

arena 开关通过 `hermit/retrieval/fastembed_patch.py` 给 fastembed 暴露 `enable_cpu_mem_arena` / `enable_mem_pattern` 注入点。背景测量见 `problems/concurrent-search-rss-blowup.md`、`problems/dense-embedder-arena-creep.md`。

---

## Ingestion Pipeline（写入路径）

### 数据源

- 指定文件夹作为知识库唯一来源
- 文件夹内文件统一视为文本文件处理（UTF-8, errors=replace）
- 多模态文件（PDF、图片等）的解析为文本格式由外部流程负责，不在本服务范围内
- 跳过隐藏文件（路径中含 `.` 开头的部分）

### 触发方式

- **启动时**：全量扫描，逐文件对比 hash 与 SQLite 记录，增量更新索引
- **运行时**：后台线程定期轮询扫描（默认每 15 分钟，`HERMIT_POLL_INTERVAL`），逐文件对比 hash 检测变更并触发增量更新。当前实现为纯轮询（`ingestion/watcher.py` 的 `_PollingWatcher`），不使用 OS 文件事件

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

**选用 SQLite 的理由**：启动时对每个文件查 LanceDB 效率低；SQLite 单次全表扫描即可完成对比，天然适合存储关系型元数据。

### 文本切片

按文件类型分派（`scanner.py`：`.md` 走 `chunk_markdown`，其余走 `chunk_text`）：

- **非 Markdown（`chunk_text`）**：用 embedding 模型自带 tokenizer 按 token 数切片 + 滑动窗口重叠，默认 `chunk_tokens=256`、`overlap_tokens=32`；用模型 tokenizer 计数，消除中英文字符密度差异；短文本（≤ chunk_tokens）不切分，空文本跳过。
- **Markdown（`chunk_markdown`）**：先用正则把文档解析成语义 block（标题 / 列表 / 代码围栏 / 数学块 / 表格 / 引用 / 段落等），再按 heading-aware 滑窗（默认 4 block/chunk、overlap 1，且不孤立标题）成块。代码围栏、表格、公式等作为**不可分割的整体**参与分块，避免按固定 token 硬切打断语义。细节见 `docs/markdown-chunking.md`。

**向量化增强**：Embedding 时会将文件名作为标题拼接到切片内容前，格式为 `[{title}]\n{chunk}`，以增强语义召回。

### 嵌入缓存

dense 向量经 `hermit/retrieval/embed_cache.py`（`diskcache` 落地，存于 `~/.hermit/cache/dense/`）缓存，key 为 `sha256("{model}::{text}")`（`text` 即拼接标题后的实际模型输入），按模型名命名空间，避免换模型污染缓存。命中时做维度/长度校验（自愈），TTL **30 天**（`EMBED_CACHE_TTL_SECONDS`）。无环境开关，按设计常开。

### 索引流程

1. 递归扫描文件夹（`rglob("*")`），逐文件对比 SHA256 与 SQLite 记录
2. **新增/修改文件**：切片 → dense 向量化（命中缓存则跳过推理）→ 按 `source_file` 删除旧 chunks → 插入新 chunks（UUID 作为行 id）→ 更新 SQLite。`replace_file_chunks` 完成后调用 `tbl.optimize(cleanup_older_than=1min)` 让 FTS 索引立即包含新写入的行，同时回收瞬时旧版本（见下文压缩策略）
3. 索引任务经 `ingestion/task_queue.py` 的后台线程池执行（`HERMIT_INDEX_WORKERS`，默认 1）
4. 后台轮询线程（`_PollingWatcher`）按 `POLL_INTERVAL_SECONDS` 周期重跑 `scan_folder`，发现增删改时入队索引

---

## 存储 Schema（LanceDB）

每个知识库 = 1 张 LanceDB 表，磁盘布局为 `~/.hermit/data/lance/<name>.lance`。LanceDB 是嵌入式列存（基于 Apache Arrow），通过 `lancedb.connect(path)` 直接打开，单进程多线程并发安全，无独立服务。

### 表结构

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | string | UUID4，每个 chunk 一个 |
| `text` | string | 切片原文 |
| `title` | string | 文件名（含原始大小写） |
| `filename` | string | 小写化的 basename stem，便于 FTS / `contains()` 子串过滤 |
| `source_file` | string | 源文件绝对路径 |
| `chunk_index` | int32 | 0-based 切片序号 |
| `total_chunks` | int32 | 该源文件的切片总数 |
| `vector` | fixed_size_list<float32, 768> | jina-embeddings dense 向量 |

### 索引策略

| 索引 | 列 | 时机 | 说明 |
|---|---|---|---|
| FTS（LanceDB 原生 inverted index，`use_tantivy=False`） | `text` | `ensure_collection` 即建 | keyword / hybrid 检索 |
| FTS（LanceDB 原生 inverted index，`use_tantivy=False`） | `filename` | `ensure_collection` 即建 | filename 子串过滤 |
| Scalar BTREE | `source_file` | `ensure_collection` 即建 | 按文件删除 |
| 向量 IVF/HNSW | `vector` | `count_rows() ≥ 50_000` 后懒建 | 小规模下裸扫更快 |

### 压缩与版本回收

LanceDB 默认为每次写入保留旧版本 **7 天**。在一次索引突发里（数百文件 × N 轮重建），这会让磁盘数据集膨胀到逻辑大小的几百倍才被清理。对此：

- **每次 `replace_file_chunks` 的 `optimize()` 都带 `cleanup_older_than=1min`**（`_OPTIMIZE_CLEANUP_OLDER_THAN`）——比单次写入耗时长、又短到能在同一突发内回收瞬时版本，最新版本永不删除。
- **启动时 `compact_collection()` 一次性激进压缩**（`cleanup_older_than=0`），清理旧版本 Hermit 留下的历史版本垃圾；版本数 ≤2 时直接跳过，幂等且廉价。

### 查询文件全部切片

通过 `source_file` 列上的 SQL `where` 谓词查询，配合 `chunk_index` 排序还原原文顺序。`total_chunks` 字段让调用方判断是否获取完整。

---

## 检索与融合

### 搜索模式

`searcher.py` 暴露四种模式（`SEARCH_MODES`，默认 `hybrid`）：

| 模式 | 召回方式 | 融合/排序 |
|---|---|---|
| `hybrid` | dense 向量 + 原生 FTS 双路 | LanceDB `RRFReranker()` 融合 |
| `semantic` | 仅 dense 向量 | 向量距离 |
| `keyword` | 仅原生 FTS | BM25/FTS 分 |
| `fuzzy` | `text`/`filename` 上的 SQL `contains()` 子串扫描 | 子串命中 |

`hybrid` 内部并行执行向量召回与 FTS 召回，再用 Reciprocal Rank Fusion 融合两个排序，取 Top-N（`DEFAULT_RERANK_CANDIDATES=20`）候选送入 cross-encoder 精排。

`w_dense`/`w_sparse` 显式权重不再暴露——LanceDB `RRFReranker` 仅按排序融合，没有等价的加权 API。

### filename 过滤

正交的文件名过滤（与四种模式叠加）：

- 纯子串：SQL `contains(lower(filename), '...')`，下推到 LanceDB
- glob 模式（含 `*?[`）：表扫描后用 Python `fnmatch` 后过滤

---

## 模型选型

### 模型清单

| 模型 | 用途 | 源模型大小 | 说明 |
|---|---|---|---|
| `jinaai/jina-embeddings-v2-base-zh` | Dense Embedding (768 维) | ~0.64GB | 中英双语语义向量；服务时以 INT8 量化 ONNX 加载 |
| `jinaai/jina-reranker-v2-base-multilingual` | Reranker (Cross-Encoder) | ~0.7GB | 多语言 reranker；服务时以 INT8 量化 ONNX 加载 |

模型下载由 `hermit/models.py` 经 `huggingface_hub.snapshot_download` 拉取到 `~/.hermit/models/`，再由 quantizer 产出 INT8 版本。

---

## 精排（Reranker）

| 维度 | 说明 |
|---|---|
| 模型 | `jinaai/jina-reranker-v2-base-multilingual`（INT8 量化） |
| 架构 | Cross-Encoder |
| 接口 | fastembed `TextCrossEncoder` |
| 语言 | 多语言（中英均适用） |
| 候选数 | `DEFAULT_RERANK_CANDIDATES=20` |

### 实现细节

- Reranker 对候选集重新排序，返回排序后的 **索引列表**
- 最终返回的 `score` 字段优先取 LanceDB 的 `_relevance_score`（hybrid + rerank 后的 RRF 分数），其次 `_distance` 或 `_score`

---

## Idle Unload（双模型）

两个 ONNX session（dense embedder、reranker）各自有后台线程在闲置超过阈值后销毁实例并 `gc.collect()`，把 ONNX session 持有的内存整块还给 OS（arena 关闭的前提下，这是回收 `MALLOC_LARGE` 高水位的主要手段）。下次请求懒重载。

| 模型 | 闲置阈值 env | 默认 | 检查频率 env | 默认 | 冷启动 |
|---|---|---|---|---|---|
| Reranker | `HERMIT_RERANKER_IDLE_TIMEOUT` | `300`s（5min） | `HERMIT_RERANKER_IDLE_CHECK_INTERVAL` | `60`s | ~1-3s |
| Dense embedder | `HERMIT_DENSE_IDLE_TIMEOUT` | `1800`s（30min） | `HERMIT_DENSE_IDLE_CHECK_INTERVAL` | `120`s | ~0.2-0.5s |

阈值差异的理由：dense 被每次搜索（query 编码）和每个索引批次触碰，工作时段基本常热，30 分钟阈值只在真正安静的间隙（隔夜、编辑突发之间）触发以回收索引累积的激活池；reranker 仅搜索时用，5 分钟即可快速回收。任一阈值设为 `0`（或负）关闭对应的闲置卸载。

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
- `MODEL_ROOT = HERMIT_HOME / "models"`（含原始权重 + `quantized/` 量化产物）
- `DATA_ROOT = HERMIT_HOME / "data"`（含 LanceDB 表 + SQLite 元数据）
- `CACHE_ROOT = HERMIT_HOME / "cache"`（dense 嵌入缓存）
- 所有模型加载均基于此路径，不在各模块中硬编码
- fastembed 的 `cache_dir` 参数指向 `MODEL_ROOT`

### 下载策略

- **服务启动时自动检测**：模型缺失则经 `huggingface_hub` 自动下载，随后 `ensure_quantized_models()` 产出 INT8 量化版本
- `app.py` lifespan 中调用 `warmup()` 预加载 embedding 和 reranker 模型
- 也可通过 `hermit download` 命令提前下载

---

## 服务层（FastAPI）

单进程，模型启动时预加载。`app.py` 使用 `asynccontextmanager` lifespan 管理启动/关闭。知识库管理（注册、删除、更新忽略规则）既可以通过 CLI（`hermit kb add/remove/update`），也可以通过 HTTP（`POST /collections` / `DELETE /collections/{name}`）。CLI 在服务运行时会优先转发到 HTTP API；服务未运行时直接操作本地注册表。

### 检索流程

1. fastembed 编码 query → dense 向量（命中缓存则跳过）
2. LanceDB 同时跑向量召回 + FTS 召回，RRFReranker 融合 → Top 候选
3. Cross-Encoder 对候选集精排
4. 返回 Top-K（`DEFAULT_TOP_K=5`）结果（含切片原文、来源文件路径、切片位置、融合分数）

### 并发模型

- **搜索**：`app.py` 内 `ThreadPoolExecutor`，大小 `HERMIT_SEARCH_WORKERS`（默认 1，串行）。ONNX session 线程安全且推理时释放 GIL，agent 式 deep-search 可调高并发，但每个并发 run 会抬升 arena 峰值——需配合 reranker idle-unload 回收。
- **索引**：`task_queue.py` 线程池，大小 `HERMIT_INDEX_WORKERS`（默认 1，避免与搜索争用共享 ONNX session；初次批量索引可调高）。

### 持久化与启动恢复

- **Collection 注册表持久化**：`~/.hermit/data/collections.json` 记录所有已注册知识库（folder_path、ignore_patterns、ignore_extensions）。服务启动时自动加载并恢复所有 collection，无需重新注册。
- **轮询 watcher 自动恢复**：服务启动时为每个已注册 collection 自动启动后台轮询线程。
- **模型变更检测**：`~/.hermit/data/model_signature.json` 记录上次使用的 dense embedding 模型。若模型发生变更，启动时自动触发所有 collection 的全量重建索引。

---

## 内存预算

稳态内存由三项设计共同压低：**arena 默认关闭**（RSS 不随突发/累积索引爬升）+ **dense / reranker 双 idle-unload**（安静期把整块 session 内存还给 OS）+ **INT8 量化**（权重减半）。

| 状态 | 典型 RSS | 说明 |
|---|---|---|
| 闲置（两模型均已 unload） | **~320MB** | 仅 Python + FastAPI + LanceDB + 常驻 |
| 活跃峰值 | **~7GB** | 重并发搜索 + 索引同时进行；单 worker 串行下远低于此 |

并发缩放代价：搜索峰值大致随 worker 线性增长（`peak ≈ 3.5 × HERMIT_SEARCH_WORKERS` GB，见 `problems/concurrent-search-rss-blowup.md`），所以默认 `HERMIT_SEARCH_WORKERS=1`。历史上 arena 开启 + 无 dense unload 时，守护进程曾在数十小时索引累积下爬到 5–10GB（见 `problems/dense-embedder-arena-creep.md`），现已由上述三项设计收敛。

---

## 目录结构

```text
hermit/
├── main.py                    # 开发模式入口（uvicorn 直接运行）
├── pyproject.toml
├── hermit/
│   ├── app.py                 # FastAPI 应用 + lifespan（模型预加载、量化、collection 恢复、search executor）
│   ├── cli.py                 # CLI 入口（hermit start/stop/kb/search/collection/...）
│   ├── config.py              # 配置管理（HERMIT_HOME、模型名、idle-timeout、worker 数、ONNX arena/threads 等）
│   ├── models.py              # 模型下载与校验（huggingface_hub）
│   ├── ingestion/
│   │   ├── scanner.py         # 文件夹扫描 + 变更检测 + 索引
│   │   ├── watcher.py         # 后台轮询 watcher（_PollingWatcher，按 POLL_INTERVAL_SECONDS）
│   │   ├── chunker.py         # token 级文本切片 + markdown 语义切片
│   │   └── task_queue.py      # 后台索引任务队列（线程池，HERMIT_INDEX_WORKERS）
│   ├── retrieval/
│   │   ├── embedder.py        # Dense (TextEmbedding)，含 idle unload
│   │   ├── embed_cache.py     # diskcache 落地的 dense 嵌入缓存（sha256 keyed，TTL 30 天）
│   │   ├── searcher.py        # hybrid/semantic/keyword/fuzzy + filename filter + rerank
│   │   ├── reranker.py        # TextCrossEncoder，含 idle unload
│   │   └── fastembed_patch.py # 给 fastembed 暴露 ONNX arena/mem-pattern 注入点
│   ├── storage/
│   │   ├── lance.py           # LanceDB 表管理 + replace_file_chunks + 压缩/版本回收
│   │   ├── quantizer.py       # dense embedder / reranker 的 INT8 量化加载
│   │   ├── metadata.py        # SQLite 元数据管理
│   │   ├── registry.py        # 知识库注册表（~/.hermit/data/collections.json）
│   │   └── model_signature.py # 模型变更检测（~/.hermit/data/model_signature.json）
│   └── api/
│       ├── routes.py          # API 路由
│       └── schemas.py         # Pydantic 请求/响应模型
├── docs/
│   ├── design.md
│   ├── lancedb.md
│   ├── markdown-chunking.md
│   └── skill-distribution.md
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
