# Ingestion 写入路径设计

负责把知识库文件夹的内容增量地解析、切片、向量化并写入存储。涉及模块：
`hermit/ingestion/scanner.py`、`watcher.py`、`chunker.py`、`task_queue.py`、
`hermit/storage/metadata.py`、`hermit/retrieval/embed_cache.py`。

## 数据源

- 指定文件夹作为知识库唯一来源，递归扫描（`rglob("*")`）。
- **扩展名 allowlist**：仅处理 `scanner._TEXT_EXTENSIONS` 内的文本类扩展名（笔记 `.md/.markdown/.txt/.rst/.org`、结构化文本 `.json/.jsonl/.yaml/.yml/.toml/.csv/.tsv/.xml`、网页 `.html/.htm`、各类源码、配置/日志 `.ini/.cfg/.conf/.env/.log` 等）。不在表内的扩展名直接跳过。
- 文件按 UTF-8 读取（`errors=replace`）。
- 跳过隐藏文件（路径中含 `.` 开头的部分）。
- 可按 collection 配置 `ignore_patterns` / `ignore_extensions` 进一步排除。
- 多模态文件（PDF、图片等）→ 文本的解析由外部流程负责，不在本服务范围内。

## 触发方式

- **启动时**：全量扫描，逐文件对比 hash 与 SQLite 记录，增量更新索引。
- **运行时轮询**：后台线程定期重跑 `scan_folder`，默认每 **900s（15 分钟）**（`HERMIT_POLL_INTERVAL`）。当前实现为纯轮询（`watcher.py` 的 `_PollingWatcher`，daemon 线程），**不使用 OS 文件事件 / watchdog**。首次轮询跳过（启动已扫过一遍）。

## 变更检测（SQLite 元数据库）

每个知识库一个 SQLite 库，路径 `~/.hermit/data/metadata/{collection}.db`，由 `MetadataStore` 管理（按 collection 名做进程内单例，连接 thread-local）。

### 表 Schema

```sql
CREATE TABLE IF NOT EXISTS files (
    file_path       TEXT PRIMARY KEY,   -- 文件绝对路径
    file_hash       TEXT NOT NULL,      -- 文件内容 SHA256
    file_mtime      REAL NOT NULL,      -- 修改时间（记录用，不参与判定）
    chunk_count     INTEGER NOT NULL,   -- 该文件切片数
    last_indexed_at REAL NOT NULL       -- 上次索引时间（time.time()）
);
```

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `file_path` | TEXT | PRIMARY KEY | 文件绝对路径 |
| `file_hash` | TEXT | NOT NULL | 文件内容 SHA256 |
| `file_mtime` | REAL | NOT NULL | 修改时间；保留为记录字段，**不参与**变更判定 |
| `chunk_count` | INTEGER | NOT NULL | 该文件切片数 |
| `last_indexed_at` | REAL | NOT NULL | 上次索引 Unix 时间戳 |

### 行为

- **变更判定仅对比 `file_hash`（SHA256）**；`file_mtime` 仅作记录。
- `upsert` 用 `INSERT ... ON CONFLICT(file_path) DO UPDATE` 幂等写入。
- `get_all_records()` 单次全表扫描返回 `{file_path: (hash, mtime)}`，供扫描时做差集。
- **自愈**：若底层表被外部删除（`no such table: files`），自动重置连接并重建表后重试。
- `destroy()` 关闭连接并删库文件（collection 删除时调用）。

**选用 SQLite 的理由**：启动时对每个文件查 LanceDB 效率低；SQLite 单次全表扫描即可完成对比，天然适合关系型元数据。

## 文本切片

按文件类型分派（`scanner.py`）：

```python
chunks = chunk_markdown(text) if suffix == '.md' else chunk_text(text)
```

- **非 Markdown（`chunk_text`）**：用 embedding 模型自带 tokenizer 按 token 数切片 + 滑动窗口重叠，默认 `chunk_tokens=256`、`overlap_tokens=32`；用模型 tokenizer 计数消除中英文字符密度差异；短文本（≤ chunk_tokens）不切，空文本跳过。
- **Markdown（`chunk_markdown`）**：先正则解析成语义 block（标题/列表/代码围栏/数学块/表格/引用/段落等），再按 heading-aware 滑窗成块；代码围栏、表格、公式作为**不可分割整体**参与分块。完整设计见 [`markdown-chunking.md`](./markdown-chunking.md)。

## 向量化增强（标题前缀）

Embedding 前把文件名作为标题拼接到切片前：

```python
title = file_path.stem            # 去扩展名的 basename
embed_inputs = [f"[{title}]\n{chunk}" for chunk in chunks]
```

增强语义召回。注意：
- `title` 保留**原始大小写**，用于 embedding 前缀；写入 LanceDB 的 `title` 列亦同。
- `filename = title.lower()`（小写化的 stem，去扩展名）写入 LanceDB `filename` 列，供子串/glob 过滤。例：`/path/to/My Notes.md` → `title="My Notes"`，`filename="my notes"`。

## 索引流程

1. 递归扫描文件夹，逐文件对比 SHA256 与 SQLite 记录，得出新增/修改/删除集。
2. **新增/修改**：切片 → dense 向量化（先查 [embed cache](#嵌入缓存)，命中则跳过推理）→ 生成 UUID 行 id + payload → 调 `replace_file_chunks`（按 `source_file` 删旧 chunks 再插新）→ 更新 SQLite。
3. `replace_file_chunks` 完成后 `tbl.optimize(cleanup_older_than=1min)`，让新行立即进入 FTS 索引，同时回收瞬时旧版本（见 [`lancedb.md`](./lancedb.md)）。
4. **删除**：按 `source_file` 删 chunks + 删 SQLite 记录。

## 并发：索引任务队列

索引任务经 `task_queue.py` 的后台线程池执行：

- 池大小 `HERMIT_INDEX_WORKERS`，**默认 1**——个人场景增量更新不频繁，单 worker 避免与搜索争用共享 ONNX session。
- 初次批量索引可调高（`HERMIT_INDEX_WORKERS=2+`）。
- 轮询扫描以 `defer_indexing=True` 把任务投递到队列，不在扫描线程内同步索引。

## 嵌入缓存

dense 向量经 `embed_cache.py`（`diskcache` 落地，存于 `~/.hermit/cache/dense/`）缓存：

- **key** = `sha256("{model}::{text}")`，`text` 即拼接标题后的实际模型输入；按模型名命名空间，换模型不污染。
- 命中时做维度/长度校验（自愈，坏值视为未命中）。
- **TTL = 30 天**（`EMBED_CACHE_TTL_SECONDS`）。
- 无环境开关，按设计常开（自愈 + TTL 保证异常状态自动清理）。
