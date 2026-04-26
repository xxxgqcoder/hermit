# Ingestion Pipeline

> 写入路径：文件夹扫描 → 变更检测 → 文本切片 → 向量化 → 入库

---

## 数据源

- 指定文件夹作为知识库唯一来源
- 采用**白名单制**：仅索引扩展名在 `_TEXT_EXTENSIONS` 集合内的文件，其余文件（含 PDF、图片、Office 文档等二进制文件）直接跳过并打印 `WARNING: Skipping non-text file` 日志，不会被读取或入库
- 支持的文本扩展名包括：`.md`、`.txt`、`.rst`、`.json`、`.jsonl`、`.yaml`、`.toml`、`.csv`、`.html`、`.py`、`.js`、`.ts`、`.go`、`.sh` 等约 40 种
- 跳过隐藏文件（路径中含 `.` 开头的部分）
- 跳过符号链接（symlink）

---

## 触发方式

| 时机 | 方式 | 说明 |
|---|---|---|
| **启动时** | 全量扫描 | 逐文件对比 hash 与 SQLite 记录，增量更新索引 |
| **运行时** | 定期轮询（默认 15 分钟） | 检测文件变更并触发增量更新 |
| **文件事件** | watchdog 实时监听 | 防抖 2s 后触发全量 rescan |

---

## 变更检测（SQLite 元数据库）

每个知识库维护一个 SQLite 元数据库（`~/.hermit/data/metadata/{collection}.db`），记录已索引文件状态：

| 字段 | 类型 | 说明 |
|---|---|---|
| `file_path` (PK) | TEXT | 文件绝对路径 |
| `file_hash` | TEXT | 文件内容 SHA256 |
| `file_mtime` | REAL | 修改时间 |
| `chunk_count` | INTEGER | 该文件切片数 |
| `last_indexed_at` | REAL | 上次索引时间 |

变更检测仅对比 `file_hash`（SHA256），`file_mtime` 作为记录字段保留但不参与判定。

**选用 SQLite 的理由**：启动时对每个文件查 Qdrant payload 效率低；SQLite 单次全表扫描即可完成对比，天然适合存储关系型元数据。

---

## 文本切片

使用 embedding 模型自带的 tokenizer 按 token 数切片 + 滑动窗口重叠：

- 默认 `chunk_tokens=256` tokens，`overlap_tokens=32` tokens
- 使用模型 tokenizer 计数，消除中英文字符密度差异
- 短文本（≤ chunk_tokens）不做切分
- 空文本跳过
- **向量化增强**：Embedding 时将文件名作为标题拼接到切片内容前，格式为 `[{title}]\n{chunk}`，以增强语义召回

详见 [markdown-chunking.md](markdown-chunking.md)。

---

## 索引流程

1. 递归扫描文件夹（`rglob("*")`），白名单过滤后逐文件对比 SHA256 与 SQLite 记录
2. **新增/修改文件**：切片 → 向量化 → 按 `source_file` 删除旧 chunks → **批量插入**新 chunks（每批 ≤100 个 point，UUID 作为 point ID）→ 更新 SQLite
3. **删除文件**：删除对应 Qdrant chunks → 删除 SQLite 记录
4. 后台线程定期轮询执行扫描逻辑（Scanning Watcher）
5. watchdog 监听到文件事件时执行全量 rescan 逻辑（防抖后）

> **批量写入**：单次 upsert 拆分为 ≤100 点的批次，避免触及 Qdrant 32MB Payload 上限。

---

## 相关模块

| 模块 | 说明 |
|---|---|
| `hermit/ingestion/scanner.py` | 文件夹扫描 + 白名单过滤 + 变更检测 + 索引 |
| `hermit/ingestion/watcher.py` | watchdog 实时监听（2s 防抖） |
| `hermit/ingestion/chunker.py` | token 级文本切片 + overlap |
| `hermit/ingestion/task_queue.py` | 后台索引任务队列（线程池） |
| `hermit/storage/metadata.py` | SQLite 元数据读写 |
