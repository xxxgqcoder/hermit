# Storage

> Qdrant 向量存储 + SQLite 元数据，支持 Local 和 Standalone 两种运行模式。

---

## Qdrant 运行模式

通过环境变量 `QDRANT_HOST` 切换：

| 模式 | 触发条件 | 说明 |
|---|---|---|
| **Local（嵌入式）** | `QDRANT_HOST` 未设置（默认） | `qdrant_client` 以路径模式直连本地目录，数据在 `~/.hermit/data/qdrant/` |
| **Standalone（独立服务）** | `QDRANT_HOST=localhost` | 连接外部 Qdrant HTTP/gRPC 服务；当 `QDRANT_MANAGED=true`（localhost 默认开启）时由 Hermit 自动管理 Docker 容器生命周期 |

详见 [qdrant_standalone.md](qdrant_standalone.md)。

---

## Standalone 模式：Docker 容器管理

**模块**：`hermit/storage/qdrant_docker.py`

### 持久化容器设计

容器生命周期独立于 hermit 进程：

| 操作 | 行为 |
|---|---|
| `hermit start` | 首次运行创建容器；后续 adopt 运行中的容器（快速路径）或 `docker start` 重启已停止的容器 |
| `hermit stop` | `docker stop`（非 `rm -f`），容器保留数据和端口配置 |
| 崩溃 / SIGKILL | 容器继续运行；下次 `hermit start` 通过健康检查 adopt |
| 强制重置 | `docker rm -f hermit_qdrant`，下次启动重新创建 |

### 安全配置

- `docker run` 传入 `--user uid:gid`，确保容器写出的文件属主为当前用户
- 通过 `-e QDRANT__STORAGE__SNAPSHOTS_PATH=/qdrant/storage/snapshots` 将快照目录重定向到已挂载的 volume 内（避免容器内 `/qdrant/snapshots` 权限拒绝）
- Volume 挂载：`~/.hermit/data/qdrant:/qdrant/storage:z`

---

## Local 模式：并发安全

- `qdrant_client` 本地模式内部使用 numpy 数组，非线程安全
- Hermit 用全局 `threading.Lock` 串行化所有写操作
- Standalone 模式绕过此锁（Qdrant 服务端自行处理并发）

---

## Collection Schema

每个知识库 = 1 个 Qdrant collection，使用 **named vectors**。

### 向量配置

| 向量名 | 维度 | 距离 | 说明 |
|---|---|---|---|
| `dense` | 768 | Cosine | jina-embeddings-v2-base-zh 语义向量 |
| `sparse` | 可变 | Dot | BM25 稀疏词权重向量 |

### Payload 结构

| 字段 | 类型 | 说明 |
|---|---|---|
| `title` | string | 文件名（不含扩展名） |
| `text` | string | 切片原文 |
| `source_file` | string | 源文件绝对路径 |
| `chunk_index` | int | 该切片在源文件中的位置（0-based） |
| `total_chunks` | int | 该源文件的总切片数 |

在 `source_file` 上建 **payload index**（KEYWORD 类型），确保按文件过滤/删除高效。

### 查询文件全部切片

通过 `source_file` payload filter 查询，配合 `chunk_index` 排序还原原文顺序。`total_chunks` 字段让调用方判断是否获取完整。

---

## 相关模块

| 模块 | 说明 |
|---|---|
| `hermit/storage/qdrant.py` | Qdrant 客户端（Local/Standalone 双模式）+ collection 管理 |
| `hermit/storage/qdrant_docker.py` | Docker 容器生命周期管理（Standalone 模式） |
| `hermit/storage/metadata.py` | SQLite 元数据读写 |
| `hermit/storage/registry.py` | 知识库注册表（`~/.hermit/data/collections.json`） |
| `hermit/storage/model_signature.py` | 模型变更检测（`~/.hermit/data/model_signature.json`） |
