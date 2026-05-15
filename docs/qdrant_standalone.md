# Qdrant 独立部署与本地模式 (Dual-Mode) 设计方案

## 1. 背景与目标

Hermit 目前默认使用 Qdrant 的本地模式（Local Mode），通过 Python `qdrant-client` 将向量数据和索引持久化到本地（如 `~/.hermit/data/qdrant`）。这种“开箱即用”的设计虽然实现了零依赖和 self-contained 目标，但在面对未来大规模数据和并发处理时存在明显瓶颈：
- **大量内存占用**：即便配置了 `on_disk=True`，加载大规模集合时 Python 进程仍会有不可忽视的内存开销。
- **并发性能瓶颈（GIL与锁）**：由于 Qdrant Local Mode（底层基于 Numpy）非线程安全，Hermit 在写入时强加了全局锁 `_lock`，导致多线程环境下无法发挥真实的吞吐能力，严重拖慢大型语料的索引构建速度。

为了应对更严苛的生产级规模，同时不损失轻量化属性，Hermit 计划实施 **Dual-Mode** 架构：
- **Local Mode（默认）**：适用于轻量级本机分析，无需配置环境直接运行。
- **Stand-alone Mode（外接模式）**：适用于高强度、大数据量场景。用户在一旁运行官方 Qdrant Docker 容器，突破并发局限。

本方案的核心亮点是：**通过挂载同一底层数据目录，在这两种模式间实现数据的无缝对接和自由切换**。

---

## 2. 核心架构设计

### 2.1 目录共享原则
Qdrant 官方提供的 Python 客户端（用于 Local Mode）与 Qdrant 服务器（Rust 核心引擎）的底层存储结构是 **100% 兼容** 的。因此，我们确立 **`~/.hermit/data/qdrant` 为唯一的标准数据目录**。

- **Local Mode**：Hermit Python 进程直接对该目录做 mmap 和 SQLite 存取。
- **Stand-alone Mode**：Hermit 发现配置（如环境变量 `QDRANT_HOST`）后，通过 HTTP/gRPC 与外部 Qdrant 通信。外部的 Qdrant 以 Docker 运行，并通过 Volumes（`-v`）直接挂载这个同一路径。

### 2.2 Docker 启动规范
为了确保数据在容器内外流转正常，启动 Stand-alone Qdrant 容器必须使用规范的形式。**通常无需手动执行这条命令**——当用户设置 `QDRANT_HOST=localhost`（默认就识别为本地）时，Hermit 会自动用等价参数 `docker run` 一个名为 `hermit_qdrant` 的容器（见 `hermit/storage/qdrant_docker.py`），并在进程退出时清理。下面这条命令仅作为参考，用于在没有 Hermit 管理或希望自行配置时使用：
```bash
docker run -d --name hermit_qdrant \
  --user $(id -u):$(id -g) \
  -p 6333:6333 -p 6334:6334 \
  -v ~/.hermit/data/qdrant:/qdrant/storage:z \
  qdrant/qdrant:v1.17.0
```
*(注：`:z` 用于 SELinux 环境，确保挂载点读写有效。镜像版本以 `hermit/config.py` 的 `QDRANT_IMAGE` 为准，需要时用 `QDRANT_IMAGE` 环境变量覆盖。)*

---

## 3. 防范核心隐患与应对机制

在“同一份数据，两套引擎共用”的方案中，需着重解决三大隐患：

### 3.1 文件权限污染 (Permission Denied)
如果用户不小心以 root 身份运行 Qdrant 容器，新增的向量数据文件会属于 `root`。此时若切回 Local 模式，Hermit 进程（普通用户）会报权限拒绝错误崩溃。
**应对策略**：
在文档和自动化启动脚本中强制/推荐加入 `--user $(id -u):$(id -g)` 参数，保证写出的文件属主为当前系统宿主用户。

### 3.2 多开导致的数据损坏 (File Locks & Corruption)
底层存储引擎对目录具有严格的排他锁机制，绝不可以让 Hermit 的 Local Mode 和启动的 Docker Qdrant 容器同时访问这个目录，一旦发生会直接死锁或损坏数据。
**应对策略**：三重防线防护
1. **端口探针 (Port Probing)**：在 Hermit 以 Local 模式启动时，检测本地的 `6333/6334` 端口是否被占用。如果连通，说明大概率跑着 Docker，主动拦截 Local 模式加载并警告。
2. **应用层文件锁 (App Lock)**：Hermit 本身可以在数据目录下放置一个自定义的 `.hermit.lock`，启动时进行锁定检测。
3. **引擎错误捕获 (Exception Catch)**：如果前两层漏掉，底层引擎本身会报锁定异常。在初始化 `QdrantClient(path=...)` 时加入 try-except，捕获特定的文件锁错误文本（如 *resource temporarily unavailable* 或 *lock*），拦截抛出的底层错误，吐出人类友好的中文提示：“Qdrant 数据目录已被占用，请检查是否已启动了 Stand-alone Docker 容器”。

### 3.3 数据格式版本锁定 (Version Mismatch)
新版本的 Docker 镜像可能涉及引擎的数据结构迁移。如果使用 `latest` 拉取了过新的版本进行了结构升级，之后退回到旧版本的 Python `qdrant-client` Local Mode 时，会无法解析数据。
**应对策略**：
- **强对齐版本要求**：Hermit 在 `hermit/config.py` 中将默认 Docker 镜像锁定为与 `pyproject.toml` 中 `qdrant-client` 兼容的 tag（当前 `qdrant/qdrant:v1.17.0`）。**避免使用 `latest` 标签**；若需自定义镜像版本，请通过 `QDRANT_IMAGE` 环境变量显式指定。

---

## 4. 实施状态

本设计在 2026-05 已完整落地：

- [x] **配置模块**：`hermit/config.py` 解析 `QDRANT_HOST`、`QDRANT_PORT`、`QDRANT_GRPC_PORT`、`QDRANT_MANAGED`、`QDRANT_CONTAINER_NAME`、`QDRANT_IMAGE` 等环境变量。
- [x] **客户端初始化**：`hermit/storage/qdrant.py` 的 `get_client()` 按 `QDRANT_HOST` 是否设置自动分流到 server (`host=/port=`) 或 local (`path=`)；并实现了端口探针 + 应用层 lock + 引擎错误捕获 三重防护。
- [x] **Docker 容器生命周期**：`hermit/storage/qdrant_docker.py` 在 stand-alone 模式自动 `docker run` 容器（`hermit_qdrant`），并通过 `atexit` 在进程退出时清理。
- [x] **API/CLI collection 管理**：`POST /collections` / `DELETE /collections/{name}` 已在 `hermit/api/routes.py` 实现；CLI 侧 `hermit kb add/remove/update/list` 已支持，对运行中的服务通过 HTTP 转发，未运行时本地直接操作注册表。
- [x] **健康检查暴露模式**：`GET /health` 返回 `qdrant_mode`（`"local"` / `"standalone"`）和 `qdrant_host`，参见 `routes.py` 中 `health()` 实现。
- [x] **README 同步**：README.md / README_cn.md 已补充 stand-alone 启动说明、CLI 完整命令清单、HTTP API 全量端点（含 `POST/DELETE /collections`）。

后续可选改进项（未实施，仅记录）：

- 并行入库解放：当前 stand-alone 模式下索引仍通过 `INDEX_WORKERS`（默认 1）控制；如果实测瓶颈是入库吞吐，可在 stand-alone 下放宽默认值（server 端无 `_lock` 限制，可天然支持高并发）。
- 容器健康监控：目前依赖 Hermit 进程退出时的 atexit 钩子清理容器；若 Hermit 异常崩溃，容器可能孤立。可以加 `docker --restart=unless-stopped` + Hermit 启动时检测既有容器并 reattach。