# API & Service Layer

> FastAPI 服务：HTTP 端点、CLI、持久化恢复。

---

## HTTP API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/search` | 混合召回 + 精排检索 |
| `POST` | `/collections` | 注册新知识库（扫描 + 开始监听） |
| `DELETE` | `/collections/{name}` | 删除知识库（Qdrant collection + SQLite + 注册表 + watcher） |
| `POST` | `/collections/{name}/sync` | 手动触发增量同步 |
| `GET` | `/collections/{name}/status` | 查询知识库状态（文件数、chunks 数） |
| `GET` | `/collections/{name}/tasks` | 查询待处理索引任务数量 |
| `GET` | `/health` | 服务健康检查 |

---

## CLI

`hermit` CLI 与 daemon 协同工作：

- daemon 运行时，`hermit kb add/remove` 通过 HTTP 端点操作
- daemon 未运行时，CLI 直接操作本地存储

| 命令 | 说明 |
|---|---|
| `hermit start` | 启动 daemon（含 Qdrant 容器管理） |
| `hermit stop` | 停止 daemon |
| `hermit status` | 查看 daemon 状态、各 KB 统计 |
| `hermit kb add <path>` | 注册知识库 |
| `hermit kb remove <name>` | 删除知识库 |
| `hermit search <query>` | 命令行检索 |
| `hermit download` | 提前下载模型 |
| `hermit install-skills` | 安装 hermit-search Agent skill |

---

## 持久化与启动恢复

| 机制 | 说明 |
|---|---|
| **Collection 注册表** | `~/.hermit/data/collections.json` 记录所有已注册知识库配置（`folder_path`、`ignore_patterns`、`ignore_extensions`）；服务启动时自动加载并恢复，无需重新注册 |
| **Watchdog 自动恢复** | 服务启动时为每个已注册 collection 自动重启文件监听 |
| **模型变更检测** | 模型发生变更时，启动时自动触发所有 collection 全量重建索引（详见 [models.md](models.md)） |

---

## 服务架构

- 单进程，FastAPI + uvicorn
- `app.py` 使用 `asynccontextmanager` lifespan 管理启动/关闭（模型预加载、collection 恢复）
- 搜索请求通过 `run_in_executor` 在独立线程池（`SEARCH_THREADS`）中执行，避免阻塞事件循环

---

## 相关模块

| 模块 | 说明 |
|---|---|
| `hermit/app.py` | FastAPI 应用 + lifespan |
| `hermit/cli.py` | CLI 入口 |
| `hermit/api/routes.py` | API 路由实现 |
| `hermit/api/schemas.py` | Pydantic 请求/响应模型 |
| `hermit/storage/registry.py` | 知识库注册表持久化 |
