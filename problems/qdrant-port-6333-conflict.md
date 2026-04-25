# 问题：端口 6333 冲突导致服务启动后 collection 全部丢失

## 现象

`hermit start` 成功返回，服务响应正常，但 `/health` 显示 `collections: []`，
所有已注册的 collection 均无法访问，`/collections/{name}/status` 返回 404。

## 根本原因

Hermit 在 Local 模式下使用 Qdrant embedded client（数据存储在 `~/.hermit/data/qdrant/`），
该 client 在初始化时会检查 6333 端口是否已被占用。

若之前以 Standalone 模式（`QDRANT_HOST=localhost`）运行过 hermit，停止服务时 Qdrant Docker
容器或独立进程未能随 hermit 一起终止，端口 6333 残留占用。新启动的 Local 模式 hermit
在 `lifespan` 阶段调用 `scan_folder` → `qdrant.ensure_collection` 时抛出：

```
RuntimeError: 端口 6333 已被占用，检测到可能存在运行中的 Qdrant 服务。
请设置 QDRANT_HOST 环境变量以使用 Stand-alone 模式，或停止该服务后重试。
```

错误被 `app.py` 的 `except Exception` 捕获并记录为 WARNING，服务继续启动（`_server_ready = True`），
但 `_collections` dict 为空，所有 collection 操作均返回 404。

## 触发路径

```
hermit stop  # 停止 hermit，但 Qdrant 进程未随之终止
hermit start # 新进程启动 Local 模式，端口 6333 被旧 Qdrant 占用
             # → lifespan 中 restore collection 全部失败
             # → collections: [] 但服务看似正常
```

## 诊断方法

1. 查看日志，搜索 "Failed to restore collection"：
   ```sh
   grep "Failed to restore" ~/.hermit/logs/hermit.log
   ```
2. 检查端口占用：
   ```sh
   lsof -ti :6333
   ```

## 解决方法

```sh
# 1. 杀掉占用 6333 端口的残留进程
lsof -ti :6333 | xargs kill -9

# 2. 重启 hermit
hermit stop
hermit start
```

## 预防措施

- 切换存储模式前，确保旧模式的 Qdrant 进程已完全退出
- Standalone 模式下 `hermit stop` 会尝试停止 Docker 容器，但若 Docker 容器异常退出，
  主机端口可能仍被占用，需手动清理
