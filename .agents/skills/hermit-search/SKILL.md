------














































































































错误输出：`{"error": "message"}`，退出码非零。```hermit search leetcode "two sum" --pretty```sh所有 CLI 命令输出 JSON。使用 `--pretty` 获取缩进格式：## 输出格式| `GET` | `/health` | 健康检查 || `GET` | `/collections/{name}/tasks` | 任务队列状态 || `POST` | `/collections/{name}/sync` | 触发同步 || `GET` | `/collections/{name}/status` | 集合状态 || `POST` | `/search` | 语义搜索 ||---|---|---|| 方法 | 路径 | 说明 |服务启动后暴露 REST API（默认 `http://127.0.0.1:8000`）：### 5. API 端点| `hermit logs` | 查看服务日志 || `hermit stop` | 停止服务 || `hermit status` | 查看服务状态 || `hermit collection tasks <name>` | 查看索引任务队列 || `hermit collection sync <name>` | 触发手动同步 || `hermit collection status <name>` | 查看集合索引状态 || `hermit kb remove <name>` | 移除知识库 || `hermit kb list` | 列出所有知识库集合 ||---|---|| 命令 | 说明 |### 4. 其他命令返回 JSON 包含匹配的文本片段、来源文件路径和相关性分数。```hermit search leetcode "dynamic programming sliding window"```sh示例：- `--rerank-candidates`：精排候选池大小（默认 30）- `--top-k`：返回结果数（默认 5）```hermit search <collection-name> "<query>" [--top-k N] [--rerank-candidates N]```sh### 3. 搜索```hermit kb add leetcode ~/projects/hermit/documents```sh示例：- `folder-path`：文件夹绝对路径，文件夹内的文本文件将被自动索引- `collection-name`：字母或数字开头，仅含字母、数字、下划线、连字符，最长 64 字符```hermit kb add <collection-name> <folder-path>```sh### 2. 添加知识库返回 JSON：`{"status": "started", "pid": <pid>, "port": 8000}````hermit start```sh### 1. 启动服务## 使用方式```hermit download```sh首次使用前需下载模型（约 1.5 GB）：### 模型下载```hermit install-skills# 部署 Skill 到全局目录（可选，使 Agent 自动发现）uv tool install hermit```sh### 安装## 前置条件- **服务生命周期**：后台启动/停止/状态查询- **自动索引**：文件变更自动检测并增量索引- **知识库管理**：注册/注销文件夹为知识库集合- **语义搜索**：基于 dense + sparse 双路召回 + cross-encoder 精排## 功能通过 Hermit 本地语义搜索服务，对文件夹知识库进行混合检索（Dense + Sparse）并精排。# Skill: hermit-search---description: 'Local semantic search over knowledge base folders using Hermit service. Use when: searching knowledge base, semantic search, querying documents, finding relevant text, managing knowledge base collections.'name: hermit-searchname: hermit-search
description: 'Local semantic search over knowledge base collections powered by Hermit. Use when: searching knowledge base, semantic search, querying documents, managing collections, adding knowledge base, indexing files.'
---

# Skill: hermit-search

Hermit 本地语义搜索服务的使用指南。Hermit 提供基于向量的混合检索（Dense + Sparse）和 Cross-Encoder 精排，支持多知识库管理。

## 前置条件

### 安装

```sh
uv tool install hermit
# 部署 Skill 到全局目录（可选，使 Agent 自动发现）
hermit install-skills
```

### 模型下载

首次使用前需下载模型（约 1GB）：

```sh
hermit download
```

## 使用流程

### 1. 启动服务

```sh
hermit start
```

输出示例：`{"status": "started", "pid": 12345, "port": 8000}`

### 2. 添加知识库

将一个文件夹注册为知识库 collection：

```sh
hermit kb add <name> <directory>
```

- `name`：collection 别名（字母数字 + 下划线/连字符，以字母或数字开头）
- `directory`：文件夹路径，Hermit 会递归扫描并索引其中的文本文件

示例：

```sh
hermit kb add my-notes ~/Documents/notes
```

### 3. 语义搜索

```sh
hermit search <collection> "<query>"
```

参数：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `collection` | str | — | Collection 名称 |
| `query` | str | — | 搜索查询 |
| `--top-k` | int | 5 | 返回结果数 |
| `--rerank-candidates` | int | 30 | 精排候选池大小 |

示例：

```sh
hermit search my-notes "如何实现二分查找"
```

### 4. 其他管理命令

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

### 5. 服务生命周期

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

## 技术细节

- **Embedding 模型**：jinaai/jina-embeddings-v2-base-zh（768 维）
- **Sparse 模型**：Qdrant/bm25
- **Reranker**：jinaai/jina-reranker-v2-base-multilingual
- **向量数据库**：Qdrant（嵌入式模式）
- **推理后端**：fastembed（ONNX Runtime，纯 CPU）
- **数据目录**：`~/.hermit/`（可通过 `HERMIT_HOME` 环境变量覆盖）
