# Hermit Agent 开发指引

## 环境与文档边界

- 使用 `uv` 和 `pyproject.toml` 管理 Python 3.12～3.13 的依赖与运行环境。
- 首次准备开发环境时运行 `uv sync --group dev`，项目命令和测试通过 `uv run ...` 执行，不在 README 中维护另一套 Agent 开发环境。
- `README.md` 是默认中文用户文档，`README_en.md` 是英文版；面向 Agent 的开发与执行约束只写在本文件，Skill 打包原理写在 `docs/skill-distribution.md`。
- 修改用户可见行为时同步更新中英文 README；修改 Skill 行为、安装方式或 CLI 参数时同步检查 `.agents/skills/hermit-search/SKILL.md`。

## CLI 与 Skill 安装

仅当用户明确要求安装、更新或修复 CLI/Skill 时执行本节；在本仓库中必须安装当前 checkout，不要改装 GitHub 或包索引中的其他版本。

```sh
hermit_repo_root="$(git rev-parse --show-toplevel)"
cd "${hermit_repo_root}"
test -f pyproject.toml
rg '^name = "hermit"$' pyproject.toml
command -v uv
uv tool install . --force
```

安装后先验证 CLI：

```sh
uv tool list
command -v hermit
hermit --help
```

确认 `uv tool list` 中的 Hermit 版本与 `pyproject.toml` 一致，并确认 `hermit --help` 包含 `install-skills`；不一致时不能继续安装旧 Skill。

仅当 `command -v hermit` 失败时，更新后续 shell 并为当前进程补齐 uv 工具目录，再重复验证：

```sh
uv tool update-shell
hermit_bin_dir="$(uv tool dir --bin)"
export PATH="${hermit_bin_dir}:${PATH}"
command -v hermit
hermit --help
```

当用户要求完成 Skill 安装时，不能以 repo 内已经自动发现 Skill 为由跳过全局部署：

```sh
hermit install-skills
test -f ~/.agents/skills/hermit-search/SKILL.md
cmp -s \
  .agents/skills/hermit-search/SKILL.md \
  ~/.agents/skills/hermit-search/SKILL.md
```

确认 `hermit install-skills` 返回的 JSON 包含 `"status": "installed"` 和 `"hermit-search"`，且 `cmp` 返回 0；若当前会话未刷新 Skill 列表，告知用户新开任务或重启 Agent，不要反复安装。

CLI/Skill 安装不包含约 1.3 GB 模型下载和服务启动；只有用户准备实际检索或明确要求时才执行 `hermit download`、`hermit start`。

## 开发与验证

- 优先先跑与改动直接相关的测试，再按风险决定是否运行 `uv run pytest` 全量测试。
- `test_real_dense_cold_reload_latency` 和 `test_real_reranker_cold_reload_latency` 依赖真实模型缓存或网络；离线基础回归使用 `uv run pytest -k 'not real_dense_cold_reload_latency and not real_reranker_cold_reload_latency'`，并在交付时明确报告未验证的真实模型测试。
- 文档和代码提交前运行 `git diff --check`。
- 修改 Skill 或打包配置时构建 wheel，并确认其中包含 `hermit/_skills/hermit-search/SKILL.md`。
- 创建 PR 前检查 `.agents/skills/`、`README.md`、`README_en.md` 和 `docs/skill-distribution.md` 是否仍与实现一致。
- 保留用户已有且与任务无关的工作树改动；只暂存和提交本次任务范围内的文件。
