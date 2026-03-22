# Hermit Skill 分发方式说明

本文档梳理 `hermit` 仓库中 Skill 的**组织、打包、安装与使用**方式，便于统一维护与发布。

## 1. 分发目标

本仓库采用“**仓库内可直接用 + 全局可发现复用**”的双轨分发策略：

1. **项目级分发**：Skill 文件保存在仓库 `.agents/skills/`，在当前仓库上下文中可直接被 Agent 发现。
2. **全局分发**：通过 CLI 子命令 `hermit install-skills` 将 Skill 安装到 `~/.agents/skills/hermit/`，供其他工作区复用（按项目命名空间隔离）。

## 2. 目录与角色划分

### 2.1 源 Skill 目录（开发态）

- 路径：`.agents/skills/`
- 每个 Skill 一个子目录，核心文档为 `SKILL.md`
- `SKILL.md` 顶部使用 YAML frontmatter（`name`、`description`）提升 Agent 发现性

当前仓库中可见 Skill 目录：

- `.agents/skills/hermit-search/`

### 2.2 包内 Skill 目录（发布态）

- 目标路径：`hermit/_skills/`
- 来源：`pyproject.toml` 的 `[tool.hatch.build.targets.wheel.force-include]`
- 作用：确保 `SKILL.md` 被打进 wheel，安装后 CLI 可读取
- 注意：`hermit/_skills/` 是构建产物，已加入 `.gitignore`，不应提交到仓库

### 2.3 全局安装目录（运行态）

- 路径：`~/.agents/skills/{project}/`（如 `~/.agents/skills/hermit/`）
- 作用：跨仓库共享 Skill，按项目命名空间隔离，避免不同项目的同名 Skill 冲突
- 安装时会写入 `.origin` 文件（含包名 + 版本），用于来源追踪
- 卸载时若项目目录为空，自动清理

## 3. 分发链路（端到端）

### 3.1 开发阶段

1. 在 `.agents/skills/<skill-name>/SKILL.md` 维护 Skill 内容。
2. 保持 frontmatter 与实际能力一致。
3. 若新增 Skill，同步更新 `pyproject.toml` 的 `force-include`。

### 3.2 打包阶段

- `pyproject.toml` 使用 `hatchling` 打包。
- `project.scripts` 暴露 CLI 入口：`hermit = "hermit.cli:main"`。
- `force-include` 将 Skill 文档注入 wheel 内部目录 `hermit/_skills/`。

当前 `pyproject.toml` 配置：

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel.force-include]
".agents/skills/hermit-search" = "hermit/_skills/hermit-search"
```

### 3.3 安装与部署阶段

用户执行：

```sh
uv tool install hermit
hermit install-skills
```

CLI 逻辑（`hermit/cli.py` 中的 `_find_skills_dir` + `cmd_install_skills`）：

1. 优先读取包内目录 `hermit/_skills/`（通过 `__file__` 相对路径定位）。
2. 若不存在（开发模式），回退读取仓库 `.agents/skills/`（从 `hermit/` 向上一级定位项目根）。
3. 将每个含 `SKILL.md` 的 Skill 目录拷贝到 `~/.agents/skills/{project}/<skill-name>/`（如 `~/.agents/skills/hermit/hermit-search/`）。
4. 在目标目录写入 `.origin` 文件：`{"package": "hermit", "version": "0.1.0"}`。
5. 若传入 `--uninstall`，仅删除本包定义的 Skill 对应的全局目录。

## 4. 使用方式

### 4.1 项目内直接使用

在本仓库中，Agent 可直接读取 `.agents/skills/` 内文档执行对应流程。

### 4.2 全局复用

安装到 `~/.agents/skills/hermit/` 后，其他工作区中的 Agent 也可复用这些 Skill。

## 5. 维护建议（发布检查清单）

每次发布前建议检查：

1. `.agents/skills/` 下每个 Skill 均含 `SKILL.md`。
2. `SKILL.md` frontmatter 的 `name` 与目录名一致。
3. `pyproject.toml` 的 `force-include` 路径与真实目录一致。
4. `hermit install-skills` 可返回成功 JSON，并正确复制到 `~/.agents/skills/hermit/`。
5. `hermit install-skills --uninstall` 可正确清理安装内容。
6. `uv build` 后检查 wheel 内含 `hermit/_skills/<skill-name>/SKILL.md`。

## 6. 当前分发配置

当前分发配置与仓库目录已对齐：

- `.agents/skills/hermit-search/SKILL.md` → `hermit/_skills/hermit-search/SKILL.md`（wheel 内）→ `~/.agents/skills/hermit/hermit-search/SKILL.md`（全局安装）

---

如需扩展新 Skill，沿用同样流程：

1. 在 `.agents/skills/<new-skill>/SKILL.md` 创建 Skill（含 YAML frontmatter）。
2. 在 `pyproject.toml` 的 `force-include` 添加对应映射。
3. 运行 `hermit install-skills` 验证安装。
4. 运行 `uv build && unzip -l dist/*.whl | grep _skills` 验证打包。
