# 当前仓库 Skill 分发方式说明

本文档梳理 `paper_reading` 仓库中 Skill 的**组织、打包、安装与使用**方式，便于团队统一维护与发布。

## 1. 分发目标

本仓库采用“**仓库内可直接用 + 全局可发现复用**”的双轨分发策略：

1. **项目级分发**：Skill 文件保存在仓库 `.agents/skills/`，在当前仓库上下文中可直接被 Agent 发现。
2. **全局分发**：通过 CLI 子命令 `paper-reading install-skills` 将 Skill 安装到 `~/.agents/skills/`，供其他工作区复用。

## 2. 目录与角色划分

### 2.1 源 Skill 目录（开发态）

- 路径：`.agents/skills/`
- 每个 Skill 一个子目录，核心文档为 `SKILL.md`
- `SKILL.md` 顶部使用 YAML frontmatter（`name`、`description`）提升 Agent 发现性

当前仓库中可见 Skill 目录：

- `.agents/skills/digest-to-md/`
- `.agents/skills/extract-pages/`

### 2.2 包内 Skill 目录（发布态）

- 目标路径：`paper_reading/_skills/`
- 来源：`pyproject.toml` 的 `[tool.hatch.build.targets.wheel.force-include]`
- 作用：确保 `SKILL.md` 被打进 wheel，安装后 CLI 可读取

### 2.3 全局安装目录（运行态）

- 路径：`~/.agents/skills/`
- 作用：跨仓库共享 Skill，提升 Agent 自动发现率

## 3. 分发链路（端到端）

### 3.1 开发阶段

1. 在 `.agents/skills/<skill-name>/SKILL.md` 维护 Skill 内容。
2. 保持 frontmatter 与实际能力一致。
3. 若新增 Skill，同步更新 `pyproject.toml` 的 `force-include`。

### 3.2 打包阶段

- `pyproject.toml` 使用 `hatchling` 打包。
- `project.scripts` 暴露 CLI 入口：`paper-reading = "paper_reading.cli:main"`。
- `force-include` 将 Skill 文档注入 wheel 内部目录 `paper_reading/_skills/`。

### 3.3 安装与部署阶段

用户执行：

- `uv tool install paper-reading`
- `paper-reading install-skills`

CLI 逻辑（`paper_reading/cli.py`）：

1. 优先读取包内目录 `paper_reading/_skills/`。
2. 若不存在（开发模式），回退读取仓库 `.agents/skills/`。
3. 将每个 Skill 目录拷贝到 `~/.agents/skills/<skill-name>/`。
4. 若传入 `--uninstall`，删除对应全局目录。

## 4. 使用方式

### 4.1 项目内直接使用

在本仓库中，Agent 可直接读取 `.agents/skills/` 内文档执行对应流程。

### 4.2 全局复用

安装到 `~/.agents/skills/` 后，其他工作区中的 Agent 也可复用这些 Skill。

## 5. 维护建议（发布检查清单）

每次发布前建议检查：

1. `.agents/skills/` 下每个 Skill 均含 `SKILL.md`。
2. `SKILL.md` frontmatter 的 `name` 与目录名一致。
3. `pyproject.toml` 的 `force-include` 路径与真实目录一致。
4. `paper-reading install-skills` 可返回成功 JSON，并正确复制到 `~/.agents/skills/`。
5. `paper-reading install-skills --uninstall` 可正确清理安装内容。

## 6. 差异处理结果（已完成）

已完成以下一致性修复：

1. 将 `pyproject.toml` 中 `force-include` 从 `process-pdf` 修正为 `digest-to-md`。
2. 将 `README.md` 中 Skill 列表与文档链接同步为 `digest-to-md`。

当前分发配置与仓库目录已对齐：

- `.agents/skills/digest-to-md/SKILL.md` -> `paper_reading/_skills/digest-to-md/SKILL.md`
- `.agents/skills/extract-pages/SKILL.md` -> `paper_reading/_skills/extract-pages/SKILL.md`

---

如需扩展新 Skill，建议沿用本文档同样流程：`仓库目录定义 -> wheel 打包注入 -> install-skills 全局部署 -> 跨仓库复用验证`。
