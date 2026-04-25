# Markdown 语义切片设计

## 背景

通用的 token 滑动窗口切片（`chunk_text`）对 Markdown 文件效果较差：

- 代码块、数学公式、表格会被从中间截断，产生语义残缺的切片
- 切片无法感知文档结构，heading 与 body 可能分散到不同 chunk
- 中英文混合密度差异大，固定 token 数对应的语义量差别显著

为此，对 `.md` 文件引入专用的**语义切片流程**：先将全文解析为语义块（semantic blocks），再用 heading-aware 滑动窗口将块组合为 chunk。

---

## 第一阶段：语义块解析（`parse_md_blocks`）

### 状态机解析器

`parse_md_blocks(text) -> list[str]` 使用线性状态机逐行扫描，将 Markdown 文本拆分为**语义完整的最小单元**（semantic block），每个 block 是一个独立的字符串。

空行作为分隔符跳过，不产生 block。

### 11 种 block 类型（优先级从高到低）

| 优先级 | 类型 | 识别规则 | 边界 |
|---|---|---|---|
| 1 | YAML frontmatter | 文件首行为 `---` | 至下一个 `---` |
| 2 | Fenced block | `` ``` `` 或 `~~~` 开头 | 至同字符同长度的关闭行 |
| 3 | Math block | 独立的 `$$` 行 | 至下一个独立 `$$` 行 |
| 4 | ATX heading | `#` 至 `######` 开头 | 单行 |
| 5 | Table | `\|` 开头的连续行 | 至第一个非 `\|` 开头行 |
| 6 | Blockquote | `>` 开头的连续行 | 至第一个非 `>` 开头行 |
| 7 | Horizontal rule | `---` / `***` / `___` 等 | 单行（优先于 List 检测） |
| 8 | List | 列表标记行开始 | 包含所有子项和嵌套项，整个列表为一个 block |
| 9 | Standalone image | 单行仅含 `![alt](url)` 或 `![[path]]` | 单行（支持 Obsidian wiki-link） |
| 10 | Setext heading | 文本行 + `===`/`---` 下划线行 | 两行 |
| 11 | Paragraph | 连续非空、非特殊行 | 至空行或特殊块开头 |

### 设计要点

**List 合并**：整个列表（包括多级嵌套子项）作为一个 block，避免将逻辑相关的列举项分散到不同 chunk。跨空行的列表延续（`  ` 缩进后续）也会被并入同一 block。

**支持 Unicode 项目符号**：`_LIST_RE` 覆盖 `•·–—▪▸◦` 等常见 Unicode 符号，确保从 PDF 转换的 Markdown 中使用非标准 bullet 的列表也能正确识别。

**Obsidian 兼容**：`_IMG_RE` 同时匹配标准 `![alt](url)` 和 Obsidian wiki-link `![[path.jpg]]` 语法。

---

## 第二阶段：Heading-Aware 滑动窗口（`chunk_markdown`）

### 基础滑动窗口

将 N 个语义块组合为一个 chunk，默认参数：

- `blocks_per_chunk=4`：每个 chunk 包含 4 个语义块
- `overlap=1`：相邻 chunk 间有 1 个块的重叠（机械模式下）

### 问题：纯机械滑动窗口的两个缺陷

**缺陷一：Heading 孤悬尾部（orphan heading）**

当 heading 恰好落在窗口的最后一个位置时，该 heading 没有跟随任何 body 内容，导致：

- 当前 chunk 末尾是一个没有内容的标题
- 下一个 chunk 从 body 开始，缺失所属 section 的标题上下文

```
机械切法示例（blocks_per_chunk=4）：
chunk[0] = [# Intro, P1, P2, ## Section 1]   ← ## Section 1 孤悬
chunk[1] = [P3, P4, P5, P6]                  ← 无标题上下文
```

**缺陷二：Overlap 语义价值低**

机械重叠 1 个段落，不保证重叠内容是有价值的语义锚点。两个 chunk 之间的共享内容可能只是证明过程、公式推导的中间片段，检索时无法提供有效的 section 上下文。

---

### 解决方案：两条 Heading-Aware 规则

#### Rule 1 — 不孤立 heading（orphan prevention）

```
检查条件：
  当前 chunk 的最后一个 block 是 heading（blocks[end-1]）
  且紧跟的下一个 block 不是 heading（blocks[end]）

动作：
  end += 1   （自动多包一个 body block）

保护条件：
  下一个 block 也是 heading 时不触发，避免全 heading 文档中无限延伸
```

修正后效果：

```
chunk[0] = [# Intro, P1, P2, ## Section 1, P3]   ← heading 带着第一个 body
chunk[1] 从 ## Section 1（或更后的 heading）开始
```

#### Rule 2 — Heading 锚定下一个 chunk 的起点

当前 chunk 结束后，决定下一个 chunk 从哪里开始：

**Rule 2(a)：Clean boundary — 下一个 block 本身是 heading**

```
if _is_heading(blocks[end]):
    start = end   # 直接从该 heading 开始，无重复
```

最理想的情况：chunk 恰好在两个 section 之间切开，每个 chunk 都以自己的 heading 开头，相邻 chunk 之间没有内容重叠。

**Rule 2(b)：Backward search — 下一个 block 是普通内容**

```
mechanical_start = end - overlap
next_start = mechanical_start

# 从机械位置向前回搜，找最近的 heading
for k in range(mechanical_start, start, -1):
    if _is_heading(blocks[k]):
        next_start = k
        break

start = max(next_start, start + 1)  # 保证不原地踏步
```

效果：下一个 chunk 从该 section 最近的 heading 开始，即使这个 heading 比机械 overlap 位置早几个 block。相邻 chunk 的重叠内容是"heading + heading 之后到当前 chunk 末尾的内容"，而不是随机的一个段落。

---

### Overlap 大小的上界

Rule 2(b) 的回搜范围是 `[mechanical_start, start)`，即当前 chunk 内部，**不会跨 chunk 回搜**。

$$\text{overlap}_{\max} = \text{overlap} + (\text{blocks\_per\_chunk} - \text{overlap} - 1) = \text{blocks\_per\_chunk} - 1$$

默认参数下（`blocks_per_chunk=4, overlap=1`），overlap 最大为 **3 个 blocks**，仅在一个 chunk 内密集出现多个 heading 时触发。长 section（无子标题）中找不到 heading，Rule 2(b) 退化为机械 overlap=1，不膨胀。

---

### 两条规则的目标

两条规则协同实现同一个目标：**让每个 chunk 在没有外部上下文的情况下，能够自己表达清楚"这是哪个 section 的内容"**，从而在向量检索时语义更完整、召回更精准。

```
Rule 1:  heading 不压尾  → heading 必然带着 body 出现在同一 chunk
Rule 2a: clean boundary → section 边界处无重复，每个 chunk 以自己的 heading 开头
Rule 2b: heading 锚点   → 无 clean boundary 时，回退到最近 heading 作为起点
```

---

## 与其他文件的集成

`hermit/ingestion/scanner.py` 的 `_index_file()` 根据文件扩展名分支：

```python
chunks = chunk_markdown(text) if file_path.suffix.lower() == '.md' else chunk_text(text)
```

其他文件类型继续使用原有的 token 滑动窗口切片（`chunk_text`），行为不变。

---

## 已知局限

**长 section 的后续 chunk 缺失 heading 上下文**

若某个 section 包含大量段落，第一个 chunk 包含该 section 的 heading，但后续 chunk 只有段落，没有重复的 heading。检索时命中后续 chunk 的向量，模型不知道这些段落属于哪个 section。

可行的改进方向（未实现）：**Contextual prefix** — 给每个 chunk 前置"最近一次见到的 heading"，作为 chunk 的隐式标题。这与当前的 overlap 策略正交，可独立实现。

**PDF 转换噪音**

从 PDF 提取的 Markdown 常含页眉/页脚行（如 `Do not distribute without permission...`），这些行会被解析为独立的 paragraph block，混入 chunk 内容。需要在 ingestion 前增加预处理过滤步骤。
