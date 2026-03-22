这份文档旨在为你的 **M4 Pro (64GB RAM)** 定制一套完全独立、零外部依赖（Self-contained）的高性能语义检索 Skill。该方案集成了 **BGE-M3** 混合召回与 **Gemma-v2-Reranker** 深度精排。

---

## 🛠️ 项目架构设计 (Architecture)

为了实现“一键安装”和“完全本地化”，项目采用以下分层结构：



1.  **数据接入层**: 支持 Markdown/PDF 自动切片（Chunking）。
2.  **检索层 (Recall)**: 
    * **Dense**: 1024 维向量（支持语义）。
    * **Sparse**: 词权重向量（支持关键词/专有名词）。
3.  **精排层 (Rerank)**: 基于 Gemma-2B 的 Cross-Encoder，处理 Top 100 候选集。
4.  **服务层**: FastAPI 暴露标准 OpenAPI 接口。

---

## 📂 目录结构规范

```text
/Local-Search-Skill
├── bootstrap.sh            # 一键安装与环境检查脚本
├── main.py                 # FastAPI 服务入口
├── core/
│   ├── retriever.py        # BGE-M3 & Qdrant 逻辑
│   └── reranker.py         # Gemma Reranker (MLX/ONNX) 逻辑
├── models/                 # 核心模型存放地 (Git LFS 或脚本下载)
│   ├── bge-m3/
│   └── gemma-2b-reranker/
├── data/                   # Qdrant 本地数据库文件
└── requirements.txt        # 核心依赖 (fastembed, mlx, qdrant-client)
```

---

## 🚀 核心组件实现

### 1. 混合召回 (BGE-M3 + Qdrant)
利用 `fastembed` 在 Mac MPS 上加速推理，并开启 **Binary Quantization** 提升检索吞吐量，同时通过 `rescore=True` 在内存中利用原始向量补偿精度。

```python
# core/retriever.py
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

class retriever:
    def __init__(self):
        # 强制模型使用本地路径，确保 self-contained
        self.model = TextEmbedding(model_name="BAAI/bge-m3", cache_dir="./models")
        self.client = QdrantClient(path="./data/qdrant")

    def search(self, query: str, limit=100):
        query_vector = list(self.model.embed([query]))[0]
        return self.client.query_points(
            collection_name="docs",
            query=query_vector,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(rescore=True)
            ),
            limit=limit
        ).points
```

### 2. 深度精排 (Gemma Reranker)
利用 **MLX** 框架压榨 M4 Pro 的统一内存性能。Gemma-2B 作为 Reranker 能理解复杂的语境逻辑。

```python
# core/reranker.py
import mlx.core as mx
from mlx_lm import load, generate

class Reranker:
    def __init__(self):
        # 加载针对 MLX 优化的 Gemma-2b-it 重排权重
        self.model, self.tokenizer = load("./models/gemma-2b-reranker")

    def score(self, query: str, passages: list):
        # 构造 Cross-Encoder 输入: [CLS] Query [SEP] Passage [SEP]
        # 返回相关性得分 (Logits)
        scores = []
        for p in passages:
            input_str = f"Query: {query} \nDocument: {p} \nRelevant: "
            # 执行单 token 推理获取 "Yes/No" 的概率分布
            score = self.get_logit(input_str) 
            scores.append(score)
        return scores
```

---

## 📦 一键安装脚本 (`bootstrap.sh`)

该脚本负责从零构建环境，确保用户只需运行一次即可使用。

```bash
#!/bin/bash
set -e

echo "🔍 Checking Hardware: Apple Silicon detected..."
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装极速推理后端
pip install uv
uv pip install fastembed qdrant-client fastapi uvicorn mlx-lm

# 预下载模型到项目目录 (关键：实现 Self-contained)
echo "📥 Downloading Models (BGE-M3 & Gemma-Reranker)..."
export HF_HUB_CACHE="./models"
python3 -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-m3')"

echo "✅ Setup Complete. Run 'source .venv/bin/activate && python main.py' to start."
```

---

## ⚡ M4 Pro 性能调优参数

| 参数 | 推荐设置 | 理由 |
| :--- | :--- | :--- |
| **Batch Size** | 16 - 32 | 充分利用 M4 Pro 的 273GB/s 内存带宽 |
| **Qdrant Threading** | 12 Threads | 匹配 M4 Pro 核心数，加速 HNSW 索引构建 |
| **Quantization** | Binary (BQ) | 极大减少内存占用，配合 100 召回位次补齐精度 |
| **MLX Cache** | 4-bit / 8-bit | 若内存压力大可量化 Reranker，但 64G 建议直跑 FP16 |

---

## 📝 使用说明 (API Usage)

Skill 启动后，通过 `POST /search` 访问：

* **输入**: `{"query": "如何在 Mac 上配置动态环境变量？", "top_k": 5}`
* **过程**: 
    1.  BGE-M3 毫秒级捞回 100 个片段。
    2.  Gemma-2B 对 100 个片段进行语义打分。
* **输出**: 经过重排的最相关 5 条知识点，总延时预计 **600ms - 800ms**。

---

> **💡 提示**: 建议将 `./models` 目录加入 `.gitignore`，但在分发给他人使用时，确保 `bootstrap.sh` 具备模型校验功能，防止运行时因缺少权重报错。