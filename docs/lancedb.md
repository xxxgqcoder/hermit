# LanceDB 存储说明

Hermit 使用 [LanceDB](https://lancedb.github.io/lancedb/) 作为唯一的向量存储后端：嵌入式、列存（Apache Arrow）、单进程开箱即用，无需 Docker、无独立服务。

## 磁盘布局

每个知识库对应一张 LanceDB 表：

```
~/.hermit/data/lance/
└── <collection-name>.lance/    # Lance 数据集目录
    ├── _versions/              # 版本元数据
    ├── _indices/               # FTS / 标量 / 向量索引
    └── data/                   # Arrow 数据文件
```

`lancedb.connect("~/.hermit/data/lance")` 即可打开整库。

## 表 Schema

```text
id          : string                          (PK，UUID)
text        : string                          (FTS 索引)
title       : string                          (原始大小写文件名)
filename    : string                          (小写化的 basename stem，FTS 索引)
source_file : string                          (绝对路径，BTREE 标量索引)
chunk_index : int32
total_chunks: int32
vector      : fixed_size_list<float32, 768>   (jina-embeddings dense 向量)
```

## 索引策略

| 索引 | 列 | 时机 | 说明 |
|---|---|---|---|
| FTS（LanceDB 原生 inverted index） | `text` | `ensure_collection` 时建 | hybrid / keyword 模式的关键词召回。`use_tantivy=False, with_position=False` |
| FTS（LanceDB 原生 inverted index） | `filename` | `ensure_collection` 时建 | filename 子串过滤候选 |
| BTREE | `source_file` | `ensure_collection` 时建 | `delete_by_source_file` 必经路径 |
| IVF/HNSW | `vector` | 行数 ≥ `VECTOR_INDEX_THRESHOLD`（默认 50_000）时懒建 | 小规模下裸扫更快，避免索引构建/再训练开销 |

懒建的好处：百万级以下的数据用 brute-force scan 反而更快、零维护；只有真正需要近似检索时才付出索引训练成本。阈值由 `hermit/storage/lance.py` 的常量 `VECTOR_INDEX_THRESHOLD` 控制。

## FTS 行为

- **分词器**：LanceDB 原生 FTS（`use_tantivy=False`）默认使用简单 lowercase 分词器，无词干提取
- **多列**：单个 FTS 索引只能覆盖一列，因此 `text` 与 `filename` 各建一个索引
- **更新可见性**：`tbl.add(...)` 写入后必须显式 `tbl.optimize()` 才能让新行进入 FTS 索引。`replace_file_chunks(optimize=True)` 默认内置该调用（同步/一次性写入用）；后台索引队列改为写入时 `optimize=False`、整批排空后只 `optimize_collection()` 一次，避免每文件重建索引（见下「压缩与版本回收」）
- **查询时**：`tbl.search(query, query_type="fts", fts_columns="text")` 显式指定查询哪一列

## 混合检索（hybrid）

```python
from lancedb.rerankers import RRFReranker

tbl.search((dense_vec, query_text), query_type="hybrid")
   .where(filter_sql, prefilter=True)         # filename substring，可选
   .limit(rerank_candidates)
   .rerank(RRFReranker())
   .to_list()
```

LanceDB 内部并行跑向量召回和 FTS 召回，由 `RRFReranker` 按倒数排名融合。`RRFReranker` 仅按排序融合，**不接受 per-query 权重**——这是相对 Qdrant 时代 `w_dense`/`w_sparse` 的一处显式收窄，需要加权融合时只能各自跑一遍向量与 FTS、再在 Python 侧拼接。

## 子串与 glob 过滤

filename 过滤分两条路径：

- 普通子串 → 转成 SQL `contains(lower(filename), '<needle>')`，作为 LanceDB 的 prefilter 下推
- glob（含 `*?[`）→ 走全表扫描 + Python 端 `fnmatch` 后置过滤

fuzzy 模式同样基于 `tbl.search().where("contains(lower(text), '<needle>')")` 的标量扫描——不再需要 Qdrant 时代的滚动分页 + 上限保护，因为 LanceDB 的 `where` 谓词由 DataFusion 下推到底层数据文件，扫描成本与命中数线性相关。

## 压缩与版本回收

**核心坑**：`optimize()` 每次都把 FTS 倒排索引**重建**到一个新的 `_indices/<uuid>` 目录，旧目录变成孤儿；而 LanceDB 的 `cleanup_old_versions` 只回收过期的**数据版本**（manifest），**根本不删 `_indices/` 下的孤儿索引目录**（在 lancedb 0.30.2 上实测：40 轮 delete+add+optimize 配 `cleanup_older_than=0` 后,磁盘仍残留 ~79 个孤儿索引目录）。早期 Hermit 每写一个文件就 `optimize()` 一次,叠加一个 re-index 死循环,使 `_indices/` 累积到 5 万+ 目录、单表上百 GB,而逻辑数据只有几百 MB。

对此分三层处理：

- **不再每文件 optimize**。`replace_file_chunks` 只写行；后台索引队列在一批文件排空后只调用一次 `optimize_collection()`（见 `ingestion/task_queue.py` 的 `_flush_collection`）。这把繁忙集合上从「每文件一次、累计数十万次」的 optimize 降到「每突发一次」,churn 与 CPU 同步骤降。`cleanup_older_than=1min`（`lance._OPTIMIZE_CLEANUP_OLDER_THAN`）让单次突发的瞬时版本及时回收；最新版本永不删除。
- **唯一可靠的孤儿回收 = 重建表**。`vacuum_collection()` 把当前 live 行重建成一张只带 live 索引的新表,磁盘体积塌回逻辑数据大小。**崩溃安全**是设计要点:先在临时表 `<name>__vacuum` 里建好数据+索引（所有耗时且可能 panic 的步骤——尤其 FTS builder——都发生在这里,原表毫发无损）,**校验行数一致后**才 `drop` 原表 + `os.rename` 临时目录就位（同盘原子 rename,窗口极小且零计算）。任何步骤在 rename 前失败,原表都完好;若恰好崩在 drop 与 rename 之间,启动时 `recover_vacuum_temp()` 会按目录状态机收拾残局（原表在→丢弃临时表;原表不在→把已建好的临时表提升上位）。注意 `to_arrow()` 会把全部行（含 dense 向量）读进内存,当前规模（数万行 ≈ 数百 MB）无碍,逼近 100 万 chunk 的 deep-search 目标时需改成分批流式。
- **按需触发**。`maybe_vacuum()` 仅当磁盘 `_indices` 目录数超过 `VACUUM_INDEX_DIR_THRESHOLD`（默认 32,健康表只有 ~3）时才重建——平时只是一次 `listdir`,极廉价。启动时（`app.py` 后台线程）和每次突发 flush 后都会调一次,实现自愈,无需人工干预。崩溃恢复 `recover_vacuum_temp()` 则在 `app.py` 恢复 collection、打开表**之前**先跑。

一次性回收历史垃圾：停掉 server 后跑 `scripts/reclaim_lance.py`（对每个集合做一次 `vacuum_collection`）。

背景见提交 “Stop LanceDB from accumulating 100x its logical size” 以及本次孤儿索引回收。

## 并发性

- 读：`open_table` 多次调用、跨线程都安全，每次返回最新版本视图
- 写：LanceDB 通过版本化 manifest + 原子提交支持并发写
- Hermit 不再持有自己的全局锁（旧 Qdrant local 模式因为 numpy 数组非线程安全才需要的）

## 与 Qdrant 时代的差异（破坏性变更）

- `~/.hermit/data/qdrant/` 不再使用，可手工删除
- `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_MANAGED` 等环境变量已全部移除
- `/health` 响应去掉 `qdrant_mode` / `qdrant_host`，新增 `storage: "lance"`
- `SearchRequest` 去掉 `w_dense` / `w_sparse`
- 不再加载 `Qdrant/bm25` sparse embedding 模型；`~/.hermit/cache/sparse/` 目录可手工删除
- `qdrant_mode_signature` 文件不再写入；遗留的 `~/.hermit/data/qdrant_mode.json` 可删除
