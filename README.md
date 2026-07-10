# Hermit

[中文说明 / Chinese version](./README_cn.md)

Hermit is a **self-contained local semantic search service** for turning one or more document folders into searchable knowledge-base collections.

It is designed for local-first workflows and works well as a lightweight retrieval backend for notes, technical documents, and small RAG-style applications.

## Highlights

- **Runs fully locally**: models, vector data, and metadata live inside the project
- **Semantic Markdown chunking**: a state-machine parser splits `.md` files into 11 semantically coherent block types (headings, fenced code, math, tables, blockquotes, lists, …) before chunking — code blocks and tables are never split mid-content
- **Heading-aware sliding window**: chunks always start at a section heading when possible, so each retrieved chunk is self-contained and retrieval-friendly even without surrounding context
- **Embedded vector store**: LanceDB columnar storage, no Docker, no external server
- **Multi-collection support**: one folder maps to one collection
- **Hybrid retrieval**: dense vectors + tantivy FTS (BM25) fused via RRF
- **Reranking**: a cross-encoder reranks the fused candidates
- **Incremental sync**: startup scan plus periodic polling
- **CPU-friendly**: built on `fastembed` + ONNX Runtime, no GPU required

## What it is good for

Hermit is a good fit when you want to:

- search a local notes or markdown repository semantically
- expose a simple retrieval API for a local tool or agent
- build a private, small-footprint RAG layer without cloud dependencies

The current implementation reads files as text using `UTF-8` with replacement on decode errors, so it works best with plain-text sources such as `.md` and `.txt` files.

## How it works

### Retrieval pipeline

Hermit uses the following search flow:

1. Encode the query into a dense vector
2. Run hybrid retrieval in LanceDB (vector + tantivy FTS) with RRF fusion
3. Rerank the candidate set with a cross-encoder
4. Return the top matching chunks

### Indexing pipeline

Each registered folder goes through:

1. startup scan
2. SQLite metadata diffing
3. text chunking (see below)
4. dense embedding generation
5. LanceDB upsert (FTS index updates automatically)
6. ongoing periodic polling

### Markdown semantic chunking

For `.md` files, Hermit uses a two-phase strategy instead of a simple token sliding window:

**Phase 1 — block parsing (`parse_md_blocks`)**: A state-machine parser scans the file line-by-line and groups content into 11 semantically coherent block types:

| Block type | Examples |
|---|---|
| YAML frontmatter | `---` … `---` at file start |
| Fenced code block | ` ``` ` … ` ``` ` or `~~~` … `~~~` |
| Math block | `$$` … `$$` |
| ATX heading | `# H1` … `###### H6` |
| Setext heading | underline with `===` or `---` |
| Table | pipe-delimited rows |
| Blockquote | `>` prefixed lines |
| Horizontal rule | `---` / `***` / `___` |
| List | entire list including nested items |
| Standalone image | `![alt](url)` or Obsidian `![[path]]` |
| Paragraph | any other contiguous non-blank text |

This ensures fenced code blocks, math formulas, and tables are always kept intact as a unit.

**Phase 2 — heading-aware sliding window (`chunk_markdown`)**: Blocks are grouped into chunks (default: 4 blocks per chunk) with two structural rules:

- **Rule 1 — no orphan headings**: if the last block in a chunk is a heading, the chunk is automatically extended by one block so the heading always enters a chunk together with at least its first body block.
- **Rule 2 — heading-anchored start**: the next chunk begins at the nearest preceding heading rather than at an arbitrary paragraph, so every chunk carries its own section context.

Other file types continue to use the token-based sliding window (`chunk_text`).

See [docs/markdown-chunking.md](docs/markdown-chunking.md) for the full design.

### Default settings

- Chunk size: `256` tokens (using the embedding model's tokenizer)
- Chunk overlap: `32` tokens
- Search `top_k`: `5`
- Default rerank candidates: `20`
- Max collections: `4`
- Max collection name length: `64`
- Default port: `8000`

## Tech stack

- **API framework**: FastAPI
- **Vector database**: LanceDB (embedded, columnar, tantivy FTS)
- **Inference backend**: fastembed (ONNX-based, parallelized via `ThreadPoolExecutor`)
- **Metadata store**: SQLite
- **Filesystem watcher**: periodic polling

Current models:

- Dense embedding: `jinaai/jina-embeddings-v2-base-zh`
- Reranker: `jinaai/jina-reranker-v2-base-multilingual`

Keyword recall is provided by LanceDB's tantivy FTS index — no separate
sparse embedding model is loaded. See [docs/lancedb.md](docs/lancedb.md)
for the storage layout, index strategy, and query semantics.

## Performance & Memory

Hermit is optimized for stable local search memory usage:
- **Serialized Search**: Search requests run through a single-worker executor. This keeps the shared ONNX sessions from serving multiple reranker requests concurrently.
- **Bounded ONNX Inference**: Uses `HERMIT_ONNX_THREADS=2` by default to keep ONNX Runtime per-thread arenas small. Raise it only after measuring that latency improves enough to justify the extra resident memory.
- **Smaller Rerank Pool**: Uses 20 candidates per query by default while keeping cross-encoder reranking enabled.
- **Embedding Cache**: Indexing skips ONNX inference for chunks whose exact model input was seen before. Dense vectors are cached on disk (`HERMIT_HOME/cache/dense`, sha256-keyed by `model_name::input_text`) with a 30-day TTL. Cache hits validate the vector dimension and fall back to a fresh embed on mismatch — model upgrades or partially-corrupted entries are self-healing. Always on by design; the cache is bounded and self-reaping.

## Project layout

```text
.
├── main.py
├── pyproject.toml
├── README.md / README_cn.md
├── docs/
│   ├── design.md
│   ├── lancedb.md
│   ├── markdown-chunking.md
│   └── skill-distribution.md
├── hermit/
│   ├── app.py                 # FastAPI app + lifespan
│   ├── cli.py                 # CLI entry (JSON output)
│   ├── config.py              # paths, defaults, env vars
│   ├── models.py              # model download + verification
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── ingestion/
│   │   ├── chunker.py         # markdown / token chunking
│   │   ├── scanner.py
│   │   ├── task_queue.py
│   │   └── watcher.py
│   ├── retrieval/
│   │   ├── embedder.py
│   │   ├── embed_cache.py     # diskcache-backed dense embedding cache
│   │   ├── reranker.py
│   │   └── searcher.py
│   └── storage/
│       ├── lance.py           # LanceDB-backed vector store
│       ├── metadata.py        # SQLite per-collection
│       ├── model_signature.py
│       ├── quantizer.py       # INT8 quantization for dense embedder
│       └── registry.py
└── tests/

~/.hermit/                     # runtime data (override with HERMIT_HOME)
├── models/                    # ONNX weights (fastembed cache)
├── cache/dense/               # dense embedding cache (sha256-keyed, 30-day TTL)
├── data/
│   ├── lance/                 # LanceDB tables (one per collection)
│   ├── metadata/              # SQLite per-collection
│   ├── collections.json       # registry
│   └── model_signature.json
├── logs/hermit.log
└── hermit.pid
```

## Installation

### Requirements

- Python `3.12 ~ 3.13`
- macOS or Linux

### Install as a CLI tool (recommended)

```bash
uv tool install git+https://github.com/xxxgqcoder/hermit.git
```

This drops a `hermit` executable into `~/.local/bin/` with its own isolated environment — no venv to activate. To upgrade later: `uv tool install git+https://github.com/xxxgqcoder/hermit.git --force` (or `uv tool upgrade hermit`).

After installing, deploy the bundled agent skill so Claude / other agents can discover Hermit automatically:

```bash
hermit install-skills
```

### Development install (from source)

```bash
git clone https://github.com/xxxgqcoder/hermit.git
cd hermit
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you plan to use `hermit download`, make sure `huggingface_hub` is available in your environment, since the CLI uses it to download model snapshots.

## Quick start

### 1. Download models (optional but recommended)

```bash
hermit download
```

Optional flags:

```bash
hermit download --force
hermit download --skip-verify
```

Notes:

- missing models can also be downloaded automatically on first service startup
- downloading them explicitly makes first boot less surprising and easier to monitor

### 2. Register a knowledge-base folder

```bash
hermit kb add my_docs ./documents
```

Optional flags:

```bash
hermit kb add my_docs ./documents \
  --ignore "build/**" --ignore "*.tmp" \
  --ignore-ext .pdf --ignore-ext .png
```

List collections:

```bash
hermit kb list
```

Update ignore rules:

```bash
hermit kb update my_docs --ignore "dist/**" --ignore-ext .log
hermit kb update my_docs --clear-ignore --clear-ignore-ext
```

Remove a collection:

```bash
hermit kb remove my_docs
```

Collection naming rules:

- must start with a letter or digit
- may contain only letters, digits, underscores, and hyphens
- must be unique

### 3. Start the service

```bash
hermit start
```

On startup, Hermit will:

- warm up embedding and reranker models
- start the background indexing worker
- restore persisted collections from `~/.hermit/data/collections.json`
- scan each collection folder
- start watching registered folders for changes

Default bind address:

- Host: `0.0.0.0`
- Port: `8000`

Other server commands:

```bash
hermit status     # health JSON (mode, collections, pending tasks, ...)
hermit logs       # tail the server log
hermit stop       # graceful shutdown
```

### 4. Search

CLI (recommended — JSON output, no curl glue needed):

```bash
hermit search my_docs "two sum approach"
hermit search my_docs "lancedb" --mode keyword          # FTS only, faster
hermit search my_docs "binary"  --mode fuzzy            # substring scan
hermit search my_docs "design"  --mode fuzzy --filename "*.md"
hermit search my_docs "embedding" --no-rerank --top-k 3
```

HTTP equivalent:

```bash
curl -X POST http://127.0.0.1:8000/search \
	-H 'Content-Type: application/json' \
	-d '{
		"query": "two sum approach",
		"collection": "my_docs",
		"top_k": 5,
		"rerank_candidates": 20,
		"mode": "hybrid"
	}'
```

Supported `mode` values: `hybrid` (default), `semantic`, `keyword`, `fuzzy`.

## CLI

All commands output JSON to stdout. Add `--pretty` for indented output. Errors are reported as `{"error": "message"}` with non-zero exit code.

### Server lifecycle

| Command | Purpose |
|---|---|
| `hermit start` | Start the server in background (uvicorn daemon). |
| `hermit stop` | Graceful shutdown (SIGTERM, falls back to SIGKILL after 10s). |
| `hermit status` | Health JSON: mode, uptime, collections, pending tasks. |
| `hermit logs` | Tail `~/.hermit/logs/hermit.log` (streaming, not JSON). |

### `hermit download`

Download all required models and optionally run a basic verification step.

Flags:

- `--force`: force re-download
- `--skip-verify`: skip post-download verification

### `hermit search <collection> [<query>]`

Semantic / keyword / fuzzy search. JSON-formatted results.

Flags:

- `--mode {hybrid,semantic,keyword,fuzzy}` — default `hybrid`
- `--top-k N` — number of results (default 5)
- `--rerank-candidates N` — recall pool before rerank (default 20)
- `--filename PATTERN` — orthogonal filename filter (substring or glob)
- `--rerank` / `--no-rerank` — override the mode's default rerank behavior

In `fuzzy` mode the `query` is optional when `--filename` is supplied.

### `hermit kb add <name> <dir>`

Register a folder as a collection.

Flags:

- `--ignore PATTERN` — glob path to ignore (repeatable)
- `--ignore-ext EXT` — file extension to ignore, e.g. `.pdf` (repeatable)

### `hermit kb update <name>`

Replace ignore rules for an existing collection (not additive).

Flags:

- `--ignore PATTERN` — new path-ignore list (replaces previous)
- `--ignore-ext EXT` — new ext-ignore list (replaces previous)
- `--clear-ignore` — drop all path ignore patterns
- `--clear-ignore-ext` — drop all extension ignore patterns

### `hermit kb remove <name>`

Remove a collection and delete its metadata store.

### `hermit kb list`

List all registered collections.

### `hermit collection <subcommand> <name>`

Query live collection state from a running server (requires `hermit start`).

- `hermit collection status <name>` — indexing status (`indexed_files`, `total_chunks`, `watching`)
- `hermit collection sync <name>` — trigger a manual rescan
- `hermit collection tasks <name>` — background task queue status

### `hermit install-skills`

Install bundled agent skill specs to `~/.agents/skills/`.

- `--uninstall` removes previously installed skills

## HTTP API

The current codebase exposes the following endpoints.

### `POST /search`

Run hybrid / semantic / keyword / fuzzy search.

Request example:

```json
{
	"query": "sliding window maximum",
	"collection": "my_docs",
	"top_k": 5,
	"rerank_candidates": 20,
	"mode": "hybrid",
	"filename": null,
	"rerank": null
}
```

`mode`: `hybrid` (default), `semantic`, `keyword`, `fuzzy`. `filename` accepts a substring or glob (e.g. `*.md`). `rerank` overrides the mode's default cross-encoder behavior (`null` keeps the default; `false` skips rerank; `true` forces it).

Response example:

```json
{
	"results": [
		{
			"text": "...",
			"source_file": "/abs/path/to/file.md",
			"chunk_index": 0,
			"total_chunks": 3,
			"score": 0.82
		}
	]
}
```

### `POST /collections`

Register a new collection. The server scans the folder asynchronously and persists the entry to the registry; subsequent restarts auto-restore it.

Request:

```json
{
	"name": "my_docs",
	"folder_path": "/abs/path/to/docs",
	"ignore_patterns": ["build/**"],
	"ignore_extensions": [".pdf"]
}
```

Returns `409` if the name already exists, `400` for invalid name or folder path.

### `DELETE /collections/{name}`

Remove a collection: stop the watcher, drain its task queue, drop the LanceDB table, delete the SQLite metadata DB, and unregister.

Returns `409` if background indexing tasks for that collection cannot drain within 30s — retry shortly.

### `POST /collections/{name}/sync`

Trigger a manual scan/sync for an existing collection. Response: `{ "added": N, "updated": M, "deleted": K }`.

### `GET /collections/{name}/status`

Returns `{ name, folder_path, indexed_files, total_chunks, watching }`.

### `GET /collections/{name}/tasks`

Returns `{ collection, pending_tasks, queued_tasks, in_progress_tasks, worker_alive }`.

### `GET /health`

Server health and runtime info.

Response fields:

- `status` — `"ready"` or `"starting"`
- `uptime` — seconds since server start
- `models_loaded` — whether embedding/reranker models are loaded
- `collections` — list of `{name, indexed_files, total_chunks}` per collection
- `pending_index_tasks` — total background indexing tasks waiting across all collections
- `storage` — always `"lance"` (LanceDB-backed embedded store)

## Storage layout

Hermit keeps all runtime data under `~/.hermit/` by default (override with `HERMIT_HOME` env var):

- `~/.hermit/models/`: local model cache (fastembed ONNX weights)
- `~/.hermit/data/lance/`: LanceDB tables — one per collection
- `~/.hermit/data/metadata/`: one SQLite database per collection
- `~/.hermit/data/collections.json`: persisted collection configuration
- `~/.hermit/cache/dense/`: dense embedding cache (sha256-keyed, 30-day TTL)
- `~/.hermit/logs/hermit.log`: server log, rotated at 256 MiB with one backup (`hermit.log.1`)
- `~/.hermit/hermit.pid` and `~/.hermit/port.json`: daemon bookkeeping

Single directory makes Hermit easy to back up, move, and clean up.

## Indexing behavior

### File handling

- recursively scans all non-hidden files
- skips any path segment starting with `.`
- reads files as text with `utf-8` and `errors="replace"`

### Change detection

Hermit tracks indexed files in SQLite and uses **SHA256** to detect content changes.

During scanning it handles:

- **new files**: enqueue or index them
- **modified files**: rechunk, re-embed, and replace old chunks
- **deleted files**: remove them from LanceDB and SQLite

### Chunking rules

- default chunk size is `256` tokens (using the embedding model's tokenizer)
- adjacent chunks overlap by `32` tokens
- empty text is skipped
- short text stays as a single chunk

## Known limitations

- Hybrid fusion uses LanceDB's built-in RRF reranker; explicit `w_dense`/`w_sparse` knobs are no longer exposed
- all files are treated as text; PDF, image, and Office parsing are out of scope
- the maximum number of collections is currently `4`
- first-time model downloads may take a while and use noticeable disk space

## Development and testing

The test suite currently covers:

- CLI validation and collection management
- scanner add/update/delete logic
- task queue status reporting
- selected API route behavior

Run tests with:

```bash
pytest
```

## Design notes

For implementation details, see:

- `docs/design.md`

## In one sentence

If you want a small, local-first, multi-collection semantic search service that quietly gets the job done, Hermit fits the brief nicely.
