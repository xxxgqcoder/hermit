# Hermit

[中文说明 / Chinese version](./README_cn.md)

Hermit is a **self-contained local semantic search service** for turning one or more document folders into searchable knowledge-base collections.

It is designed for local-first workflows and works well as a lightweight retrieval backend for notes, technical documents, and small RAG-style applications.

## Highlights

- **Runs fully locally**: models, vector data, and metadata live inside the project
- **Multi-collection support**: one folder maps to one collection
- **Hybrid retrieval**: dense + sparse recall
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

1. Encode the query into dense and sparse representations
2. Run hybrid retrieval in Qdrant
3. Fuse candidates with RRF
4. Rerank the candidate set
5. Return the top matching chunks

### Indexing pipeline

Each registered folder goes through:

1. startup scan
2. SQLite metadata diffing
3. text chunking
4. embedding generation
5. Qdrant upsert
6. ongoing periodic polling

### Default settings

- Chunk size: `256` tokens (using the embedding model's tokenizer)
- Chunk overlap: `32` tokens
- Search `top_k`: `5`
- Default `w_dense`: `0.7`
- Default `w_sparse`: `0.3`
- Default rerank candidates: `50`
- Max collections: `4`
- Max collection name length: `64`
- Default port: `8000`

## Tech stack

- **API framework**: FastAPI
- **Vector database**: Qdrant embedded mode
- **Inference backend**: fastembed
- **Metadata store**: SQLite
- **Filesystem watcher**: periodic polling

Current models:

- Dense embedding: `jinaai/jina-embeddings-v2-base-zh`
- Sparse embedding: `Qdrant/bm25`
- Reranker: `jinaai/jina-reranker-v2-base-multilingual`

## Project layout

```text
.
├── main.py
├── pyproject.toml
├── README.md
├── README_cn.md
├── docs/
│   └── design.md
├── hermit/
│   ├── cli.py
│   ├── config.py
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── scanner.py
│   │   ├── task_queue.py
│   │   └── watcher.py
│   ├── retrieval/
│   │   ├── embedder.py
│   │   ├── reranker.py
│   │   └── searcher.py
│   └── storage/
│       ├── metadata.py
│       ├── model_signature.py
│       ├── qdrant.py
│       └── registry.py
├── data/
│   ├── collections.json
│   ├── metadata/
│   └── qdrant/
└── models/
```

## Installation

### Requirements

- Python `3.12+`
- macOS or Linux

Using a virtual environment is recommended.

### Install from source

```bash
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

List collections:

```bash
hermit kb list
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
python main.py
```

On startup, Hermit will:

- warm up embedding and reranker models
- start the background indexing worker
- restore persisted collections from `data/collections.json`
- scan each collection folder
- start watching registered folders for changes

Default bind address:

- Host: `0.0.0.0`
- Port: `8000`

### 4. Search

```bash
curl -X POST http://127.0.0.1:8000/search \
	-H 'Content-Type: application/json' \
	-d '{
		"query": "two sum approach",
		"collection": "my_docs",
		"top_k": 5,
		"w_dense": 0.7,
		"w_sparse": 0.3,
		"rerank_candidates": 30
	}'
```

## CLI

Hermit currently provides these CLI commands.

### `hermit download`

Download all required models and optionally run a basic verification step.

```bash
hermit download
```

Flags:

- `--force`: force re-download
- `--skip-verify`: skip post-download verification

### `hermit kb add <name> <dir>`

Register a folder as a collection.

```bash
hermit kb add notes ./documents
```

### `hermit kb remove <name>`

Remove a collection and delete its metadata store.

```bash
hermit kb remove notes
```

### `hermit kb list`

List all registered collections.

```bash
hermit kb list
```

## HTTP API

The current codebase exposes the following endpoints.

### `POST /search`

Run hybrid semantic search.

Request example:

```json
{
	"query": "sliding window maximum",
	"collection": "my_docs",
	"top_k": 5,
	"w_dense": 0.7,
	"w_sparse": 0.3,
	"rerank_candidates": 30
}
```

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

### `POST /collections/{name}/sync`

Trigger a manual scan/sync for a collection.

Response fields:

- `added`
- `updated`
- `deleted`

Example:

```bash
curl -X POST http://127.0.0.1:8000/collections/my_docs/sync
```

### `GET /collections/{name}/status`

Get collection status.

Response fields:

- `name`
- `folder_path`
- `indexed_files`
- `total_chunks`
- `watching`

Example:

```bash
curl http://127.0.0.1:8000/collections/my_docs/status
```

### `GET /collections/{name}/tasks`

Get background indexing task status for a collection.

Response fields:

- `collection`
- `pending_tasks`
- `queued_tasks`
- `in_progress_tasks`
- `worker_alive`

Example:

```bash
curl http://127.0.0.1:8000/collections/my_docs/tasks
```

## Storage layout

By default, Hermit stores its runtime data inside the project directory:

- `models/`: local model cache
- `data/qdrant/`: Qdrant embedded data
- `data/metadata/`: one SQLite database per collection
- `data/collections.json`: persisted collection configuration

That makes the project easy to back up, move, and clean up. No mysterious hidden cave system under your home directory.

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
- **deleted files**: remove them from Qdrant and SQLite

### Chunking rules

- default chunk size is `256` tokens (using the embedding model's tokenizer)
- adjacent chunks overlap by `32` tokens
- empty text is skipped
- short text stays as a single chunk

## Known limitations

- there is currently **no API** to create or delete collections; use the CLI for that
- `w_dense` and `w_sparse` are accepted by the API, but the current implementation uses **RRF fusion** rather than explicit weighted score fusion
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