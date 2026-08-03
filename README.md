# Local AI Document RAG Stack

A reusable, framework-agnostic Python package that converts business documents (PDFs, scans, images, spreadsheets) into structured, searchable, and answerable data without sending data to cloud APIs.

## Features

- **Hybrid retrieval**: combines text-extraction RAG with visual document retrieval.
- **Late-interaction multi-vector support**: ColQwen/ColPali patch-level embeddings with MaxSim scoring.
- **Local-first**: runs entirely on-premise via Ollama and/or Apple MLX.
- **Pluggable storage**: sqlite-vec for dense vectors, JSONL-backed in-memory store for multi-vector pages, Qdrant optional.
- **Multimodal Q&A**: answers with citations over text chunks and page images.
- **Appliance-ready**: single-file vector store, health checks, and simple packaging.

## Architecture

```
Document
   ├─→ Docling / Marker / Plaintext → structured text chunks
   │                                    └─→ dense text embeddings (sqlite-vec/Qdrant)
   └─→ Render pages
        ├─→ Qwen3-VL / Ollama → dense visual embeddings
        └─→ ColQwen2 / ColPali → multi-vector late-interaction embeddings
```

Query time:
1. Retrieve from dense text index.
2. Retrieve from dense visual index (if enabled).
3. Retrieve from late-interaction multi-vector index via MaxSim (if enabled).
4. Fuse all ranked lists with Reciprocal Rank Fusion (RRF).
5. Rerank top-K with a cross-encoder or Ollama prompt.
6. Send best page images + text chunks to VLM for cited answer.

## Quick Start

```bash
# Use Python 3.10–3.12 (3.14 is not supported by many ML wheels)
python3.12 -m venv .venv
source .venv/bin/activate

# Install with all recommended backends
pip install -e ".[ollama,docling,visual,qdrant]"

# Install Apple Silicon extras on macOS
pip install -e ".[mlx,mac]"

# Pull models into Ollama
ollama pull qwen3:8b
ollama pull qwen3-vl
ollama pull qwen3-embedding:4b

# Run the API
lrs serve

# Or use the CLI
lrs ingest sample.pdf
lrs query "What is the effective date of the contract?"
```

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `LRS_OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `LRS_TEXT_EMBED_MODEL` | `qwen3-embedding:4b` | Text embedding model (Ollama or sentence-transformers) |
| `LRS_VISION_EMBED_MODEL` | `qwen3-vl` / `vidore/colqwen2-v1.0` | Visual embedding model |
| `LRS_VISION_EMBED_BACKEND` | `ollama` | `ollama` for dense visual, `hf` / `colqwen` for late-interaction multi-vector |
| `LRS_RERANK_MODEL` | `bge-reranker-v2-m3` | Reranker model |
| `LRS_RERANK_BACKEND` | `ollama` | `ollama`, `sentence_transformers`, or `none` |
| `LRS_LLM_MODEL` | `qwen3-vl` | Answer generation model |
| `LRS_VECTOR_STORE` | `sqlite` | `sqlite` or `qdrant` (dense store only) |
| `LRS_SQLITE_VEC_PATH` | `./data/lrs.vec.sqlite` | sqlite-vec file path |
| `LRS_EXTRACTION_ENGINE` | `docling` | `docling`, `marker`, or `plaintext` |
| `LRS_FORCE_FULL_PAGE_OCR` | `false` | Force OCR when a PDF has an unreliable native text layer |
| `LRS_OCR_ENGINE` | `auto` | `auto` prefers macOS Vision (OCRMac) on Apple Silicon; otherwise uses Docling defaults |

## Docker / standalone service deployment

The recommended production boundary is a **standalone HTTP container** that CaaS / decision-saas call over the network.

```bash
# 1. Copy and edit environment variables
cp .env.example .env

# 2. Build and start the API + a co-located Ollama service
docker compose up --build

# 3. Pull the models you need into the Ollama container
docker exec -it lrs-ollama ollama pull qwen3:8b
docker exec -it lrs-ollama ollama pull qwen3-embedding:4b

# 4. Smoke test
curl http://localhost:8000/health
curl -F file=@sample.pdf http://localhost:8000/ingest
curl -X POST -H "Content-Type: application/json" \
  -d '{"query":"Summarise the contract"}' \
  http://localhost:8000/query
```

The default image uses the lightweight text-only extras (`ollama,qdrant`) and the `plaintext` extraction engine, so it does not need Docling or Marker inside the container. To enable ColQwen visual retrieval, rebuild with the `visual` extra:

```bash
docker compose build --build-arg LRS_EXTRAS=ollama,visual,qdrant lrs
```

To use an Ollama instance running on the Docker host instead of the compose service, override `LRS_OLLAMA_HOST`:

```bash
docker run -p 8000:8000 \
  -e LRS_OLLAMA_HOST=http://host.docker.internal:11434 \
  -v lrs-data:/app/data \
  local-rag-stack:latest
```

## Tenant-aware API wrapper

When `LRS_AUTH_REQUIRED=true`, every API request must carry an API key. The key maps to a tenant, and each tenant gets isolated storage:

```bash
# inline keys
LRS_AUTH_REQUIRED=true
LRS_TENANT_KEYS_JSON='{"tenant-a-key":"tenant-a","tenant-b-key":"tenant-b"}'

# or load from file
LRS_TENANT_KEYS_FILE=./config/tenant_keys.json
```

Then call the API with the key:

```bash
curl -H "X-API-Key: tenant-a-key" http://localhost:8000/health
curl -H "X-API-Key: tenant-a-key" -F file=@report.pdf http://localhost:8000/ingest
```

Per-tenant isolation is enforced at the storage layer:

- SQLite vector DB: `data/tenants/<tenant-id>/lrs.vec.sqlite`
- Graph DB: `data/tenants/<tenant-id>/lrs.graph.sqlite`
- Multi-vector JSONL: stored next to the tenant sqlite DB
- Qdrant collection: `<base-collection>-<tenant-id>`

Embedders, generators, and the Ollama client are reused across tenants; only the stores are namespaced.

## Multi-vector / ColQwen mode

Set `LRS_VISION_EMBED_BACKEND=hf` and `LRS_VISION_EMBED_MODEL=vidore/colqwen2-v1.0` to enable late-interaction visual retrieval. The multi-vector index is persisted as a JSONL file next to the sqlite-vec database and supports exact MaxSim scoring. For very large corpora, replace `MultiVectorStore` with an approximate nearest-neighbor backend that supports multi-vector search.

If you already have a recent PyTorch installed and want to avoid a forced downgrade, install the visual backend manually:

```bash
pip install torch torchvision transformers peft accelerate
pip install --no-deps colpali-engine
```

## API Endpoints

- `POST /ingest` — ingest a document (multipart upload or path)
- `POST /query` — ask a question over all ingested documents
- `GET /documents` — list ingested documents
- `GET /health` — service health and model status

## Experiments

Run labeled eval sets and record keep/revert decisions with the evaluator-optimizer loop:

```bash
# Run an eval set (JSONL of EvalQuestion records)
lrs experiment run --eval-set scripts/sample_eval_set.jsonl --name "text-only baseline"

# Override strategy knobs for the experiment
lrs experiment run \
  --eval-set scripts/sample_eval_set.jsonl \
  --name "plaintext + bge-small" \
  --strategy-json '{"extraction_engine":"plaintext","text_embed_model":"BAAI/bge-small-en-v1.5"}'

# Review and decide
lrs experiment list
lrs experiment show <experiment_id>
lrs experiment decide <experiment_id> --outcome keep --reason "beats baseline on p@k"
```

Experiments, per-question results, and decisions are stored in a separate SQLite database (`data/lrs.experiments.sqlite`) so eval metadata does not mix with production vector/graph data.

## Production hand-off

See [`HANDOFF.md`](HANDOFF.md) for the complete production hand-off:

- Tenant auth setup and per-tenant storage isolation.
- Docker Compose deployment steps.
- Integration patterns for CaaS / decision-saas.
- Production checklist and known limitations.
- Evaluator-optimizer usage with `lrs experiment`.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

MIT. Optional dependencies such as `marker-pdf` are GPL-3 and must be used in compliance with their respective licenses.
