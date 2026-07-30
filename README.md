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
pip install -e ".[mlx]"

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

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

## License

MIT. Optional dependencies such as `marker-pdf` are GPL-3 and must be used in compliance with their respective licenses.
