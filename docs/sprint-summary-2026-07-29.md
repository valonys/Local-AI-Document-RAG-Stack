# Local RAG Stack — Sprint Summary (2026-07-29)

## Executive summary

This sprint delivered a working **local, multi-vector document RAG appliance** with two proven retrieval paths and the first phase of a Karpathy-style agentic graph layer.

- **Text-only RAG** (dense embeddings + sqlite-vec + Ollama generation) is production-ready for fast, low-cost document Q&A.
- **Visual/ColQwen RAG** (late-interaction multi-vector embeddings over rendered pages) is working on M5 Pro Apple Silicon after fixing model-loading and binary-persistence bugs.
- **Knowledge-graph Phase 1** (entity/relation extraction via structured LLM outputs + SQLite graph store + API/CLI queries) is wired into the ingestion pipeline.
- All changes are contained in `local_rag_stack` and pass the existing test suite.

---

## Benchmark results

### Text-only baseline

| File | Chunks | Embed time |
|---|---|---|
| `CA600B-VII.pdf` | 12 | ~49.7 s |
| `73192866_VIE_BDV2088B_2022.xlsx` | 252 | ~0.6 s |
| **Total** | **264** | **~50 s** |

- Embed model: `BAAI/bge-small-en-v1.5` on MPS
- Generation: `qwen3.6:latest` via Ollama (~20–30 s per query)
- Retrieval: <0.1 s
- Correctly identified the XLSX as an inspection report for equipment `BDV 2088 B`, cited sheets/pages, and answered that no financial figures were present.

### ColQwen visual RAG

| File | Pages | Embed time |
|---|---|---|
| `CA600B-VII.pdf` | 3 | 17.28 s |
| `73192866_VIE_BDV2088B_2022.xlsx` (rendered sheets) | 11 | 15.74 s |
| **Total** | **14** | **34.09 s** |

- Model: `vidore/colqwen2-base` (~8.8 GB total)
- Device: MPS via `device_map="auto"`
- Avg embed time per page: **2.43 s**
- Query embedding: ~0.1 s
- MaxSim search: <0.01 s
- Queries returned relevant XLSX sheets and PDF pages with scores ~0.80–0.82.

### Graph extraction (Phase 1)

- Extractor: `gemma4:latest` via Ollama structured JSON output
- Store: SQLite graph tables (`entities`, `relations`)
- On 50 representative chunks: **380 entities**, **221 relations**; query latency <1 ms
- Sample extracted entities: `GIR/FPSOT/GAS/FLARE/41-DS-BDV2088B`, wall-thickness metrics, inspection codes (`01NF`, `02CO`, etc.)

---

## What was built

### 1. Graph memory layer (`src/local_rag_stack/graph/`)

- `graph/models.py` — typed `Entity`/`Relation` schema aligned with Karpathy/Anthropic vocabulary:
  - `EntityType`: person, organization, object, location, document, concept, metric, attribute, event
  - `RelationType`: mentions, part_of, located_in, has_attribute, has_metric, authored_by, references, contains, relates_to, occurred_at
- `graph/extractor.py` — `OllamaGraphExtractor` using Ollama JSON-schema `format`; handles Qwen3 reasoning output by reading the `thinking` field
- `graph/store.py` — `SQLiteGraphStore` with full-text search, type/relation filters, document scoping, and n-hop neighborhood expansion
- Wired into `RAGPipeline.ingest()` so every text chunk produces graph nodes/edges
- API endpoints: `POST /graph/query`, `POST /graph/neighborhood`
- CLI commands: `lrs graph-query`, `lrs graph-neighborhood`

### 2. ColQwen visual path fixes

- `src/local_rag_stack/embedding/visual_embedder.py`:
  - Changed `ColQwen2.from_pretrained(..., torch_dtype=_get_torch_dtype(), device_map=self.device)`
  - To `torch_dtype="auto", device_map="auto"`
  - This reduced model load time from **>1 hour (hang)** to **~11 seconds** on MPS
- `src/local_rag_stack/storage/multivec_store.py`:
  - Excluded binary `image_bytes` from JSON-lines persistence to avoid UTF-8 serialization errors
  - Visual store now persists multi-vector embeddings and metadata only

### 3. Robust extraction fallback

- `src/local_rag_stack/extraction/plaintext_extractor.py`:
  - Added PyMuPDF-based PDF text extraction
  - Added openpyxl-based XLSX text extraction
  - Falls back to plain-text read for `.txt`, `.md`, `.csv`, etc.
  - This makes the default `plaintext` engine usable without Docling/Marker

### 4. Resilient model downloader

- `scripts/resume_hf_download.py`:
  - Segmented HTTP Range-request downloader with per-chunk retry
  - sha256 verification
  - Used to finish the incomplete `vidore/colqwen2-base` shard after repeated HF/Xet timeouts

### 5. Tenant-aware API wrapper (`src/local_rag_stack/tenant.py`)

- `Tenant` / `TenantManager` maps `X-API-Key` or `Authorization: Bearer` headers to tenants.
- Per-tenant storage isolation:
  - SQLite vector DB: `data/tenants/<tenant-id>/lrs.vec.sqlite`
  - Graph DB: `data/tenants/<tenant-id>/lrs.graph.sqlite`
  - Multi-vector JSONL next to the tenant vector DB
  - Qdrant collection: `<base>-<tenant-id>`
- Embedders/generators/Ollama clients are reused; only stores are namespaced.
- All API endpoints (`/health`, `/ingest`, `/query`, `/graph/*`) are tenant-scoped.
- Auth can be disabled via `LRS_AUTH_REQUIRED=false` for single-tenant local use.

### 6. Benchmark scripts

- `scripts/text_rag_benchmark.py` — text-only end-to-end benchmark
- `scripts/colqwen_benchmark.py` — visual/ColQwen end-to-end benchmark
- `scripts/graph_benchmark.py` — graph extraction benchmark with `--max-chunks` cap
- `scripts/resume_hf_download.py` — resilient HF file downloader

### 6. Tests

- Added `tests/test_graph.py`:
  - `test_canonicalise_assigns_stable_ids`
  - `test_sqlite_graph_store_roundtrip`
- Full suite result: **18 passed**

---

## Files changed

```
src/local_rag_stack/config.py
src/local_rag_stack/models.py
src/local_rag_stack/pipeline.py
src/local_rag_stack/cli.py
src/local_rag_stack/api/routes.py
src/local_rag_stack/extraction/plaintext_extractor.py
src/local_rag_stack/embedding/visual_embedder.py
src/local_rag_stack/storage/multivec_store.py
src/local_rag_stack/graph/__init__.py          (new)
src/local_rag_stack/graph/models.py            (new)
src/local_rag_stack/graph/extractor.py         (new)
src/local_rag_stack/graph/store.py             (new)
src/local_rag_stack/tenant.py                  (new)
tests/test_tenant.py                           (new)
scripts/text_rag_benchmark.py                  (new)
scripts/colqwen_benchmark.py                   (new)
scripts/graph_benchmark.py                     (new)
scripts/resume_hf_download.py                  (new)
tests/test_graph.py                            (new)
docs/sprint-summary-2026-07-29.md              (this file)
```

---

## Configuration knobs added

| Env var / setting | Default | Purpose |
|---|---|---|
| `LRS_GRAPH_STORE_PATH` | `./data/lrs.graph.sqlite` | SQLite graph database |
| `LRS_GRAPH_EXTRACTOR_MODEL` | `gemma4:latest` | LLM for entity/relation extraction |
| `LRS_GRAPH_EXTRACT_ON_INGEST` | `true` | Run graph extraction during ingest |
| `LRS_VISION_EMBED_BACKEND` | `none` | Set to `hf` or `colqwen` to enable ColQwen |
| `LRS_VISION_EMBED_MODEL` | `qwen3-vl` | For ColQwen use `vidore/colqwen2-base` |

---

## Production readiness checklist

| Item | Status | Notes |
|---|---|---|
| Core text RAG | ✅ Ready | Fast, deterministic, no HF download dependency |
| Graph extraction Phase 1 | ✅ Ready | Optional; adds ~15–25 s per chunk on `gemma4:latest` |
| ColQwen visual RAG | ⚠️ Working, tune before scale | 5 GB model, ~2.4 s/page; verify memory on target hardware |
| API endpoints | ✅ Ready | `/ingest`, `/query`, `/graph/query`, `/graph/neighborhood` |
| CLI | ✅ Ready | `lrs ingest`, `lrs query`, `lrs graph-query`, etc. |
| Tests | ✅ 18 passed | |
| Observability | ⚠️ Partial | Add metrics/logging around extraction times, cache hits |
| Auth / multi-tenancy | ✅ Ready | API-key → tenant mapping; per-tenant stores |
| Tenant-aware API wrapper | ✅ Ready | `src/local_rag_stack/tenant.py` + API tests |
| GPU/MLX optimization | ❌ Not implemented | ColQwen on MLX could be much faster than MPS |
| Docling/Marker extraction | ⚠️ Optional | `plaintext` engine now sufficient for PDF/XLSX |
| Docker packaging | ✅ Ready | Dockerfile + docker-compose.yml for standalone HTTP service |

---

## Known limitations

1. **ColQwen model size**: ~8.8 GB on disk. First download requires a stable connection; the resilient downloader helps but is slow on poor links.
2. **Graph extraction speed**: ~15–25 s per 512-token chunk with `gemma4:latest`. Use `max_graph_chunks` to cap cost on large documents.
3. **Entity typing accuracy**: LLM-assigned types are best-effort (e.g., equipment sometimes tagged `concept` instead of `object`). Rule-based post-processing can improve this.
4. **XLSX visual rendering**: Uses matplotlib tables; high-density sheets may be truncated or hard to read. A higher-fidelity renderer would improve ColQwen results.
5. **MPS vs MLX**: ColQwen currently runs via PyTorch MPS. An MLX-native path would likely be faster and lower-memory on Apple Silicon.

---

## Next steps

### Immediate (this week)

1. **Decide integration boundary**: ✅ Standalone HTTP service (Option B) — `docker-compose.yml` provided.
2. **Auth and multi-tenancy**: ✅ API-key → tenant mapping with per-tenant stores (`src/local_rag_stack/tenant.py`).
3. **Pick the default stack per workload**:
   - Text-only for high-volume / low-cost perusal.
   - ColQwen for layout-heavy / visual documents where table structure matters.
4. **Add observability**: log ingest time, embed time, query time, cache hit/miss, model load time.

### Short term (next 2–4 weeks)

4. **Graph Phase 2 — Evaluator-optimizer loop** (in progress):
   - Added an `Experiment` abstraction around (extraction strategy, chunk size, embed model, prompt).
   - New modules: `src/local_rag_stack/evaluation/models.py`, `evaluator.py`, `store.py`.
   - New CLI commands: `lrs experiment run`, `lrs experiment list`, `lrs experiment show`, `lrs experiment decide`.
   - Sample eval set: `scripts/sample_eval_set.jsonl`.
   - SQLite-backed `ExperimentStore` keeps experiments, per-question `EvalResult`s, and keep/revert `Decision`s separate from the graph store.
5. **Graph Phase 3 — DAG of experiments**:
   - Branch experiments from prior commits.
   - Parallel fan-out of ingestion/retrieval variants.
6. **MLX ColQwen path**:
   - Evaluate `mlx-vlm` or an MLX-converted ColQwen model for faster Apple-Silicon inference.
7. **Auth and multi-tenancy**: ✅ Done — API-key → tenant mapping with per-tenant stores.

### Longer term

8. **Graph-grounded sub-agents** (Phase 4):
   - Sub-agents query the knowledge graph for context instead of receiving full transcripts.
   - Use graph neighborhood to route questions and cite provenance.
9. **Human-in-the-loop promotion points** before experiments are merged into the active pipeline.

---

## How to hand off to CaaS / decision-saas

The package is ready to be handed off as a **standalone tenant-aware HTTP service**. Recommended hand-off artifacts:

1. This repository (`Local-AI-Document-RAG-Stack`) at the current commit.
1. `HANDOFF.md` — the production hand-off runbook.
2. A one-page integration diagram showing:
   - CaaS ingestion service calls `POST /ingest` (or imports `RAGPipeline.ingest()`).
   - Decision-saas query service calls `POST /query` and/or `POST /graph/query`.
3. Environment template (`.env.example`) with `LRS_*` variables.
4. Docker packaging (`Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh`, `.dockerignore`).
5. Tenant / auth config:
   - `.env.example` with `LRS_AUTH_REQUIRED`, `LRS_TENANT_KEYS_JSON`, `LRS_TENANT_KEYS_FILE`.
   - `src/local_rag_stack/tenant.py` and tests.
6. Experiment / evaluator scaffolding (Phase 2):
   - `src/local_rag_stack/evaluation/*`
   - `lrs experiment run/list/show/decide`
   - `scripts/sample_eval_set.jsonl`
7. Operational runbook for:
   - Downloading the ColQwen model once to a shared cache.
   - Monitoring Ollama availability.
   - Tuning `max_graph_chunks` for cost vs. coverage.
   - Provisioning tenant API keys and mapping them to tenant IDs.

Before production, the receiving team should:
- Confirm target hardware (Apple Silicon / CUDA / CPU).
- Build a labeled eval set for their specific PDF/XLSX report types.
- Decide whether to run Ollama co-located or remote.
