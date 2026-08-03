# Local AI Document RAG Stack — Production Hand-off

## What this is

A **standalone, tenant-aware HTTP service** that ingests PDF, XLSX, and text documents, embeds them locally, and answers questions with citations. It is designed to sit beside CaaS / decision-saas as a black-box RAG appliance: other services call it over HTTP and never need to import Python code or share a runtime.

This hand-off covers the state of the repo **after** the sprint that added Docker packaging, the tenant-aware API wrapper, and Phase 2 evaluator-optimizer scaffolding.

---

## What is delivered

| Component | Location | Status |
|---|---|---|
| Core text RAG pipeline | `src/local_rag_stack/pipeline.py` | ✅ Ready |
| Graph extraction Phase 1 | `src/local_rag_stack/graph/` | ✅ Ready |
| ColQwen visual RAG | `src/local_rag_stack/embedding/visual_embedder.py` | ⚠️ Working, verify on target hardware |
| Standalone HTTP API | `src/local_rag_stack/api/` | ✅ Ready |
| Tenant auth + per-tenant stores | `src/local_rag_stack/tenant.py` | ✅ Ready |
| Docker packaging | `Dockerfile`, `docker-compose.yml` | ✅ Ready |
| Evaluator-optimizer Phase 2 | `src/local_rag_stack/evaluation/` | ✅ Scaffolding ready |
| Benchmark scripts | `scripts/*_benchmark.py` | ✅ Ready |
| Tests | `tests/` | ✅ 33 passed |
| Sprint summary | `docs/sprint-summary-2026-07-29.md` | ✅ Updated |
| Roadmap | `docs/agentic_loop_roadmap.md` | ✅ Updated |

---

## Quick start

### 1. Configure

```bash
cp .env.example .env
```

At minimum set:

```env
LRS_OLLAMA_HOST=http://ollama:11434
LRS_AUTH_REQUIRED=true
LRS_TENANT_KEYS_JSON='{"tenant-a-key":"tenant-a","tenant-b-key":"tenant-b"}'
```

Or use a JSON file:

```bash
mkdir -p config
cp config/tenant_keys.example.json config/tenant_keys.json
# edit config/tenant_keys.json
```

```env
LRS_AUTH_REQUIRED=true
LRS_TENANT_KEYS_FILE=./config/tenant_keys.json
```

### 2. Run with Docker Compose

```bash
docker compose up --build

# Pull the default text models into the co-located Ollama container
docker exec -it lrs-ollama ollama pull qwen3:8b
docker exec -it lrs-ollama ollama pull qwen3-embedding:4b
```

Smoke test:

```bash
curl -H "X-API-Key: tenant-a-key" http://localhost:8000/health

curl -H "X-API-Key: tenant-a-key" \
  -F file=@sample.pdf \
  http://localhost:8000/ingest

curl -H "X-API-Key: tenant-a-key" \
  -X POST -H "Content-Type: application/json" \
  -d '{"query":"What is the effective date?"}' \
  http://localhost:8000/query
```

### 3. Run experiments

```bash
# Inside the container or a local venv
lrs experiment run \
  --eval-set scripts/sample_eval_set.jsonl \
  --name "text-only baseline"

lrs experiment list
lrs experiment show <experiment_id>
lrs experiment decide <experiment_id> --outcome keep --reason "beats baseline on p@k"
```

---

## Integration with CaaS / decision-saas

Recommended boundaries:

- **CaaS ingestion service** → `POST /ingest` (multipart upload) with the tenant's API key.
- **Decision-saas query service** → `POST /query` and/or `POST /graph/query` with the tenant's API key.
- **Document parser fallback** → `POST /extract` returns markdown + chunks without indexing.

The receiving team does **not** need to import `local_rag_stack` as a library. They only need HTTP clients and API keys.

---

## Tenant isolation

Each tenant gets its own storage namespace:

| Store | Tenant path |
|---|---|
| sqlite-vec | `data/tenants/<tenant-id>/lrs.vec.sqlite` |
| Graph | `data/tenants/<tenant-id>/lrs.graph.sqlite` |
| Multi-vector JSONL | next to tenant sqlite DB |
| Qdrant | collection `<base>-<tenant-id>` |

Model objects (embedders, generators, Ollama client) are reused across tenants to avoid loading ColQwen multiple times.

---

## Production checklist

- [ ] Confirm target hardware: Apple Silicon / CUDA / CPU.
- [ ] Decide Ollama location: co-located in compose, host daemon, or remote cluster.
- [ ] Provision tenant API keys and map them to tenant IDs.
- [ ] Enable `LRS_AUTH_REQUIRED=true` in production.
- [ ] If using Qdrant, set `LRS_QDRANT_URL` and per-tenant collection policy.
- [ ] Build a labeled eval set from real PDF/XLSX reports to tune retrieval.
- [ ] If using ColQwen, verify `device_map="auto"` behavior on the target GPU/MLX.
- [ ] Add observability: ingest/query/embed latency, cache hits, model load time.
- [ ] Add backups for `data/tenants/` and `data/lrs.experiments.sqlite`.
- [ ] Decide on auth source: env JSON, mounted key file, or an external identity service.

---

## Known limitations

1. **ColQwen model size**: ~8.8 GB on disk. First download needs a stable connection.
2. **Graph extraction speed**: ~15–25 s per 512-token chunk with `gemma4:latest`. Use `max_graph_chunks` to cap cost.
3. **Entity typing**: LLM-assigned types are best-effort; add rule-based post-processing if needed.
4. **Chunk-size strategy**: `StrategyConfig.chunk_size` is recorded by experiments but is not yet wired into extractors.
5. **MLX path**: Not implemented; ColQwen currently uses PyTorch MPS on Apple Silicon.

---

## Next steps for the receiving team

1. **Hardware validation** — run `scripts/text_rag_benchmark.py` and `scripts/colqwen_benchmark.py` on the target appliance.
2. **Labeled eval set** — replace `scripts/sample_eval_set.jsonl` with real Q&A over your reports.
3. **Tune retrieval** — use `lrs experiment run` to compare text-only vs. ColQwen vs. hybrid configs.
4. **Auth integration** — swap the inline key file for your existing API-key/tenant service if needed.
5. **Observability** — add structured logging / metrics around the pipeline lifecycle.
6. **Phase 3** — branch experiments from prior commits and merge winning strategies into the active config.
