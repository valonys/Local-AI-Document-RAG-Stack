# Local AI Document RAG Stack — Roadmap

> Hardware target: Apple M5 Pro (arm64) with 48 GB unified memory.  
> Deployment model: on-premise appliance or developer-owned Mac accessed via Tailscale for early sprints.  
> Last updated: 2026-07-15

---

## 1. Goal

Build a local, framework-agnostic stack that converts business documents (PDFs, scans, images, spreadsheets) into structured, searchable, and answerable data without sending data to cloud APIs.

The stack should support:

- Text, table, and layout extraction.
- Optional OCR for scanned documents.
- Semantic / multimodal search over documents.
- Reranking for precision.
- Question-answering with citations.
- Remote access for customer demos and sprints.

---

## 2. Why This Hardware

The M5 Pro 48 GB Mac is well-suited because:

- **Unified memory** lets CPU/GPU/NPU share 48 GB, removing the VRAM ceiling of discrete GPUs.
- **MLX** provides native Apple Silicon acceleration for LLMs, VLMs, embeddings, and rerankers.
- A 30–35B parameter model (or several smaller specialist models) can run simultaneously.
- Low power, quiet, and portable — good for customer-site appliances.

Estimated headroom:

| Model type | Typical memory | Fits? |
|---|---|---|
| 7–8B dense LLM / VLM | 8–15 GB | Yes |
| 30–35B MoE LLM | 15–25 GB | Yes |
| Qwen3-VL Embedding/Reranker 2B | 2–4 GB | Easily |
| ColQwen2 2B | 2–4 GB | Easily |
| Docling / Marker pipeline | 2–8 GB | Yes |

---

## 3. Architecture Decision: Two Paths

### Path A: Extract-Then-Search (Traditional RAG)

```
PDF / Image → Document Parser → Markdown/JSON chunks
                                   ↓
                         Text embeddings (Qwen3-Embedding)
                                   ↓
                         Vector DB + BM25 hybrid search
                                   ↓
                         Reranker (Qwen3-Reranker)
                                   ↓
                         LLM answer (Qwen3 / Gemma)
```

**Best when:** customers need exact text export, editable tables, or downstream processing of extracted content.

**Tools:**

- **Docling** — broad format support, layout-aware, tables, formulas, VLM pipeline via MLX.
- **Marker** — faster, cleaner markdown, good tables, lower hallucination risk.
- **PaddleOCR-VL** — strong multilingual OCR and layout detection.
- **Apple Vision / Tesseract** — lightweight OCR fallback.

### Path B: Visual Document Retrieval (Multimodal RAG)

```
PDF → Render pages as images → ColQwen2 / Qwen3-VL Embedding
                                          ↓
                         Late-interaction multi-vector index
                                          ↓
                         Qwen3-VL Reranker
                                          ↓
                         VLM answer over retrieved page images
```

**Best when:** the question is "which page/figure/section answers this?" rather than "give me the exact text."

**Advantages:**

- No brittle OCR pipeline.
- Tables, charts, signatures, stamps, handwriting, and layout are first-class.
- Retrieval returns exact page images with coordinates.

**Disadvantages:**

- Higher storage cost per page (~150–500 KB multi-vector vs ~5–20 KB text).
- Poor at producing clean exportable text/tables.

### Recommended Hybrid

Use **Path A for extraction/export** and **Path B for retrieval/Q&A**:

```
Document
   ├─→ Docling / Marker → structured markdown/JSON/text chunks
   │                        └─→ text embeddings + BM25
   └─→ Render pages → ColQwen2/Qwen3-VL embeddings
                        └─→ visual retrieval

At query time:
   1. Hybrid retrieve from both indexes.
   2. Fuse with RRF.
   3. Rerank top-K with Qwen3-VL-Reranker.
   4. Send best page images + text chunks to VLM for answer.
```

---

## 4. Component Choices

### 4.1 Document Extraction

| Tool | Strengths | Weaknesses | License |
|---|---|---|---|
| **Docling** | Many formats, layout, tables, formulas, VLM pipeline, MLX support | Slower VLM pipeline, dense tables can hallucinate | MIT |
| **Marker** | Fast, clean markdown, good tables, lower hallucination | GPL-3, less format breadth | GPL-3 |
| **PaddleOCR-VL** | Strong Chinese/multilingual, layout + OCR | Slower on CPU, dependency-heavy | Apache 2.0 |
| **MinerU + MLX** | GPU-accelerated on Apple Silicon, good OCR | AGPL v3, newer tool | AGPL v3 |

### 4.2 Large Language / Vision Models

| Model | Role | Size | Served via |
|---|---|---|---|
| `qwen3:8b` / `qwen3:14b` | General reasoning, extraction | 8–14B | Ollama, vllm-mlx |
| `qwen3-vl` | Document Q&A over retrieved images | 4–8B | Ollama, mlx-vlm, vllm-mlx |
| `gemma3` | Alternative VLM / chat | 4–12B | Ollama |

### 4.3 Embeddings

| Model | Type | Best for | Served via |
|---|---|---|---|
| `qwen3-embedding:0.6b` | Text | Fast, edge, 32K context | Ollama |
| `qwen3-embedding:4b` | Text | Quality/speed sweet spot | Ollama, mlx-embeddings |
| `qwen3-embedding:8b` | Text | Best quality, 40K context | Ollama, mlx-embeddings |
| `Qwen3-VL-Embedding-2B` | Multimodal | Text + image + video | mlx-embeddings, HF transformers |
| `ColQwen2` / `ColQwen2.5` | Late-interaction page patches | Visual document retrieval | colpali-engine, HF transformers |

### 4.4 Rerankers

| Model | Type | Notes |
|---|---|---|
| `Qwen3-Reranker-0.6B/4B/8B` | Text cross-encoder | Pairs with Qwen3-Embedding |
| `Qwen3-VL-Reranker-2B` | Multimodal | Reranks text-image pairs |
| `bge-reranker-v2-m3` | Text | Strong baseline |

**Important:** Ollama does **not** natively expose a `/rerank` endpoint. Use `mlx-embeddings`, `vllm-mlx`, or `sentence-transformers` for reranking.

### 4.5 Vector Stores

| Store | Best for |
|---|---|
| **sqlite-vec** | Zero-dependency, single-file, good for appliances |
| **Qdrant** | Multi-vector / late-interaction support, scalable |
| **Chroma** | Easy Python integration |

---

## 5. Networking & Deployment

### 5.1 Recommended Customer Deployment

For customer sprints, do **not** use a personal laptop over Tailscale as production infrastructure. Instead:

1. **Dedicated Mac Mini M5 Pro 48–64 GB** left on customer premises.
2. **Single appliance image** (Docker / native) installed and tested before delivery.
3. **Customer apps connect over local network** to the appliance.
4. **Tailscale used only for your remote admin access**, not customer traffic.

### 5.2 Tailscale Security (if used for access)

| Do | Don't |
|---|---|
| Use `tailscale serve --bg http://localhost:11434` for HTTPS ingress | Bind Ollama to `0.0.0.0` |
| Restrict ACLs to specific users/devices | Leave tailnet open to all devices |
| Add host firewall rules allowing Ollama port only on `tailscale0` | Expose Ollama to the public internet |
| Use MagicDNS hostnames instead of hard-coded IPs | Rely on Tailscale alone for authentication |

### 5.3 Service Layout on Appliance

```
Mac Mini M5 Pro
├── Ollama (localhost:11434)
│   ├── qwen3:8b or qwen3:14b
│   ├── qwen3-vl
│   └── qwen3-embedding:4b
├── MLX-native service (Python/FastAPI)
│   ├── ColQwen2 / Qwen3-VL-Embedding
│   └── Qwen3-VL-Reranker-2B
├── Optional extraction service
│   └── Docling or Marker
├── Vector store
│   └── sqlite-vec or Qdrant
└── Orchestrator API (FastAPI)
    └── Single customer-facing endpoint
```

---

## 6. Implementation Phases

### Phase 0 — Validate on Your Mac (1–2 weeks)

- Install Ollama, Docling, Marker, `mlx-embeddings`, `colpali-engine`.
- Test extraction quality on 20–50 representative customer documents.
- Benchmark latency and memory for one model at a time.
- Decide: Path A, Path B, or hybrid.

### Phase 1 — Local Pipeline (2–3 weeks)

- Build orchestrator in Python/FastAPI.
- Wire extraction → embedding → storage → retrieval → rerank → answer.
- Add basic evaluation: precision@K, answer correctness on a held-out set.

### Phase 2 — Appliance Packaging (1–2 weeks)

- Dockerize or script the full install.
- Test clean install on a fresh Mac.
- Add health checks, logs, and simple restart logic.

### Phase 3 — Customer Pilot (2–4 weeks)

- Deploy appliance on customer network.
- Collect feedback and failure cases.
- Iterate on chunking, retrieval, and prompt templates.

### Phase 4 — Hardening (ongoing)

- Auth, audit logging, backups, update mechanism.
- Optional: cloud fallback for models that do not fit locally.

---

## 7. Key Open Questions to Resolve

1. Do customers need **exportable structured text/tables**, or is **page-level Q&A** enough?
2. What is the document mix — digital PDFs, scans, photos, spreadsheets, slides?
3. Which languages and scripts must be supported?
4. Is handwriting or stamped/signed content common?
5. What is the acceptable query latency? (seconds per question)
6. Will the appliance be air-gapped, or can it reach the internet for model downloads?
7. What is the data retention / compliance requirement?

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Single Mac becomes a point of failure | Use dedicated appliance; plan backup/restore; consider two units |
| Tailscale exposes customer data to your laptop | Use Tailscale only for admin; customer traffic stays local |
| Ollama misconfiguration leaks API | Bind to localhost; use Tailscale Serve; firewall tailscale0 only |
| Qwen3-VL-Reranker unsupported in some MLX servers | Use `mlx-embeddings` directly until server support stabilizes |
| Dense tables hallucinate in extraction | Add visual retrieval fallback or human review queue |
| Model licensing for customer use | Review Qwen, ColQwen, Docling, Marker licenses before distribution |

---

## 9. Reference Links

- Docling: https://github.com/docling-project/docling
- Marker: https://github.com/datalab-to/marker
- Qwen3-VL Embedding & Reranker tutorial: https://learnopencv.com/how-to-master-qwen3-vl-embedding-and-reranker-for-multimodal-search/
- Qwen3-VL Embedding & Reranker code: https://github.com/spmallick/learnopencv/tree/master/Qwen3-VL-Embedding-Reranker
- ColPali / ColQwen: https://github.com/illuin-tech/colpali
- mlx-embeddings: https://github.com/Blaizzy/mlx-embeddings
- vllm-mlx: https://github.com/waybarrios/vllm-mlx
- Ollama: https://ollama.com
- Tailscale: https://tailscale.com

---

## 10. Suggested First Week Action

1. Install Ollama and pull `qwen3:8b`, `qwen3-embedding:4b`, and `qwen3-vl`.
2. Install Docling and Marker in separate virtual environments.
3. Run ColQwen2 on 10 sample PDFs using `colpali-engine` with `device_map="mps"`.
4. Compare three retrieval answers:
   - Text-only RAG (Docling + Qwen3-Embedding).
   - Visual-only RAG (ColQwen2 + Qwen3-VL).
   - Hybrid (both + Qwen3-VL-Reranker).
5. Document which pipeline wins on the customer's actual documents.
