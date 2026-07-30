# Agentic Graph Engineering — Future Sprint Roadmap

This document captures how Andrej Karpathy's agentic-loop / graph-engineering ideas (autoresearch → AgentHub → Anthropic Dynamic Workflows / Knowledge Graph Cookbook) can be layered on top of the Local RAG Stack.

## Goal

Turn the current one-shot RAG pipeline into a persistent, graph-grounded agent swarm that can:

1. Propose document-reading and extraction strategies.
2. Run them (ingest, embed, retrieve, evaluate).
3. Keep or revert the result based on measured outcomes.
4. Store structured facts, provenance, and decisions in a knowledge graph.
5. Allow sub-agents to query that graph instead of re-reading entire documents.

## Schema (Fig. 1)

Central nodes:

- `Agent` — the orchestrator or worker that performed work.
- `Document` / `TextSummary` — extracted chunks and summaries.
- `Entity` (Person, Object, Agency, Tool, Property, Metric) — extracted typed nodes.
- `Relation` — typed edges between entities and documents.
- `Experiment` / `Commit` — an attempted change with metrics and a parent.
- `Decision` — keep / revert / promote outcome.

## Staged build path

### Phase 1 — Structured extraction loop (1–2 weeks) ✅ DONE

- ✅ Added Pydantic schemas for entity/relation extraction (`src/local_rag_stack/graph/models.py`).
- ✅ Implemented `OllamaGraphExtractor` using Ollama JSON-schema structured outputs (`src/local_rag_stack/graph/extractor.py`).
- ✅ Built `SQLiteGraphStore` for typed entities/relations (`src/local_rag_stack/graph/store.py`).
- ✅ Wired extraction into `RAGPipeline.ingest()` with `max_graph_chunks` cost control.
- ✅ Exposed `POST /graph/query` and `POST /graph/neighborhood` endpoints.
- ✅ Added `lrs graph-query` and `lrs graph-neighborhood` CLI commands.
- ✅ Added `tests/test_graph.py` with canonicalisation and roundtrip tests.

### Phase 2 — Evaluator-optimizer loop (1–2 weeks)

- Add an `Experiment` abstraction that wraps (ingest strategy, query strategy, prompt).
- Run a small eval set, measure precision@K and answer correctness.
- Use the result to accept/reject the experiment and update a `Decision` node.

### Phase 3 — DAG of experiments (2–3 weeks)

- Branch experiments from prior `Commit` nodes.
- Allow parallel fan-out of ingestion/retrieval variants.
- Merge successful branches into the active pipeline config.

### Phase 4 — Graph-grounded sub-agents (2–4 weeks)

- Sub-agents query the knowledge graph for context instead of receiving full transcripts.
- Use the graph to route questions to the right documents, pages, or prior decisions.
- Human-in-the-loop review at promotion points.

## Integration points with current package

- `RAGPipeline.ingest()` becomes one operation an agent can invoke.
- `RAGPipeline.query()` becomes the answer operation.
- New `GraphStore` plugin sits alongside `VectorStore` and `MultiVectorStore`.
- New `AgentLoop` orchestrator wraps the pipeline in propose/run/evaluate/decide cycles.

## Open questions to resolve before Phase 1

1. Graph database: Neo4j, Kùzu, or SQLite graph tables?
2. Entity/relation schema: domain-specific (contracts, invoices, reports) or generic?
3. Evaluation data: do we have labeled Q&A for customer PDF/XLSX reports?
4. Human review UI: CLI only, or web dashboard?
