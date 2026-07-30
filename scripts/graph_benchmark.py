#!/usr/bin/env python3
"""End-to-end benchmark for the Karpathy-style graph extraction layer.

Ingests the same PDF and XLSX files as the other benchmarks, extracts
entities/relations from every text chunk using a structured-output LLM call,
stores them in the SQLite graph store, and runs sample graph queries.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from local_rag_stack.config import Settings
from local_rag_stack.models import GraphQueryRequest
from local_rag_stack.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph extraction benchmark")
    parser.add_argument("files", nargs="+", type=Path, help="PDF/XLSX files to ingest")
    parser.add_argument("--data-dir", default="./data/benchmark", help="Data directory")
    parser.add_argument("--text-embed-model", default="BAAI/bge-small-en-v1.5", help="Text embedder")
    parser.add_argument("--llm", default="gemma4:latest", help="Graph extraction LLM")
    parser.add_argument("--device", default="mps", help="torch device")
    parser.add_argument("--no-graph", action="store_true", help="Skip graph extraction")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit number of text chunks to graph-extract")
    args = parser.parse_args()

    settings = Settings(
        data_dir=Path(args.data_dir),
        sqlite_vec_path=Path(args.data_dir) / "graph_benchmark.vec.sqlite",
        graph_store_path=Path(args.data_dir) / "graph_benchmark.graph.sqlite",
        text_embed_model=args.text_embed_model,
        graph_extractor_model=args.llm,
        graph_extract_on_ingest=not args.no_graph,
        extraction_engine="plaintext",
    )  # type: ignore[call-arg]

    pipeline = RAGPipeline(settings)
    try:
        start = time.perf_counter()
        for file_path in args.files:
            logger.info("Ingesting %s", file_path)
            pipeline.ingest(file_path, max_graph_chunks=args.max_chunks)
        ingest_elapsed = time.perf_counter() - start

        health = pipeline.health()
        logger.info("Health: %s", health.model_dump_json(indent=2))

        if pipeline.graph_store is None:
            logger.info("Graph store disabled")
            return

        # Summary counts
        result = pipeline.graph_query(GraphQueryRequest(top_k=10000))
        logger.info(
            "Graph summary: %d entities, %d relations",
            len(result.entities),
            len(result.relations),
        )

        # Sample queries
        queries = [
            GraphQueryRequest(query="BDV 2088", top_k=10),
            GraphQueryRequest(entity_types=["metric"], top_k=10),
            GraphQueryRequest(entity_types=["object"], top_k=10),
            GraphQueryRequest(relation_type="has_metric", top_k=10),
        ]
        for q in queries:
            start = time.perf_counter()
            r = pipeline.graph_query(q)
            elapsed = time.perf_counter() - start
            logger.info(
                "Graph query %s -> %d entities, %d relations in %.3fs",
                q.model_dump(exclude_none=True),
                len(r.entities),
                len(r.relations),
                elapsed,
            )
            for entity in r.entities[:5]:
                logger.info("  [%s] %s", entity.entity_type, entity.name)

        logger.info("Total ingest + graph time: %.2fs", ingest_elapsed)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
