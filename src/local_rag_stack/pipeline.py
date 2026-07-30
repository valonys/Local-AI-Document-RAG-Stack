"""High-level ingest and query orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Settings
from .embedding.base import (
    EmbeddingPipeline,
    MultiVectorEmbedder,
    TextEmbedder,
    VisualEmbedder,
)
from .embedding.text_embedder import OllamaTextEmbedder, SentenceTransformersEmbedder
from .embedding.visual_embedder import ColQwenEmbedder, OllamaVisualEmbedder
from .exceptions import ConfigurationError
from .extraction.base import DocumentExtractor
from .extraction.docling_extractor import DoclingExtractor
from .extraction.marker_extractor import MarkerExtractor
from .extraction.plaintext_extractor import PlainTextExtractor
from .generation.generator import AnswerGenerator
from .graph.extractor import GraphExtractor, OllamaGraphExtractor
from .graph.models import GraphExtractionResult, GraphSearchResult
from .graph.store import GraphStore, SQLiteGraphStore
from .models import (
    DocumentSummary,
    ExtractedDocument,
    GraphNeighborhoodRequest,
    GraphQueryRequest,
    GraphQueryResponse,
    HealthResponse,
    QueryResponse,
)
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import NoOpReranker, OllamaReranker, Reranker, SentenceTransformersReranker
from .storage.base import VectorStore
from .storage.multivec_store import MultiVectorStore
from .storage.qdrant_store import QdrantStore
from .storage.sqlite_vec_store import SQLiteVecStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end local RAG pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()  # type: ignore[call-arg]
        self.settings.ensure_paths()

        self.extractor = self._build_extractor()
        self.text_embedder = self._build_text_embedder()
        self.visual_embedder: VisualEmbedder | None = None
        self.multivec_embedder: MultiVectorEmbedder | None = None
        self._build_visual_embedders()
        self.embedding_pipeline = EmbeddingPipeline(
            self.text_embedder,
            self.visual_embedder,
            self.multivec_embedder,
        )
        self.store = self._build_store()
        self.multivec_store = self._build_multivec_store()
        self.graph_store = self._build_graph_store()
        self.graph_extractor = self._build_graph_extractor()
        self.retriever = HybridRetriever(
            self.store,
            multivec_store=self.multivec_store,
            text_search_k=self.settings.text_search_k,
            visual_search_k=self.settings.visual_search_k,
            multivec_search_k=self.settings.visual_search_k,
            rrf_k=self.settings.rrf_k,
        )
        self.reranker = self._build_reranker()
        self.generator = self._build_generator()

    def _build_extractor(self) -> DocumentExtractor:
        engine = self.settings.extraction_engine.lower()
        output_dir = self.settings.data_dir / "pages"
        if engine == "docling":
            return DoclingExtractor(dpi=self.settings.extraction_dpi, output_dir=output_dir)
        if engine == "marker":
            return MarkerExtractor(dpi=self.settings.extraction_dpi, output_dir=output_dir)
        if engine == "plaintext":
            return PlainTextExtractor(dpi=self.settings.extraction_dpi, output_dir=output_dir)
        raise ConfigurationError(f"Unknown extraction engine: {engine}")

    def _select_extractor_for_file(self, file_path: Path) -> DocumentExtractor:
        """Auto-select plaintext extractor for text files when engine is docling."""
        suffix = file_path.suffix.lower()
        engine = self.settings.extraction_engine.lower()
        if engine == "docling" and suffix in PlainTextExtractor.SUPPORTED_SUFFIXES:
            output_dir = self.settings.data_dir / "pages"
            return PlainTextExtractor(dpi=self.settings.extraction_dpi, output_dir=output_dir)
        return self.extractor

    def _build_text_embedder(self) -> TextEmbedder:
        # Prefer Ollama if configured with a qwen3 model, otherwise local ST.
        if self.settings.text_embed_model.startswith("qwen3") or ":" in self.settings.text_embed_model:
            return OllamaTextEmbedder(
                model=self.settings.text_embed_model,
                host=self.settings.ollama_host,
                timeout=self.settings.ollama_timeout,
            )
        return SentenceTransformersEmbedder(model=self.settings.text_embed_model)

    def _build_visual_embedders(self) -> None:
        backend = self.settings.vision_embed_backend.lower()
        if backend == "none" or not self.settings.vision_embed_model:
            return
        if backend == "ollama":
            self.visual_embedder = OllamaVisualEmbedder(
                model=self.settings.vision_embed_model,
                host=self.settings.ollama_host,
                timeout=self.settings.ollama_timeout,
            )
        elif backend in ("hf", "transformers", "colqwen"):
            self.multivec_embedder = ColQwenEmbedder(model=self.settings.vision_embed_model)
        elif backend == "mlx":
            raise ConfigurationError(
                "MLX visual embedder not yet implemented; use 'ollama' or 'hf'"
            )
        else:
            raise ConfigurationError(f"Unknown visual embed backend: {backend}")

    def _build_store(self) -> VectorStore:
        backend = self.settings.vector_store.lower()
        if backend == "sqlite":
            return SQLiteVecStore(self.settings.sqlite_vec_path)
        if backend == "qdrant":
            return QdrantStore(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                collection=self.settings.qdrant_collection,
            )
        raise ConfigurationError(f"Unknown vector store: {backend}")

    def _build_multivec_store(self) -> MultiVectorStore | None:
        if self.multivec_embedder is None:
            return None
        path = self.settings.sqlite_vec_path.with_suffix(".multivec.jsonl")
        return MultiVectorStore(path)

    def _build_graph_store(self) -> GraphStore | None:
        if not self.settings.graph_store_path:
            return None
        return SQLiteGraphStore(self.settings.graph_store_path)

    def _build_graph_extractor(self) -> GraphExtractor | None:
        if not self.settings.graph_extract_on_ingest or self.graph_store is None:
            return None
        return OllamaGraphExtractor(
            model=self.settings.graph_extractor_model,
            host=self.settings.ollama_host,
            timeout=self.settings.ollama_timeout,
        )

    def _build_reranker(self) -> Reranker:
        backend = self.settings.rerank_backend.lower()
        if backend == "none":
            return NoOpReranker()
        if backend == "ollama":
            return OllamaReranker(
                model=self.settings.rerank_model,
                host=self.settings.ollama_host,
                timeout=self.settings.ollama_timeout,
            )
        if backend in ("st", "sentence_transformers"):
            return SentenceTransformersReranker(model=self.settings.rerank_model)
        raise ConfigurationError(f"Unknown rerank backend: {backend}")

    def _build_generator(self) -> AnswerGenerator:
        return AnswerGenerator(
            model=self.settings.llm_model,
            host=self.settings.ollama_host,
            timeout=self.settings.ollama_timeout,
        )

    def ingest(
        self,
        file_path: Path | str,
        *,
        document_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_graph_chunks: int | None = None,
    ) -> ExtractedDocument:
        """Ingest a document: extract, embed, and store."""
        file_path = Path(file_path)
        logger.info("Ingesting %s", file_path)
        extractor = self._select_extractor_for_file(file_path)
        doc = extractor.extract(
            file_path,
            document_name=document_name,
            metadata=metadata or {},
        )
        dense_items, multi_items = self.embedding_pipeline.embed(doc)
        if dense_items:
            self.store.add(dense_items)
        if multi_items and self.multivec_store is not None:
            self.multivec_store.add(multi_items)

        if self.graph_extractor is not None and self.graph_store is not None:
            self._extract_and_store_graph(doc, max_chunks=max_graph_chunks)

        logger.info(
            "Ingested %s: %d chunks, %d pages (%d multi-vector pages)",
            doc.document_id,
            len(doc.chunks),
            len(doc.pages),
            len(multi_items),
        )
        return doc

    def _extract_and_store_graph(
        self, doc: ExtractedDocument, max_chunks: int | None = None
    ) -> GraphExtractionResult:
        """Run entity/relation extraction over text chunks and store the graph."""
        all_entities: list[Any] = []
        all_relations: list[Any] = []
        chunks = doc.chunks
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
            logger.info("Limiting graph extraction to first %d chunks", len(chunks))
        for chunk in chunks:
            result = self.graph_extractor.extract(
                chunk.text,
                document_id=doc.document_id,
                document_name=doc.document_name,
                chunk_id=chunk.chunk_id,
            )
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)
        self.graph_store.add(all_entities, all_relations)
        logger.info(
            "Graph extracted for %s: %d entities, %d relations",
            doc.document_id,
            len(all_entities),
            len(all_relations),
        )
        return GraphExtractionResult(entities=all_entities, relations=all_relations)

    def query(
        self,
        query: str,
        *,
        top_k: int | None = None,
        include_images: bool = True,
    ) -> QueryResponse:
        """Ask a question over all ingested documents."""
        text_vector = self.text_embedder.embed_query(query)
        visual_vector = None
        if self.visual_embedder is not None:
            visual_vector = self.visual_embedder.embed_query(query)
        multivec_query_vectors = None
        if self.multivec_embedder is not None:
            multivec_query_vectors = self.multivec_embedder.embed_query(query)

        retrieved = self.retriever.retrieve(
            text_vector,
            visual_vector,
            multivec_query_vectors,
            top_k=self.settings.rerank_top_k * 2,
        )
        reranked = self.reranker.rerank(
            query,
            retrieved,
            top_k=top_k or self.settings.final_top_k,
        )
        return self.generator.generate(query, reranked, include_images=include_images)

    def list_documents(self) -> list[DocumentSummary]:
        dense_docs = self.store.list_documents()
        multi_docs = self.multivec_store.list_documents() if self.multivec_store else {}
        all_ids = set(dense_docs.keys()) | set(multi_docs.keys())
        summaries: list[DocumentSummary] = []
        for doc_id in all_ids:
            dense = dense_docs.get(doc_id, {})
            multi = multi_docs.get(doc_id, {})
            summaries.append(
                DocumentSummary(
                    document_id=doc_id,
                    document_name=dense.get("document_name") or multi.get("document_name", ""),
                    chunk_count=dense.get("chunk_count", 0),
                    page_count=(dense.get("page_count", 0) + multi.get("page_count", 0)),
                    ingested_at=None,
                )
            )
        return summaries

    def delete_document(self, document_id: str) -> None:
        self.store.delete_document(document_id)
        if self.multivec_store is not None:
            self.multivec_store.delete_document(document_id)
        if self.graph_store is not None:
            self.graph_store.delete_document(document_id)

    def graph_query(self, request: GraphQueryRequest) -> GraphQueryResponse:
        if self.graph_store is None:
            return GraphQueryResponse(query=request.query or "")
        from .graph.models import EntityType, RelationType

        entity_types = None
        if request.entity_types:
            entity_types = [EntityType(t) for t in request.entity_types if t]
        relation_type = None
        if request.relation_type:
            relation_type = RelationType(request.relation_type)

        result = self.graph_store.search(
            query=request.query,
            entity_types=entity_types,
            document_id=request.document_id,
            entity_name=request.entity_name,
            relation_type=relation_type,
            top_k=request.top_k,
        )
        return self._graph_result_to_response(request.query or "", result)

    def graph_neighborhood(
        self, request: GraphNeighborhoodRequest
    ) -> GraphQueryResponse:
        if self.graph_store is None:
            return GraphQueryResponse()
        result = self.graph_store.get_neighborhood(request.entity_id, hops=request.hops)
        return self._graph_result_to_response(f"neighborhood:{request.entity_id}", result)

    @staticmethod
    def _graph_result_to_response(query: str, result: GraphSearchResult) -> GraphQueryResponse:
        from .models import GraphEntityResponse, GraphRelationResponse

        return GraphQueryResponse(
            query=query,
            entities=[
                GraphEntityResponse(
                    id=e.id,
                    name=e.name,
                    entity_type=e.entity_type.value,
                    document_id=e.document_id,
                    chunk_id=e.chunk_id,
                    source_text=e.source_text,
                )
                for e in result.entities
            ],
            relations=[
                GraphRelationResponse(
                    id=r.id,
                    source_entity_id=r.source_entity_id,
                    target_entity_id=r.target_entity_id,
                    relation_type=r.relation_type.value,
                    document_id=r.document_id,
                    chunk_id=r.chunk_id,
                    source_text=r.source_text,
                )
                for r in result.relations
            ],
        )

    def health(self) -> HealthResponse:
        ollama_ready = self._ollama_ready()
        return HealthResponse(
            status="ok" if ollama_ready else "degraded",
            ollama_ready=ollama_ready,
            vector_store=self.settings.vector_store,
            graph_store=str(self.settings.graph_store_path) if self.graph_store else None,
            models={
                "text_embed": self.settings.text_embed_model,
                "vision_embed": self.settings.vision_embed_model,
                "rerank": self.settings.rerank_model,
                "llm": self.settings.llm_model,
                "graph_extractor": self.settings.graph_extractor_model,
            },
        )

    def _ollama_ready(self) -> bool:
        import httpx
        try:
            httpx.get(f"{self.settings.ollama_host}/", timeout=5.0)
            return True
        except Exception:
            return False

    def close(self) -> None:
        self.store.close()
        if self.multivec_store is not None:
            self.multivec_store.close()
        if self.graph_store is not None:
            self.graph_store.close()
