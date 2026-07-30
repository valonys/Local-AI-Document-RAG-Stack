"""Tests for core pipeline abstractions that do not require model downloads."""

from __future__ import annotations

import pytest

from local_rag_stack.config import Settings
from local_rag_stack.extraction.base import DocumentExtractor
from local_rag_stack.models import BoundingBox, Modality, SearchResult, TextChunk
from local_rag_stack.retrieval.hybrid import HybridRetriever
from local_rag_stack.retrieval.reranker import NoOpReranker


def test_settings_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LRS_API_PORT", "9000")
    settings = Settings()  # type: ignore[call-arg]
    assert settings.api_port == 9000


def test_document_extractor_split_text() -> None:
    class DummyExtractor(DocumentExtractor):
        def extract(self, file_path, *, document_id=None, document_name=None, metadata=None):
            raise NotImplementedError

    extractor = DummyExtractor()
    text = " ".join(["word"] * 100)
    chunks = extractor.split_text(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) > 0
    assert all(len(c) <= 100 for c in chunks)


def test_text_chunk_model() -> None:
    chunk = TextChunk(
        chunk_id="doc_chunk_00000",
        document_id="doc",
        document_name="doc.pdf",
        page_number=1,
        section_heading="heading",
        text="hello world",
        bbox=BoundingBox(x=0, y=0, width=10, height=10),
    )
    assert chunk.bbox is not None
    assert chunk.text == "hello world"


def test_hybrid_retriever_rrf_fusion() -> None:
    class FakeStore:
        def __init__(self, results: dict[str, list[SearchResult]]) -> None:
            self.results = results

        def search(
            self,
            query_vector: list[float],
            *,
            modality: str | None = None,
            top_k: int = 10,
            filters: dict | None = None,
        ) -> list[SearchResult]:
            return self.results.get(modality or "text", [])

    text_results = [
        SearchResult(
            id="t1", document_id="d1", document_name="d1.pdf",
            modality=Modality.TEXT, text="alpha", score=0.9, rank=1,
        ),
        SearchResult(
            id="t2", document_id="d1", document_name="d1.pdf",
            modality=Modality.TEXT, text="beta", score=0.8, rank=2,
        ),
    ]
    visual_results = [
        SearchResult(
            id="v1", document_id="d1", document_name="d1.pdf",
            modality=Modality.VISUAL, page_number=1, score=0.85, rank=1,
        ),
    ]
    store = FakeStore({"text": text_results, "visual": visual_results})
    retriever = HybridRetriever(store, text_search_k=10, visual_search_k=10, rrf_k=60)
    fused = retriever.retrieve([0.0] * 8, [0.0] * 8, top_k=3)
    assert len(fused) == 3
    ids = [r.id for r in fused]
    assert "t1" in ids
    assert "v1" in ids
    assert fused[0].score >= fused[-1].score


def test_noop_reranker() -> None:
    results = [
        SearchResult(
            id="r1", document_id="d1", document_name="d1.pdf",
            modality=Modality.TEXT, text="x", score=0.5, rank=1,
        ),
    ]
    reranker = NoOpReranker()
    assert reranker.rerank("q", results, top_k=1) == results
