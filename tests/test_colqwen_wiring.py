"""Wiring tests for the ColQwen multi-vector embedder (skipped if not installed)."""

from __future__ import annotations

import importlib.util

import pytest

from local_rag_stack.config import Settings
from local_rag_stack.pipeline import RAGPipeline

colpali_available = importlib.util.find_spec("colpali_engine") is not None


@pytest.mark.skipif(not colpali_available, reason="colpali-engine not installed")
def test_colqwen_pipeline_builds(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        sqlite_vec_path=tmp_path / "test.vec.sqlite",
        extraction_engine="plaintext",
        text_embed_model="BAAI/bge-small-en-v1.5",
        vision_embed_backend="hf",
        vision_embed_model="vidore/colqwen2-v1.0",
        rerank_backend="none",
    )  # type: ignore[call-arg]
    pipeline = RAGPipeline(settings)
    assert pipeline.multivec_embedder is not None
    assert pipeline.multivec_store is not None
    pipeline.close()
