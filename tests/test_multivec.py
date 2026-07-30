"""Tests for late-interaction multi-vector storage and retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from local_rag_stack.models import Modality, MultiVectorItem, SearchResult
from local_rag_stack.retrieval.hybrid import HybridRetriever
from local_rag_stack.storage.multivec_store import MultiVectorStore


@pytest.fixture
def multivec_store(tmp_path) -> MultiVectorStore:
    return MultiVectorStore(tmp_path / "multivec.jsonl")


def _make_multivec_item(idx: int, dim: int = 4, n_patches: int = 3) -> MultiVectorItem:
    # Make one patch strongly aligned with a one-hot dimension.
    vectors = np.eye(dim)[:n_patches, :].astype(float).tolist()
    return MultiVectorItem(
        id=f"page_{idx}",
        document_id="doc1",
        document_name="doc1.pdf",
        page_number=idx + 1,
        embeddings=vectors,
    )


def test_multivec_store_search(multivec_store: MultiVectorStore) -> None:
    items = [_make_multivec_item(i) for i in range(3)]
    multivec_store.add(items)

    # Query aligned with first patch of page_0.
    query = [[1.0, 0.0, 0.0, 0.0]]
    results = multivec_store.search(query, top_k=3)
    assert len(results) == 3
    assert results[0].id == "page_0"
    assert results[0].modality == Modality.VISUAL


def test_multivec_store_delete_document(multivec_store: MultiVectorStore) -> None:
    items = [_make_multivec_item(i) for i in range(2)]
    multivec_store.add(items)
    multivec_store.delete_document("doc1")
    results = multivec_store.search([[1.0, 0.0, 0.0, 0.0]], top_k=10)
    assert len(results) == 0


def test_multivec_store_persistence(multivec_store: MultiVectorStore) -> None:
    items = [_make_multivec_item(0)]
    multivec_store.add(items)
    multivec_store.close()

    store2 = MultiVectorStore(multivec_store.storage_path)
    results = store2.search([[1.0, 0.0, 0.0, 0.0]], top_k=10)
    assert len(results) == 1


def test_hybrid_retriever_with_multivec() -> None:
    class FakeDenseStore:
        def search(self, query_vector, *, modality=None, top_k=10, filters=None):
            if modality == "text":
                return [
                    SearchResult(
                        id="t1", document_id="d1", document_name="d1.pdf",
                        modality=Modality.TEXT, text="alpha", score=0.9, rank=1,
                    ),
                ]
            return []

    multivec_store = MultiVectorStore("/tmp/test_multivec_hybrid.jsonl")
    multivec_store.add([
        MultiVectorItem(
            id="v1", document_id="d1", document_name="d1.pdf",
            page_number=1, embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )
    ])

    dense_store = FakeDenseStore()
    retriever = HybridRetriever(dense_store, multivec_store=multivec_store)
    results = retriever.retrieve(
        [1.0, 0.0],
        multivec_query_vectors=[[1.0, 0.0]],
        top_k=5,
    )
    assert len(results) == 2
    multivec_store.storage_path.unlink(missing_ok=True)
