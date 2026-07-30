"""Tests for vector storage backends."""

from __future__ import annotations

import pytest

from local_rag_stack.models import EmbeddedItem, Modality
from local_rag_stack.storage.sqlite_vec_store import SQLiteVecStore


@pytest.fixture
def store(tmp_path) -> SQLiteVecStore:
    db = tmp_path / "test.sqlite"
    return SQLiteVecStore(db)


def _make_item(idx: int, dim: int, modality: Modality) -> EmbeddedItem:
    vector = [float(i == idx % dim) for i in range(dim)]
    return EmbeddedItem(
        id=f"item_{idx}",
        document_id="doc1",
        document_name="doc1.pdf",
        modality=modality,
        page_number=idx + 1,
        text="hello" if modality == Modality.TEXT else None,
        embedding=vector,
        metadata={"idx": idx},
    )


def test_sqlite_add_and_search(store: SQLiteVecStore) -> None:
    items = [_make_item(i, 8, Modality.TEXT) for i in range(5)]
    store.add(items)

    query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    results = store.search(query, modality=Modality.TEXT.value, top_k=3)
    assert len(results) == 3
    assert results[0].id == "item_0"
    assert results[0].document_id == "doc1"


def test_sqlite_delete_document(store: SQLiteVecStore) -> None:
    items = [_make_item(i, 8, Modality.TEXT) for i in range(3)]
    store.add(items)
    store.delete_document("doc1")
    results = store.search([1.0] * 8, modality=Modality.TEXT.value, top_k=10)
    assert len(results) == 0


def test_sqlite_list_documents(store: SQLiteVecStore) -> None:
    items = [_make_item(i, 8, Modality.TEXT) for i in range(2)]
    store.add(items)
    docs = store.list_documents()
    assert "doc1" in docs
    assert docs["doc1"]["chunk_count"] == 2
