"""Late-interaction multi-vector storage for ColQwen / ColPali embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import StorageError
from ..models import Modality, MultiVectorItem, SearchResult

logger = logging.getLogger(__name__)


def _maxsim(
    query_vectors: np.ndarray,
    page_vectors: np.ndarray,
) -> float:
    """Compute late-interaction MaxSim score.

    Args:
        query_vectors: array of shape ``[n_query_tokens, dim]``.
        page_vectors: array of shape ``[n_patches, dim]``.

    Returns:
        Scalar MaxSim score.
    """
    # For each query token, compute max similarity over all page patches.
    # Cosine similarity via normalized dot product.
    q_norm = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-8)
    p_norm = page_vectors / (np.linalg.norm(page_vectors, axis=1, keepdims=True) + 1e-8)
    similarities = q_norm @ p_norm.T  # [n_query_tokens, n_patches]
    return float(np.mean(np.max(similarities, axis=1)))


class MultiVectorStore:
    """In-memory late-interaction multi-vector store with disk persistence.

    This store is designed for ColQwen/ColPali-style embeddings where each
    document page is represented by many patch vectors. It computes the
    MaxSim score between a query's token vectors and each stored page.

    Persistence: items are serialized to JSON lines in a sidecar file next
    to the SQLite vec store. On init, the file is loaded if it exists.
    """

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, MultiVectorItem] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    item = MultiVectorItem.model_validate(data)
                    # Convert image_path string back to Path if present.
                    if item.image_path is not None:
                        item.image_path = Path(item.image_path)
                    self._items[item.id] = item
                    self._arrays[item.id] = np.array(item.embeddings, dtype=np.float32)
        except Exception as exc:
            raise StorageError(f"Failed to load multi-vector store: {exc}") from exc

    def _save(self) -> None:
        try:
            with self.storage_path.open("w", encoding="utf-8") as f:
                for item in self._items.values():
                    # image_bytes is binary PNG data; exclude it from JSON
                    # persistence and rely on image_path for display.
                    f.write(item.model_dump_json(exclude={"image_bytes"}) + "\n")
        except Exception as exc:
            raise StorageError(f"Failed to save multi-vector store: {exc}") from exc

    def add(self, items: list[MultiVectorItem]) -> None:
        if not items:
            return
        for item in items:
            self._items[item.id] = item
            self._arrays[item.id] = np.array(item.embeddings, dtype=np.float32)
        self._save()

    def search(
        self,
        query_vectors: list[list[float]],
        *,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search pages by MaxSim against query token vectors."""
        if not self._items:
            return []
        q = np.array(query_vectors, dtype=np.float32)
        if q.ndim != 2:
            raise StorageError("query_vectors must be a 2-D matrix [n_tokens, dim]")

        scored: list[tuple[float, str]] = []
        for item_id, page_vectors in self._arrays.items():
            if page_vectors.ndim != 2 or page_vectors.shape[1] != q.shape[1]:
                logger.warning(
                    "Skipping item %s with incompatible shape %s for query dim %s",
                    item_id,
                    page_vectors.shape,
                    q.shape,
                )
                continue
            score = _maxsim(q, page_vectors)
            scored.append((score, item_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[SearchResult] = []
        for rank, (score, item_id) in enumerate(scored[:top_k], start=1):
            item = self._items[item_id]
            results.append(
                SearchResult(
                    id=item.id,
                    document_id=item.document_id,
                    document_name=item.document_name,
                    modality=Modality.VISUAL,
                    page_number=item.page_number,
                    image_path=item.image_path,
                    score=score,
                    rank=rank,
                    metadata=item.metadata,
                )
            )
        return results

    def delete_document(self, document_id: str) -> None:
        ids_to_remove = [item_id for item_id, item in self._items.items() if item.document_id == document_id]
        for item_id in ids_to_remove:
            del self._items[item_id]
            del self._arrays[item_id]
        if ids_to_remove:
            self._save()

    def list_documents(self) -> dict[str, dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        for item in self._items.values():
            if item.document_id not in docs:
                docs[item.document_id] = {
                    "document_name": item.document_name,
                    "page_count": 0,
                }
            docs[item.document_id]["page_count"] += 1
        return docs

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "items": len(self._items),
            "path": str(self.storage_path),
        }

    def close(self) -> None:
        self._save()
