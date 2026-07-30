"""Qdrant vector store implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..exceptions import StorageError
from ..models import EmbeddedItem, Modality, SearchResult
from .base import VectorStore

logger = logging.getLogger(__name__)


class QdrantStore(VectorStore):
    """Vector store backed by Qdrant.

    Uses a single collection with a ``modality`` payload field and a dense
    vector of a fixed size. All embeddings must be projected or padded to the
    same dimension. If you mix ColQwen (multi-vector) with text embeddings,
    use sqlite-vec with separate tables or two Qdrant collections.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        *,
        api_key: str | None = None,
        collection: str = "local_rag_stack",
        vector_size: int = 1024,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.collection = collection
        self.vector_size = vector_size
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise StorageError(
                    "qdrant-client not installed. "
                    "Install with: pip install 'local-rag-stack[qdrant]'"
                ) from exc
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self) -> None:
        client = self._client
        if not client.collection_exists(self.collection):
            try:
                from qdrant_client.models import Distance, VectorParams
            except ImportError as exc:
                raise StorageError("qdrant-client models import failed") from exc
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        """Project/pad vector to vector_size (simple truncation/padding)."""
        if len(vector) == self.vector_size:
            return vector
        if len(vector) > self.vector_size:
            return vector[: self.vector_size]
        return vector + [0.0] * (self.vector_size - len(vector))

    def add(self, items: list[EmbeddedItem]) -> None:
        if not items:
            return
        client = self._get_client()
        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            raise StorageError("qdrant-client models import failed") from exc

        points = []
        for item in items:
            points.append(
                PointStruct(
                    id=item.id,
                    vector=self._normalize_vector(item.embedding),
                    payload={
                        "document_id": item.document_id,
                        "document_name": item.document_name,
                        "modality": item.modality.value,
                        "page_number": item.page_number,
                        "text": item.text,
                        "image_path": str(item.image_path) if item.image_path else None,
                        "metadata": item.metadata,
                    },
                )
            )
        client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(
        self,
        query_vector: list[float],
        *,
        modality: str | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        client = self._get_client()
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
        except ImportError as exc:
            raise StorageError("qdrant-client models import failed") from exc

        query_filter = None
        conditions = []
        if modality:
            conditions.append(
                FieldCondition(key="modality", match=MatchValue(value=modality))
            )
        if filters:
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        if conditions:
            query_filter = Filter(must=conditions)

        response = client.search(
            collection_name=self.collection,
            query_vector=self._normalize_vector(query_vector),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for rank, scored in enumerate(response, start=1):
            payload = scored.payload or {}
            results.append(
                SearchResult(
                    id=scored.id,
                    document_id=payload.get("document_id", ""),
                    document_name=payload.get("document_name", ""),
                    modality=Modality(payload.get("modality", "text")),
                    page_number=payload.get("page_number"),
                    text=payload.get("text"),
                    image_path=Path(payload["image_path"]) if payload.get("image_path") else None,
                    score=float(scored.score),
                    rank=rank,
                    metadata=payload.get("metadata", {}),
                )
            )
        return results

    def delete_document(self, document_id: str) -> None:
        client = self._get_client()
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
        except ImportError as exc:
            raise StorageError("qdrant-client models import failed") from exc
        client.delete(
            collection_name=self.collection,
            points_filter=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )

    def list_documents(self) -> dict[str, dict[str, Any]]:
        client = self._get_client()
        docs: dict[str, dict[str, Any]] = {}
        offset = None
        while True:
            response = client.scroll(
                collection_name=self.collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, offset = response
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("document_id")
                if not doc_id:
                    continue
                if doc_id not in docs:
                    docs[doc_id] = {
                        "document_name": payload.get("document_name", ""),
                        "chunk_count": 0,
                        "page_count": 0,
                    }
                modality = payload.get("modality")
                if modality == Modality.TEXT.value:
                    docs[doc_id]["chunk_count"] += 1
                elif modality == Modality.VISUAL.value:
                    docs[doc_id]["page_count"] += 1
            if offset is None:
                break
        return docs

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def health(self) -> dict[str, Any]:
        try:
            client = self._get_client()
            info = client.get_collection(self.collection)
            return {
                "status": "ok",
                "url": self.url,
                "collection": self.collection,
                "vectors_count": info.vectors_count,
            }
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
