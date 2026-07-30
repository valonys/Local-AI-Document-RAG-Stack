"""Vector store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import EmbeddedItem, SearchResult


class VectorStore(ABC):
    """Abstract vector store for text and visual embeddings."""

    @abstractmethod
    def add(self, items: list[EmbeddedItem]) -> None:
        """Add or update items in the store."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        *,
        modality: str | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for nearest neighbors."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Delete all items belonging to ``document_id``."""

    @abstractmethod
    def list_documents(self) -> dict[str, dict[str, Any]]:
        """Return a mapping of document_id -> metadata summary."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return health/status information."""
