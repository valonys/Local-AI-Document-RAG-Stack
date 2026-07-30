"""Embedding interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import EmbeddedItem, ExtractedDocument, MultiVectorItem, PageImage, TextChunk


class TextEmbedder(ABC):
    """Generate dense embeddings for text chunks."""

    @abstractmethod
    def embed_chunks(
        self,
        chunks: list[TextChunk],
        *,
        batch_size: int = 32,
    ) -> list[EmbeddedItem]:
        """Embed a list of text chunks."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a query string."""


class VisualEmbedder(ABC):
    """Generate dense embeddings for page images."""

    @abstractmethod
    def embed_pages(
        self,
        pages: list[PageImage],
        *,
        batch_size: int = 8,
    ) -> list[EmbeddedItem]:
        """Embed a list of page images."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a query string for visual retrieval."""


class MultiVectorEmbedder(ABC):
    """Generate late-interaction multi-vector embeddings for page images."""

    @abstractmethod
    def embed_pages(
        self,
        pages: list[PageImage],
        *,
        batch_size: int = 4,
    ) -> list[MultiVectorItem]:
        """Embed a list of page images into multi-vector matrices."""

    @abstractmethod
    def embed_query(self, query: str) -> list[list[float]]:
        """Embed a query into token vectors for late-interaction search."""


class EmbeddingPipeline:
    """High-level helper that embeds an extracted document."""

    def __init__(
        self,
        text_embedder: TextEmbedder,
        visual_embedder: VisualEmbedder | None = None,
        multivec_embedder: MultiVectorEmbedder | None = None,
    ) -> None:
        self.text_embedder = text_embedder
        self.visual_embedder = visual_embedder
        self.multivec_embedder = multivec_embedder

    def embed(self, doc: ExtractedDocument) -> tuple[list[EmbeddedItem], list[MultiVectorItem]]:
        """Embed all chunks and pages from a document."""
        dense_items: list[EmbeddedItem] = []
        multi_items: list[MultiVectorItem] = []
        if doc.chunks:
            dense_items.extend(self.text_embedder.embed_chunks(doc.chunks))
        if self.multivec_embedder and doc.pages:
            multi_items.extend(self.multivec_embedder.embed_pages(doc.pages))
        elif self.visual_embedder and doc.pages:
            dense_items.extend(self.visual_embedder.embed_pages(doc.pages))
        return dense_items, multi_items
