"""Text embedding implementations."""

from __future__ import annotations

import logging
from typing import Any

from ..exceptions import EmbeddingError
from ..models import EmbeddedItem, Modality, TextChunk
from .base import TextEmbedder

logger = logging.getLogger(__name__)


def _get_embeddings(response: Any) -> list[list[float]]:
    """Extract embeddings from Ollama response object or dict."""
    if hasattr(response, "embeddings"):
        embeddings = response.embeddings
    else:
        embeddings = response.get("embeddings", [])
    return [list(v) for v in embeddings]


class OllamaTextEmbedder(TextEmbedder):
    """Text embeddings via Ollama."""

    def __init__(
        self,
        *,
        model: str = "qwen3-embedding:4b",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._client_instance: Any | None = None

    def _get_client(self) -> Any:
        if self._client_instance is None:
            try:
                from ollama import Client
            except ImportError as exc:
                raise EmbeddingError(
                    "Ollama client not installed. "
                    "Install with: pip install 'local-rag-stack[ollama]'"
                ) from exc
            self._client_instance = Client(host=self.host, timeout=self.timeout)
        return self._client_instance

    def embed_chunks(
        self,
        chunks: list[TextChunk],
        *,
        batch_size: int = 32,
    ) -> list[EmbeddedItem]:
        if not chunks:
            return []
        client = self._get_client()
        try:
            response = client.embed(model=self.model, input=[c.text for c in chunks])
        except Exception as exc:
            raise EmbeddingError(f"Ollama text embedding failed: {exc}") from exc

        vectors = _get_embeddings(response)
        if len(vectors) != len(chunks):
            raise EmbeddingError(
                f"Ollama returned {len(vectors)} embeddings for {len(chunks)} chunks"
            )

        items: list[EmbeddedItem] = []
        for chunk, vector in zip(chunks, vectors):
            items.append(
                EmbeddedItem(
                    id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    modality=Modality.TEXT,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    embedding=list(vector),
                    metadata={
                        "section_heading": chunk.section_heading,
                        "bbox": chunk.bbox.model_dump() if chunk.bbox else None,
                    },
                )
            )
        return items

    def embed_query(self, query: str) -> list[float]:
        client = self._get_client()
        try:
            response = client.embed(model=self.model, input=[query])
        except Exception as exc:
            raise EmbeddingError(f"Ollama query embedding failed: {exc}") from exc
        vectors = _get_embeddings(response)
        if not vectors:
            raise EmbeddingError("Ollama returned no query embedding")
        return list(vectors[0])


class SentenceTransformersEmbedder(TextEmbedder):
    """Text embeddings via sentence-transformers (local, no Ollama)."""

    def __init__(self, *, model: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model
        self._model_instance: Any | None = None

    def _get_model(self) -> Any:
        if self._model_instance is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            self._model_instance = SentenceTransformer(self.model_name)
        return self._model_instance

    def embed_chunks(
        self,
        chunks: list[TextChunk],
        *,
        batch_size: int = 32,
    ) -> list[EmbeddedItem]:
        if not chunks:
            return []
        model = self._get_model()
        try:
            vectors = model.encode(
                [c.text for c in chunks],
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"sentence-transformers embedding failed: {exc}") from exc

        items: list[EmbeddedItem] = []
        for chunk, vector in zip(chunks, vectors):
            items.append(
                EmbeddedItem(
                    id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    modality=Modality.TEXT,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    embedding=vector.tolist(),
                    metadata={"section_heading": chunk.section_heading},
                )
            )
        return items

    def embed_query(self, query: str) -> list[float]:
        model = self._get_model()
        try:
            vector = model.encode(query, normalize_embeddings=True)
        except Exception as exc:
            raise EmbeddingError(f"sentence-transformers query embedding failed: {exc}") from exc
        return vector.tolist()
