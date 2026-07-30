"""Rerankers for text and multimodal retrieval results."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from ..exceptions import RetrievalError
from ..models import Modality, SearchResult

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Rerank retrieved items relative to a query."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Return the top_k reranked results."""


class NoOpReranker(Reranker):
    """Pass-through reranker for testing or when no model is available."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        return results[:top_k]


class OllamaReranker(Reranker):
    """Rerank using an Ollama-compatible cross-encoder model.

    Note: Ollama does not expose a standard /rerank endpoint at the time of
    writing. This implementation falls back to a prompt-based pairwise score
    using the configured LLM and should be replaced once native reranking is
    available.
    """

    def __init__(
        self,
        *,
        model: str = "qwen3:8b",
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
                raise RetrievalError(
                    "Ollama client not installed. "
                    "Install with: pip install 'local-rag-stack[ollama]'"
                ) from exc
            self._client_instance = Client(host=self.host, timeout=self.timeout)
        return self._client_instance

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []
        # Simple pointwise prompt scoring. Not ideal but works without a
        # dedicated rerank endpoint.
        client = self._get_client()
        scored: list[tuple[float, SearchResult]] = []
        for result in results:
            text = result.text or f"Page {result.page_number} from {result.document_name}"
            prompt = (
                "On a scale of 0 to 10, how relevant is the following passage "
                f"to the query? Reply with only a number.\n\nQuery: {query}\n\nPassage: {text}\n"
            )
            try:
                response = client.generate(model=self.model, prompt=prompt)
                text = getattr(response, "response", None)
                if text is None and isinstance(response, dict):
                    text = response.get("response")
                content = str(text or "5").strip()
                score = float("".join(c for c in content if c.isdigit() or c == ".") or "5")
                score = max(0.0, min(10.0, score)) / 10.0
            except Exception as exc:
                logger.warning("Reranking failed for result %s: %s", result.id, exc)
                score = 0.5
            scored.append((score, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked: list[SearchResult] = []
        for rank, (score, result) in enumerate(scored[:top_k], start=1):
            reranked.append(
                SearchResult(
                    id=result.id,
                    document_id=result.document_id,
                    document_name=result.document_name,
                    modality=result.modality,
                    page_number=result.page_number,
                    text=result.text,
                    image_path=result.image_path,
                    score=score,
                    rank=rank,
                    metadata=result.metadata,
                )
            )
        return reranked


class SentenceTransformersReranker(Reranker):
    """Cross-encoder reranker via sentence-transformers."""

    def __init__(self, *, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model
        self._model_instance: Any | None = None

    def _get_model(self) -> Any:
        if self._model_instance is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RetrievalError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            self._model_instance = CrossEncoder(self.model_name)
        return self._model_instance

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []
        model = self._get_model()
        pairs = [
            (query, result.text or f"Page {result.page_number} from {result.document_name}")
            for result in results
        ]
        try:
            scores = model.predict(pairs)
        except Exception as exc:
            raise RetrievalError(f"Reranking failed: {exc}") from exc

        scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        reranked: list[SearchResult] = []
        for rank, (score, result) in enumerate(scored[:top_k], start=1):
            reranked.append(
                SearchResult(
                    id=result.id,
                    document_id=result.document_id,
                    document_name=result.document_name,
                    modality=result.modality,
                    page_number=result.page_number,
                    text=result.text,
                    image_path=result.image_path,
                    score=float(score),
                    rank=rank,
                    metadata=result.metadata,
                )
            )
        return reranked
