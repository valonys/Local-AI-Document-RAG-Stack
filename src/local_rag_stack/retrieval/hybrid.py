"""Hybrid retrieval with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
from typing import Any

from ..models import Modality, SearchResult
from ..storage.base import VectorStore
from ..storage.multivec_store import MultiVectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieve from text and visual indexes and fuse with RRF."""

    def __init__(
        self,
        store: VectorStore,
        *,
        multivec_store: MultiVectorStore | None = None,
        text_search_k: int = 10,
        visual_search_k: int = 10,
        multivec_search_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.store = store
        self.multivec_store = multivec_store
        self.text_search_k = text_search_k
        self.visual_search_k = visual_search_k
        self.multivec_search_k = multivec_search_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        text_query_vector: list[float],
        visual_query_vector: list[float] | None = None,
        multivec_query_vectors: list[list[float]] | None = None,
        *,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Hybrid retrieve across text, dense visual, and multi-vector visual indexes.

        Args:
            text_query_vector: query embedding from a text embedder.
            visual_query_vector: query embedding from a dense visual embedder.
                If None, dense visual search is skipped.
            multivec_query_vectors: query token matrix from a late-interaction
                embedder. If None, multi-vector search is skipped.
            top_k: number of fused results to return.
        """
        all_results: list[list[SearchResult]] = []

        text_results = self.store.search(
            text_query_vector,
            modality=Modality.TEXT.value,
            top_k=self.text_search_k,
        )
        all_results.append(text_results)

        if visual_query_vector is not None:
            visual_results = self.store.search(
                visual_query_vector,
                modality=Modality.VISUAL.value,
                top_k=self.visual_search_k,
            )
            all_results.append(visual_results)

        if multivec_query_vectors is not None and self.multivec_store is not None:
            multivec_results = self.multivec_store.search(
                multivec_query_vectors,
                top_k=self.multivec_search_k,
            )
            all_results.append(multivec_results)

        flat = [r for sublist in all_results for r in sublist]
        return self._rrf_fuse(flat, top_k=top_k)

    def _rrf_fuse(self, results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        """Reciprocal Rank Fusion over ranked result lists."""
        scores: dict[str, float] = {}
        result_by_id: dict[str, SearchResult] = {}

        for result in results:
            result_by_id[result.id] = result
            rank = result.rank if result.rank is not None else 1
            scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (self.rrf_k + rank)

        sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
        fused: list[SearchResult] = []
        for rank, item_id in enumerate(sorted_ids[:top_k], start=1):
            base = result_by_id[item_id]
            fused.append(
                SearchResult(
                    id=base.id,
                    document_id=base.document_id,
                    document_name=base.document_name,
                    modality=base.modality,
                    page_number=base.page_number,
                    text=base.text,
                    image_path=base.image_path,
                    score=scores[item_id],
                    rank=rank,
                    metadata=base.metadata,
                )
            )
        return fused
