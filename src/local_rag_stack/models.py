"""Shared pydantic models for documents, chunks, and search results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    VISUAL = "visual"
    TABLE = "table"


class BoundingBox(BaseModel):
    """Normalized or absolute bounding box for a region on a page."""

    x: float
    y: float
    width: float
    height: float
    page_width: float | None = None
    page_height: float | None = None


class TextChunk(BaseModel):
    """A text chunk extracted from a document."""

    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    section_heading: str | None = None
    text: str
    bbox: BoundingBox | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageImage(BaseModel):
    """A rendered page image from a document."""

    image_id: str
    document_id: str
    document_name: str
    page_number: int
    image_path: Path | None = None
    image_bytes: bytes | None = None
    mime_type: str = "image/png"
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedDocument(BaseModel):
    """Output of the document extraction stage."""

    document_id: str
    document_name: str
    source_path: Path | None = None
    mime_type: str | None = None
    chunks: list[TextChunk] = Field(default_factory=list)
    pages: list[PageImage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EmbeddedItem(BaseModel):
    """A chunk or page with a single dense embedding vector."""

    id: str
    document_id: str
    document_name: str
    modality: Modality
    page_number: int | None = None
    text: str | None = None
    image_path: Path | None = None
    image_bytes: bytes | None = None
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultiVectorItem(BaseModel):
    """A page or image with a late-interaction multi-vector embedding.

    For ColQwen/ColPali, each page is represented by a matrix of patch
    embeddings with shape ``[n_patches, dim]``.
    """

    id: str
    document_id: str
    document_name: str
    modality: Modality = Modality.VISUAL
    page_number: int | None = None
    image_path: Path | None = None
    image_bytes: bytes | None = None
    embeddings: list[list[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single result from a retrieval step."""

    id: str
    document_id: str
    document_name: str
    modality: Modality
    page_number: int | None = None
    text: str | None = None
    image_path: Path | None = None
    score: float
    rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """API request body for ingest-by-path (server-local file)."""

    file_path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """API request body for querying the RAG stack."""

    query: str
    top_k: int = 5
    include_images: bool = True


class Citation(BaseModel):
    """Citation attached to an answer."""

    source_id: str
    document_name: str
    page_number: int | None = None
    modality: Modality
    snippet: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    """API response for a query."""

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[SearchResult] = Field(default_factory=list)
    model: str
    elapsed_seconds: float


class DocumentSummary(BaseModel):
    """Summary of an ingested document."""

    document_id: str
    document_name: str
    chunk_count: int
    page_count: int
    ingested_at: datetime | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    ollama_ready: bool
    vector_store: str
    graph_store: str | None = None
    models: dict[str, str | None]
    tenant_id: str | None = None


class GraphQueryRequest(BaseModel):
    """API request body for graph search."""

    query: str | None = None
    entity_types: list[str] | None = None
    entity_name: str | None = None
    relation_type: str | None = None
    document_id: str | None = None
    top_k: int = 20


class GraphNeighborhoodRequest(BaseModel):
    """API request body for graph neighborhood expansion."""

    entity_id: str
    hops: int = 1


class GraphEntityResponse(BaseModel):
    """Serialized entity for API responses."""

    id: str
    name: str
    entity_type: str
    document_id: str
    chunk_id: str | None = None
    source_text: str | None = None


class GraphRelationResponse(BaseModel):
    """Serialized relation for API responses."""

    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    document_id: str
    chunk_id: str | None = None
    source_text: str | None = None


class GraphQueryResponse(BaseModel):
    """API response for graph queries."""

    query: str | None = None
    entities: list[GraphEntityResponse] = Field(default_factory=list)
    relations: list[GraphRelationResponse] = Field(default_factory=list)
