"""FastAPI route handlers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..models import (
    DocumentSummary,
    GraphNeighborhoodRequest,
    GraphQueryRequest,
    GraphQueryResponse,
    HealthResponse,
    IngestRequest,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()


def _get_pipeline(request: Request):
    return request.app.state.pipeline


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return _get_pipeline(request).health()


@router.post("/ingest", response_model=DocumentSummary)
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
) -> DocumentSummary:
    import json

    try:
        meta: dict[str, Any] = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="metadata must be valid JSON") from exc

    suffix = Path(file.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        doc = _get_pipeline(request).ingest(
            tmp_path,
            document_name=file.filename,
            metadata=meta,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return DocumentSummary(
        document_id=doc.document_id,
        document_name=doc.document_name,
        chunk_count=len(doc.chunks),
        page_count=len(doc.pages),
        ingested_at=doc.extracted_at,
    )


@router.post("/ingest/path", response_model=DocumentSummary)
async def ingest_path(request: Request, body: IngestRequest) -> DocumentSummary:
    doc = _get_pipeline(request).ingest(body.file_path, metadata=body.metadata)
    return DocumentSummary(
        document_id=doc.document_id,
        document_name=doc.document_name,
        chunk_count=len(doc.chunks),
        page_count=len(doc.pages),
        ingested_at=doc.extracted_at,
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    try:
        return _get_pipeline(request).query(
            body.query,
            top_k=body.top_k,
            include_images=body.include_images,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(request: Request) -> list[DocumentSummary]:
    return _get_pipeline(request).list_documents()


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, request: Request) -> dict[str, str]:
    try:
        _get_pipeline(request).delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc


@router.post("/graph/query", response_model=GraphQueryResponse)
async def graph_query(request: Request, body: GraphQueryRequest) -> GraphQueryResponse:
    try:
        return _get_pipeline(request).graph_query(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph query failed: {exc}") from exc


@router.post("/graph/neighborhood", response_model=GraphQueryResponse)
async def graph_neighborhood(
    request: Request, body: GraphNeighborhoodRequest
) -> GraphQueryResponse:
    try:
        return _get_pipeline(request).graph_neighborhood(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph neighborhood failed: {exc}") from exc
