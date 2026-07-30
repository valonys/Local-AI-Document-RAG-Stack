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


@router.post("/extract")
async def extract(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Parse a document to per-page markdown + chunks, WITHOUT indexing it.

    Drop-in document parser for callers that just want the extracted content (e.g. CaaS, as a
    LandingAI-free provider). Returns the shape:
        { pages: [{ markdown, chunks: [{id, page, type, text, label}] }], page_count, document_name }
    Page markdown is reconstructed by grouping text chunks on their page_number.
    """
    from collections import defaultdict

    suffix = Path(file.filename or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        doc = _get_pipeline(request).parse_only(tmp_path, document_name=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extract failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    by_page: dict[int, list[Any]] = defaultdict(list)
    for c in doc.chunks:
        by_page[c.page_number or 1].append(c)

    pages: list[dict[str, Any]] = []
    for pageno in sorted(by_page):
        page_chunks = by_page[pageno]
        chunks = [
            {
                "id": c.chunk_id,
                "page": pageno,
                "type": str(c.metadata.get("type", "text")) if isinstance(c.metadata, dict) else "text",
                "text": c.text,
                "label": c.section_heading,
            }
            for c in page_chunks
        ]
        markdown = "\n\n".join(
            (f"**{c.section_heading}:** " if c.section_heading else "") + c.text for c in page_chunks
        )
        pages.append({"markdown": markdown, "chunks": chunks})

    return {
        "pages": pages,
        "page_count": max(len(pages), len(doc.pages)),
        "document_name": doc.document_name,
    }


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
