"""FastAPI route handlers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

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
from ..pipeline import RAGPipeline
from ..tenant import Tenant, get_current_tenant, get_tenant_pipeline

router = APIRouter()


def _get_pipeline(request: Request, tenant: Tenant) -> RAGPipeline:
    return get_tenant_pipeline(request, tenant)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, tenant: Tenant = Depends(get_current_tenant)) -> HealthResponse:
    response = _get_pipeline(request, tenant).health()
    response.tenant_id = tenant.id
    return response


@router.post("/ingest", response_model=DocumentSummary)
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    tenant: Tenant = Depends(get_current_tenant),
) -> DocumentSummary:
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
        doc = _get_pipeline(request, tenant).ingest(
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
async def ingest_path(
    request: Request,
    body: IngestRequest,
    tenant: Tenant = Depends(get_current_tenant),
) -> DocumentSummary:
    doc = _get_pipeline(request, tenant).ingest(body.file_path, metadata=body.metadata)
    return DocumentSummary(
        document_id=doc.document_id,
        document_name=doc.document_name,
        chunk_count=len(doc.chunks),
        page_count=len(doc.pages),
        ingested_at=doc.extracted_at,
    )


@router.post("/extract")
async def extract(
    request: Request,
    file: UploadFile = File(...),
    force_full_page_ocr: bool = Form(False),
    tenant: Tenant = Depends(get_current_tenant),
) -> dict[str, Any]:
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
        pipeline = _get_pipeline(request, tenant)
        if force_full_page_ocr and pipeline.settings.extraction_engine.lower() == "docling":
            # Per-request escalation for PDFs with a corrupt native text map.
            # It avoids applying expensive full-page OCR to clean vector documents.
            from ..extraction.docling_extractor import DoclingExtractor

            extractor = DoclingExtractor(
                dpi=pipeline.settings.extraction_dpi,
                output_dir=pipeline.settings.data_dir / "pages",
                force_full_page_ocr=True,
                ocr_engine=pipeline.settings.ocr_engine,
            )
            doc = extractor.extract(tmp_path, document_name=file.filename)
        else:
            doc = pipeline.parse_only(tmp_path, document_name=file.filename)
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
        pages.append({"page": pageno, "markdown": markdown, "chunks": chunks})

    return {
        "pages": pages,
        "page_count": max(len(pages), len(doc.pages)),
        "document_name": doc.document_name,
    }


@router.post("/query", response_model=QueryResponse)
async def query(
    request: Request,
    body: QueryRequest,
    tenant: Tenant = Depends(get_current_tenant),
) -> QueryResponse:
    try:
        return _get_pipeline(request, tenant).query(
            body.query,
            top_k=body.top_k,
            include_images=body.include_images,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    request: Request,
    tenant: Tenant = Depends(get_current_tenant),
) -> list[DocumentSummary]:
    return _get_pipeline(request, tenant).list_documents()


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    request: Request,
    tenant: Tenant = Depends(get_current_tenant),
) -> dict[str, str]:
    try:
        _get_pipeline(request, tenant).delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc


@router.post("/graph/query", response_model=GraphQueryResponse)
async def graph_query(
    request: Request,
    body: GraphQueryRequest,
    tenant: Tenant = Depends(get_current_tenant),
) -> GraphQueryResponse:
    try:
        return _get_pipeline(request, tenant).graph_query(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph query failed: {exc}") from exc


@router.post("/graph/neighborhood", response_model=GraphQueryResponse)
async def graph_neighborhood(
    request: Request,
    body: GraphNeighborhoodRequest,
    tenant: Tenant = Depends(get_current_tenant),
) -> GraphQueryResponse:
    try:
        return _get_pipeline(request, tenant).graph_neighborhood(body)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph neighborhood failed: {exc}") from exc
