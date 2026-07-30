"""Lightweight text extraction fallback using PyMuPDF and openpyxl when available.

This extractor keeps the stack self-contained: it requires no Docling or Marker,
and still produces clean text chunks from PDF, XLSX, and plain-text files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..exceptions import ExtractionError
from ..models import ExtractedDocument, TextChunk
from .base import DocumentExtractor

logger = logging.getLogger(__name__)


class PlainTextExtractor(DocumentExtractor):
    """Extract text from common document formats with lightweight deps.

    - PDF: PyMuPDF (`fitz`)
    - XLSX: openpyxl
    - Plain text: direct read
    """

    SUPPORTED_SUFFIXES = {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".log",
        ".pdf",
        ".xlsx",
        ".xls",
    }

    def extract(
        self,
        file_path: Path,
        *,
        document_id: str | None = None,
        document_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise ExtractionError(f"File not found: {file_path}")

        document_id = document_id or self.make_document_id(file_path)
        document_name = document_name or file_path.name
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            chunks = self._extract_pdf(file_path, document_id, document_name)
        elif suffix in (".xlsx", ".xls"):
            chunks = self._extract_xlsx(file_path, document_id, document_name)
        else:
            chunks = self._extract_text(file_path, document_id, document_name)

        return ExtractedDocument(
            document_id=document_id,
            document_name=document_name,
            source_path=file_path,
            mime_type=None,
            chunks=chunks,
            pages=[],
            metadata=metadata or {},
        )

    def _extract_pdf(
        self, file_path: Path, document_id: str, document_name: str
    ) -> list[TextChunk]:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ExtractionError(
                "PyMuPDF is required for PDF text extraction. "
                "Install with: pip install pymupdf"
            ) from exc

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            raise ExtractionError(f"Failed to open PDF {file_path}: {exc}") from exc

        chunks: list[TextChunk] = []
        try:
            for page_no, page in enumerate(doc, start=1):
                text = page.get_text()
                if not text.strip():
                    continue
                for i, chunk_text in enumerate(
                    self.split_text(text, chunk_size=512, chunk_overlap=64)
                ):
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{document_id}_p{page_no}_c{i:04d}",
                            document_id=document_id,
                            document_name=document_name,
                            page_number=page_no,
                            text=chunk_text,
                        )
                    )
        finally:
            doc.close()
        return chunks

    def _extract_xlsx(
        self, file_path: Path, document_id: str, document_name: str
    ) -> list[TextChunk]:
        try:
            import openpyxl
        except ImportError as exc:
            raise ExtractionError(
                "openpyxl is required for XLSX text extraction. "
                "Install with: pip install openpyxl"
            ) from exc

        try:
            wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
        except Exception as exc:
            raise ExtractionError(f"Failed to open XLSX {file_path}: {exc}") from exc

        chunks: list[TextChunk] = []
        try:
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                rows: list[str] = []
                for row in ws.iter_rows(values_only=True):
                    rows.append(
                        " | ".join(str(cell) if cell is not None else "" for cell in row)
                    )
                if not rows:
                    continue
                sheet_text = f"\nSheet: {sheet_name}\n" + "\n".join(rows)
                for i, chunk_text in enumerate(
                    self.split_text(sheet_text, chunk_size=512, chunk_overlap=64)
                ):
                    chunks.append(
                        TextChunk(
                            chunk_id=f"{document_id}_{sheet_name}_c{i:04d}",
                            document_id=document_id,
                            document_name=document_name,
                            page_number=None,
                            section_heading=sheet_name,
                            text=chunk_text,
                        )
                    )
        finally:
            wb.close()
        return chunks

    def _extract_text(
        self, file_path: Path, document_id: str, document_name: str
    ) -> list[TextChunk]:
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise ExtractionError(f"Failed to read {file_path}: {exc}") from exc

        return [
            TextChunk(
                chunk_id=f"{document_id}_chunk_{i:05d}",
                document_id=document_id,
                document_name=document_name,
                page_number=None,
                text=chunk_text,
            )
            for i, chunk_text in enumerate(
                self.split_text(text, chunk_size=512, chunk_overlap=64)
            )
        ]
