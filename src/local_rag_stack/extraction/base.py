"""Abstract base and shared helpers for document extraction."""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import ExtractedDocument, PageImage, TextChunk


class DocumentExtractor(ABC):
    """Extract text chunks and page images from a document file."""

    def __init__(self, *, dpi: int = 150, output_dir: Path | None = None) -> None:
        self.dpi = dpi
        self.output_dir = output_dir

    @abstractmethod
    def extract(
        self,
        file_path: Path,
        *,
        document_id: str | None = None,
        document_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExtractedDocument:
        """Extract chunks and page images from ``file_path``."""

    @staticmethod
    def make_document_id(file_path: Path) -> str:
        """Create a deterministic document id from the file path and content hash."""
        content_hash = hashlib.sha256()
        content_hash.update(str(file_path).encode())
        if file_path.exists():
            content_hash.update(file_path.read_bytes())
        return content_hash.hexdigest()[:16]

    @staticmethod
    def split_text(
        text: str,
        *,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> list[str]:
        """Naive overlap chunking used as a fallback when no structured chunks exist."""
        if not text.strip():
            return []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            if end >= len(text):
                break
            start += chunk_size - chunk_overlap
        return [c for c in chunks if c]

    def _page_image_path(self, document_id: str, page_number: int) -> Path | None:
        if self.output_dir is None:
            return None
        page_dir = self.output_dir / document_id / "pages"
        page_dir.mkdir(parents=True, exist_ok=True)
        return page_dir / f"page_{page_number:04d}.png"
