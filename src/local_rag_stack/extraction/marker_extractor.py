"""Marker-based document extraction (GPL-3 optional dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import ExtractionError
from ..models import ExtractedDocument, PageImage, TextChunk
from .base import DocumentExtractor


class MarkerExtractor(DocumentExtractor):
    """Extract text and page images using Marker."""

    def __init__(self, *, dpi: int = 150, output_dir: Path | None = None) -> None:
        super().__init__(dpi=dpi, output_dir=output_dir)
        try:
            from marker.converters.pdf import PdfConverter  # noqa: F401
            from marker.models import create_model_dict  # noqa: F401
        except ImportError as exc:
            raise ExtractionError(
                "Marker is not installed. Install with: pip install 'local-rag-stack[marker]'"
            ) from exc
        self._converter = PdfConverter(artifact_dict=create_model_dict())

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
        metadata = metadata or {}

        try:
            rendered = self._converter(str(file_path))
        except Exception as exc:
            raise ExtractionError(f"Marker conversion failed for {file_path}: {exc}") from exc

        md = rendered.markdown
        chunks = self._chunks_from_markdown(md, document_id, document_name)
        pages = self._render_pages(rendered, document_id, document_name)

        return ExtractedDocument(
            document_id=document_id,
            document_name=document_name,
            source_path=file_path,
            mime_type=None,
            chunks=chunks,
            pages=pages,
            metadata=metadata,
        )

    def _chunks_from_markdown(
        self,
        markdown: str,
        document_id: str,
        document_name: str,
    ) -> list[TextChunk]:
        texts = self.split_text(markdown, chunk_size=512, chunk_overlap=64)
        return [
            TextChunk(
                chunk_id=f"{document_id}_chunk_{i:05d}",
                document_id=document_id,
                document_name=document_name,
                page_number=None,
                text=text,
            )
            for i, text in enumerate(texts)
        ]

    def _render_pages(
        self,
        rendered: Any,
        document_id: str,
        document_name: str,
    ) -> list[PageImage]:
        """Render pages from Marker output images if available."""
        pages: list[PageImage] = []
        images = getattr(rendered, "images", {})
        for i, (img_id, pil_image) in enumerate(images.items(), start=1):
            image_path = self._page_image_path(document_id, i)
            try:
                if image_path:
                    pil_image.save(image_path, "PNG")
                width, height = pil_image.size
                pages.append(
                    PageImage(
                        image_id=f"{document_id}_page_{i:04d}",
                        document_id=document_id,
                        document_name=document_name,
                        page_number=i,
                        image_path=image_path,
                        width=width,
                        height=height,
                    )
                )
            except Exception:
                continue
        return pages
