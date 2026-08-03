"""Docling-based document extraction."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
from typing import Any

from ..exceptions import ExtractionError
from ..models import BoundingBox, ExtractedDocument, PageImage, TextChunk
from .base import DocumentExtractor


class DoclingExtractor(DocumentExtractor):
    """Extract text and page images using IBM Docling."""

    def __init__(
        self,
        *,
        dpi: int = 150,
        output_dir: Path | None = None,
        force_full_page_ocr: bool = False,
        ocr_engine: str = "auto",
    ) -> None:
        super().__init__(dpi=dpi, output_dir=output_dir)
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import OcrMacOptions, PdfPipelineOptions, RapidOcrOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise ExtractionError(
                "Docling is not installed. Install with: pip install 'local-rag-stack[docling]'"
            ) from exc
        options = PdfPipelineOptions()
        options.ocr_options.force_full_page_ocr = force_full_page_ocr
        selected_engine = ocr_engine.lower()
        mac_vision_available = sys.platform == "darwin" and importlib.util.find_spec("ocrmac") is not None
        if selected_engine == "ocrmac" or (selected_engine == "auto" and mac_vision_available):
            if not mac_vision_available:
                raise ExtractionError("OCRMac was requested but the macOS Vision bridge is not installed.")
            options.ocr_options = OcrMacOptions(force_full_page_ocr=force_full_page_ocr)
        elif selected_engine == "rapidocr":
            options.ocr_options = RapidOcrOptions(force_full_page_ocr=force_full_page_ocr)
        elif selected_engine != "auto":
            raise ExtractionError(f"Unknown OCR engine: {ocr_engine}")
        self._converter: Any = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

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
            conv_res = self._converter.convert(str(file_path))
        except Exception as exc:
            raise ExtractionError(f"Docling conversion failed for {file_path}: {exc}") from exc

        doc = conv_res.document
        chunks = self._extract_chunks(doc, document_id, document_name)
        pages = self._render_pages(doc, document_id, document_name)

        return ExtractedDocument(
            document_id=document_id,
            document_name=document_name,
            source_path=file_path,
            mime_type=None,
            chunks=chunks,
            pages=pages,
            metadata=metadata,
        )

    def _extract_chunks(
        self,
        doc: Any,
        document_id: str,
        document_name: str,
    ) -> list[TextChunk]:
        """Export text chunks from a Docling document."""
        try:
            from docling_core.types.doc import DocItemLabel
        except Exception:
            DocItemLabel = None

        chunks: list[TextChunk] = []
        chunk_index = 0
        for item, level in doc.iterate_items():
            label = getattr(item, "label", None)
            text = getattr(item, "text", None)
            if not text or not str(text).strip():
                continue
            page_no = None
            prov = getattr(item, "prov", None)
            if prov and len(prov) > 0:
                page_no = getattr(prov[0], "page_no", None)
            bbox = None
            if prov and len(prov) > 0:
                bbox_obj = getattr(prov[0], "bbox", None)
                if bbox_obj:
                    left = getattr(bbox_obj, "l", getattr(bbox_obj, "x", 0.0))
                    top = getattr(bbox_obj, "t", getattr(bbox_obj, "y", 0.0))
                    right = getattr(bbox_obj, "r", left + getattr(bbox_obj, "width", 0.0))
                    bottom = getattr(bbox_obj, "b", top + getattr(bbox_obj, "height", 0.0))
                    bbox = BoundingBox(
                        x=left,
                        y=top,
                        width=right - left,
                        height=bottom - top,
                    )
            chunks.append(
                TextChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index:05d}",
                    document_id=document_id,
                    document_name=document_name,
                    page_number=page_no,
                    section_heading=str(label) if label else None,
                    text=str(text).strip(),
                    bbox=bbox,
                )
            )
            chunk_index += 1

        if not chunks:
            md = doc.export_to_markdown()
            fallback = self.split_text(md, chunk_size=512, chunk_overlap=64)
            for i, text in enumerate(fallback):
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document_id}_chunk_{i:05d}",
                        document_id=document_id,
                        document_name=document_name,
                        page_number=None,
                        text=text,
                    )
                )
        return chunks

    def _render_pages(
        self,
        doc: Any,
        document_id: str,
        document_name: str,
    ) -> list[PageImage]:
        """Render each page to a PNG image."""
        pages: list[PageImage] = []
        for page_no, page in enumerate(doc.pages, start=1):
            image_path = self._page_image_path(document_id, page_no)
            try:
                pil_image = page.image.pil_image
                if image_path:
                    pil_image.save(image_path, "PNG")
                width, height = pil_image.size
                image_bytes = None
                if image_path is None:
                    import io
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    image_bytes = buf.getvalue()
                pages.append(
                    PageImage(
                        image_id=f"{document_id}_page_{page_no:04d}",
                        document_id=document_id,
                        document_name=document_name,
                        page_number=page_no,
                        image_path=image_path,
                        image_bytes=image_bytes,
                        width=width,
                        height=height,
                    )
                )
            except Exception:
                # If page rendering fails, continue without the image.
                continue
        return pages
