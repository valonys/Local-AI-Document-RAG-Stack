"""Visual / multimodal embedding implementations."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from ..exceptions import EmbeddingError
from ..models import EmbeddedItem, Modality, MultiVectorItem, PageImage
from .base import VisualEmbedder
from .text_embedder import _get_embeddings

logger = logging.getLogger(__name__)


def _load_image_bytes(page: PageImage) -> bytes:
    if page.image_bytes:
        return page.image_bytes
    if page.image_path:
        return Path(page.image_path).read_bytes()
    raise EmbeddingError(f"Page {page.image_id} has no image data")


def _load_image(page: PageImage) -> Image.Image:
    data = _load_image_bytes(page)
    return Image.open(io.BytesIO(data)).convert("RGB")


class OllamaVisualEmbedder(VisualEmbedder):
    """Visual embeddings via Ollama (qwen3-vl or similar VLM)."""

    def __init__(
        self,
        *,
        model: str = "qwen3-vl",
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
                raise EmbeddingError(
                    "Ollama client not installed. "
                    "Install with: pip install 'local-rag-stack[ollama]'"
                ) from exc
            self._client_instance = Client(host=self.host, timeout=self.timeout)
        return self._client_instance

    def embed_pages(
        self,
        pages: list[PageImage],
        *,
        batch_size: int = 8,
    ) -> list[EmbeddedItem]:
        if not pages:
            return []
        client = self._get_client()
        items: list[EmbeddedItem] = []
        # Ollama embed endpoint for images accepts base64 strings in some versions.
        # We call one image at a time for compatibility.
        for page in pages:
            try:
                import base64
                b64 = base64.b64encode(_load_image_bytes(page)).decode("utf-8")
                # Try the Ollama embed endpoint with image input.
                response = client.embed(
                    model=self.model,
                    input=f"data:image/png;base64,{b64}",
                )
                vectors = _get_embeddings(response)
                if not vectors:
                    logger.warning("No embedding returned for page %s", page.image_id)
                    continue
                items.append(
                    EmbeddedItem(
                        id=page.image_id,
                        document_id=page.document_id,
                        document_name=page.document_name,
                        modality=Modality.VISUAL,
                        page_number=page.page_number,
                        image_path=page.image_path,
                        image_bytes=page.image_bytes,
                        embedding=vectors[0],
                        metadata={"width": page.width, "height": page.height},
                    )
                )
            except Exception as exc:
                logger.warning("Failed to embed page %s: %s", page.image_id, exc)
                continue
        return items

    def embed_query(self, query: str) -> list[float]:
        client = self._get_client()
        try:
            response = client.embed(model=self.model, input=[query])
        except Exception as exc:
            raise EmbeddingError(f"Ollama visual query embedding failed: {exc}") from exc
        vectors = _get_embeddings(response)
        if not vectors:
            raise EmbeddingError("Ollama returned no query embedding")
        return vectors[0]


class ColQwenEmbedder:
    """Visual document embeddings via ColQwen2 / ColQwen2.5.

    Unlike :class:`OllamaVisualEmbedder`, this produces *multi-vector*
    late-interaction embeddings. Use it with :class:`MultiVectorStore`.
    """

    def __init__(
        self,
        *,
        model: str = "vidore/colqwen2-v1.0",
        device: str | None = None,
    ) -> None:
        self.model_name = model
        self.device = device or ("mps" if _is_mps_available() else "cpu")
        self._model_instance: Any | None = None
        self._processor: Any | None = None

    def _get_model(self) -> tuple[Any, Any]:
        if self._model_instance is None:
            try:
                from colpali_engine.models import ColQwen2, ColQwen2Processor
            except ImportError as exc:
                raise EmbeddingError(
                    "colpali-engine not installed. "
                    "Install with: pip install 'local-rag-stack[visual]'"
                ) from exc
            # device_map="auto" lets Accelerate place the model on the best
            # available device (MPS, CUDA, CPU) and loads orders of magnitude
            # faster than device_map="mps" on Apple Silicon.
            self._model_instance = ColQwen2.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto",
            ).eval()
            self._processor = ColQwen2Processor.from_pretrained(self.model_name)
        return self._model_instance, self._processor

    def embed_pages(
        self,
        pages: list[PageImage],
        *,
        batch_size: int = 4,
    ) -> list[MultiVectorItem]:
        if not pages:
            return []
        model, processor = self._get_model()
        try:
            import torch
            images = [_load_image(p) for p in pages]
            items: list[MultiVectorItem] = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_pages = pages[i : i + batch_size]
                inputs = processor.process_images(batch_images).to(model.device)
                with torch.no_grad():
                    embeddings = model(**inputs)
                for page, emb in zip(batch_pages, embeddings):
                    # emb shape: [n_patches, dim]
                    items.append(
                        MultiVectorItem(
                            id=page.image_id,
                            document_id=page.document_id,
                            document_name=page.document_name,
                            modality=Modality.VISUAL,
                            page_number=page.page_number,
                            image_path=page.image_path,
                            image_bytes=page.image_bytes,
                            embeddings=emb.cpu().float().tolist(),
                            metadata={"width": page.width, "height": page.height},
                        )
                    )
            return items
        except Exception as exc:
            raise EmbeddingError(f"ColQwen embedding failed: {exc}") from exc

    def embed_query(self, query: str) -> list[list[float]]:
        """Return query token vectors (matrix) for late-interaction search."""
        model, processor = self._get_model()
        try:
            import torch
            inputs = processor.process_queries([query]).to(model.device)
            with torch.no_grad():
                embedding = model(**inputs)
            # embedding shape: [n_query_tokens, dim]
            return embedding[0].cpu().float().tolist()
        except Exception as exc:
            raise EmbeddingError(f"ColQwen query embedding failed: {exc}") from exc


def _is_mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _get_torch_dtype() -> Any:
    try:
        import torch
        if _is_mps_available():
            return torch.float16
        return torch.float32
    except Exception:
        return None
