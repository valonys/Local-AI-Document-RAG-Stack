"""Answer generation with citations."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from ..exceptions import GenerationError
from ..models import Citation, Modality, QueryResponse, SearchResult

logger = logging.getLogger(__name__)


def _get_generated_text(response: Any) -> str:
    """Extract text from Ollama generate or chat response."""
    if hasattr(response, "response"):
        return str(response.response)
    if hasattr(response, "message") and response.message:
        return str(getattr(response.message, "content", response.message))
    if isinstance(response, dict):
        return str(response.get("response", response.get("message", "")))
    return ""


SYSTEM_PROMPT = """You are a precise document assistant. Answer the user's question using only the provided context.
Rules:
- Cite every fact with a source number like [1], [2], etc.
- If the context does not contain the answer, say you don't know.
- Do not make up information not present in the context.
- Keep answers concise but complete.
"""


class AnswerGenerator:
    """Generate a cited answer from retrieved text chunks and page images."""

    def __init__(
        self,
        *,
        model: str = "qwen3-vl",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.system_prompt = system_prompt
        self._client_instance: Any | None = None

    def _get_client(self) -> Any:
        if self._client_instance is None:
            try:
                from ollama import Client
            except ImportError as exc:
                raise GenerationError(
                    "Ollama client not installed. "
                    "Install with: pip install 'local-rag-stack[ollama]'"
                ) from exc
            self._client_instance = Client(host=self.host, timeout=self.timeout)
        return self._client_instance

    def generate(
        self,
        query: str,
        results: list[SearchResult],
        *,
        include_images: bool = True,
    ) -> QueryResponse:
        if not results:
            return QueryResponse(
                query=query,
                answer="No relevant documents were found for your question.",
                citations=[],
                retrieved=[],
                model=self.model,
                elapsed_seconds=0.0,
            )

        import time
        start = time.perf_counter()

        context_text, images = self._build_context(results, include_images=include_images)
        prompt = f"{self.system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"

        try:
            if images and self._is_vision_capable():
                response = self._generate_with_images(prompt, images)
            else:
                response = self._generate_text(prompt)
            answer = _get_generated_text(response).strip()
        except Exception as exc:
            raise GenerationError(f"Answer generation failed: {exc}") from exc

        elapsed = time.perf_counter() - start
        citations = self._build_citations(results)

        return QueryResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved=results,
            model=self.model,
            elapsed_seconds=elapsed,
        )

    def _build_context(
        self,
        results: list[SearchResult],
        *,
        include_images: bool = True,
    ) -> tuple[str, list[Path]]:
        """Build a numbered context string and collect image paths."""
        parts: list[str] = []
        images: list[Path] = []
        for i, result in enumerate(results, start=1):
            source_line = f"[{i}] {result.document_name}"
            if result.page_number:
                source_line += f" (page {result.page_number})"
            if result.text:
                parts.append(f"{source_line}\n{result.text}\n")
            elif include_images and result.image_path:
                parts.append(f"{source_line}\n[See attached image {i}]\n")
                images.append(result.image_path)
        return "\n".join(parts), images

    def _generate_text(self, prompt: str) -> Any:
        client = self._get_client()
        return client.generate(
            model=self.model,
            prompt=prompt,
            system=self.system_prompt,
            options={"temperature": 0.2, "num_ctx": 8192},
        )

    def _generate_with_images(self, prompt: str, images: list[Path]) -> Any:
        client = self._get_client()
        # Ollama chat API supports base64 image URLs.
        content = [{"type": "text", "text": prompt}]
        for image_path in images[:3]:  # limit images to avoid huge payloads
            b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        return client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            options={"temperature": 0.2, "num_ctx": 8192},
        )

    def _is_vision_capable(self) -> bool:
        # Best-effort heuristic; users can override by choosing qwen3-vl explicitly.
        return any(name in self.model.lower() for name in ("vl", "vision", "llava", "bakllava"))

    @staticmethod
    def _build_citations(results: list[SearchResult]) -> list[Citation]:
        return [
            Citation(
                source_id=result.id,
                document_name=result.document_name,
                page_number=result.page_number,
                modality=result.modality,
                snippet=(result.text or "")[:300],
                score=result.score,
            )
            for result in results
        ]
