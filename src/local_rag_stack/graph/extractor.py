"""Entity/relation extraction using structured LLM outputs.

The extractor calls a local Ollama model and constrains its response to a JSON
schema.  Returned entities and relations are then canonicalised into the graph
model types with stable IDs.
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ..exceptions import GenerationError
from .models import Entity, EntityType, GraphExtractionResult, Relation, RelationType

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a precise information-extraction assistant.

Read the document chunk below and extract a small knowledge graph.

Rules:
- Extract concrete, named entities (people, organizations, equipment/assets, locations, documents, concepts, metrics, attributes, events).
- Do not invent entities not supported by the text.
- Use the exact names or identifiers as they appear in the text.
- Link entities with relations that are directly supported by the text.
- Prefer specific relation types; use "relates_to" only when no other type fits.
- The same real-world entity may appear multiple times; use the same canonical name.
- Keep the graph compact and accurate; quality over quantity.
"""


class _ExtractedEntity(BaseModel):
    """Raw entity emitted by the LLM before ID assignment."""

    name: str = Field(..., description="Canonical entity name or identifier.")
    entity_type: EntityType = Field(..., description="Entity type.")
    source_text: str = Field(..., description="Short text span supporting the entity.")


class _ExtractedRelation(BaseModel):
    """Raw relation emitted by the LLM before ID assignment."""

    source_name: str = Field(..., description="Tail entity name.")
    target_name: str = Field(..., description="Head entity name.")
    relation_type: RelationType = Field(..., description="Relation type.")
    source_text: str = Field(..., description="Short text span supporting the relation.")


class _RawExtraction(BaseModel):
    """Schema used to constrain the LLM response."""

    entities: list[_ExtractedEntity] = Field(default_factory=list)
    relations: list[_ExtractedRelation] = Field(default_factory=list)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema()


class GraphExtractor(ABC):
    """Abstract graph extraction interface."""

    @abstractmethod
    def extract(
        self,
        text: str,
        *,
        document_id: str,
        document_name: str,
        chunk_id: str | None = None,
    ) -> GraphExtractionResult:
        """Extract entities and relations from a text chunk."""


class OllamaGraphExtractor(GraphExtractor):
    """Structured-output graph extractor backed by Ollama."""

    def __init__(
        self,
        model: str = "qwen3.6:latest",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        temperature: float = 0.2,
        num_ctx: int = 8192,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from ollama import Client
            except ImportError as exc:
                raise GenerationError(
                    "Ollama client not installed. Install with: pip install 'local-rag-stack[ollama]'"
                ) from exc
            self._client = Client(host=self.host, timeout=self.timeout)
        return self._client

    def extract(
        self,
        text: str,
        *,
        document_id: str,
        document_name: str,
        chunk_id: str | None = None,
    ) -> GraphExtractionResult:
        prompt = self._build_prompt(text, document_name)
        raw = self._call_llm(prompt)
        return self._canonicalise(raw, document_id=document_id, chunk_id=chunk_id)

    def _build_prompt(self, text: str, document_name: str) -> str:
        return (
            f"{EXTRACTION_PROMPT}\n\n"
            f"Document: {document_name}\n\n"
            f"Chunk:\n{text}\n\n"
            "Respond with a JSON object containing 'entities' and 'relations' arrays."
        )

    def _call_llm(self, prompt: str) -> _RawExtraction:
        client = self._get_client()
        schema = _RawExtraction.json_schema()
        try:
            response = client.generate(
                model=self.model,
                prompt=prompt,
                system=EXTRACTION_PROMPT,
                format=schema,
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx,
                },
            )
        except Exception as exc:
            raise GenerationError(f"Graph extraction LLM call failed: {exc}") from exc

        answer = self._get_text(response).strip()
        if not answer:
            logger.warning("Graph extractor returned empty response")
            return _RawExtraction()

        try:
            data = json.loads(answer)
        except json.JSONDecodeError as exc:
            logger.warning("Graph extractor returned invalid JSON: %s\n%s", exc, answer[:500])
            return _RawExtraction()

        try:
            return _RawExtraction.model_validate(data)
        except Exception as exc:
            logger.warning("Graph extractor JSON did not match schema: %s\n%s", exc, answer[:500])
            return _RawExtraction()

    @staticmethod
    def _get_text(response: Any) -> str:
        """Extract generated text from Ollama response.

        Some reasoning models (e.g. Qwen3) emit structured output inside the
        ``thinking`` field while leaving ``response`` empty.
        """
        if hasattr(response, "response"):
            text = str(response.response or "")
            if text:
                return text
        if hasattr(response, "thinking"):
            thinking = str(response.thinking or "")
            if thinking:
                return thinking
        if isinstance(response, dict):
            text = str(response.get("response") or "")
            if text:
                return text
            text = str(response.get("thinking") or "")
            if text:
                return text
        return ""

    @staticmethod
    def _canonicalise(
        raw: _RawExtraction,
        *,
        document_id: str,
        chunk_id: str | None,
    ) -> GraphExtractionResult:
        """Convert raw extraction into canonical Entity/Relation models with stable IDs."""

        entity_map: dict[str, Entity] = {}
        entities: list[Entity] = []

        for e in raw.entities:
            key = e.name.strip().lower()
            if key in entity_map:
                continue
            entity = Entity(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"entity:{document_id}:{e.name}:{e.entity_type.value}")),
                name=e.name.strip(),
                entity_type=e.entity_type,
                document_id=document_id,
                chunk_id=chunk_id,
                source_text=e.source_text.strip(),
            )
            entity_map[key] = entity
            entities.append(entity)

        relations: list[Relation] = []
        for r in raw.relations:
            source_key = r.source_name.strip().lower()
            target_key = r.target_name.strip().lower()
            source_entity = entity_map.get(source_key)
            target_entity = entity_map.get(target_key)
            if source_entity is None or target_entity is None:
                # One of the endpoints was not extracted; skip.
                continue
            relation = Relation(
                id=str(uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"relation:{document_id}:{source_entity.id}:{target_entity.id}:{r.relation_type.value}",
                )),
                source_entity_id=source_entity.id,
                target_entity_id=target_entity.id,
                relation_type=r.relation_type,
                document_id=document_id,
                chunk_id=chunk_id,
                source_text=r.source_text.strip(),
            )
            relations.append(relation)

        return GraphExtractionResult(entities=entities, relations=relations)
