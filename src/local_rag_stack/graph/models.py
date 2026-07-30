"""Pydantic models for the knowledge graph layer.

The schema is intentionally small and maps cleanly onto the Karpathy/Anthropic
multi-agent graph vocabulary: typed entities, directional relations, and
chunk-level provenance.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """Supported entity node types."""

    PERSON = "person"
    ORGANIZATION = "organization"  # agency, group, org, team, lab, community
    OBJECT = "object"  # machine, collaboration, asset, equipment
    LOCATION = "location"
    DOCUMENT = "document"  # contract, note, report, message
    CONCEPT = "concept"  # process, standard, technique
    METRIC = "metric"  # accuracy, loss, cost, latency, thickness
    ATTRIBUTE = "attribute"  # provider, age, size, length, area, status
    EVENT = "event"  # inspection, test, audit


class RelationType(str, Enum):
    """Supported relation edge types."""

    MENTIONS = "mentions"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    HAS_ATTRIBUTE = "has_attribute"
    HAS_METRIC = "has_metric"
    AUTHORED_BY = "authored_by"
    REFERENCES = "references"
    CONTAINS = "contains"
    RELATES_TO = "relates_to"
    OCCURRED_AT = "occurred_at"


class Entity(BaseModel):
    """A typed node in the knowledge graph."""

    id: str = Field(..., description="Stable UUID for this entity node.")
    name: str = Field(..., description="Canonical entity name or label.")
    entity_type: EntityType = Field(..., description="Entity type.")
    document_id: str = Field(..., description="Source document id.")
    chunk_id: str | None = Field(None, description="Source chunk id if known.")
    source_text: str = Field(..., description="Text span that grounded the entity.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relation(BaseModel):
    """A directional edge between two entity nodes."""

    id: str = Field(..., description="Stable UUID for this relation edge.")
    source_entity_id: str = Field(..., description="Tail entity id.")
    target_entity_id: str = Field(..., description="Head entity id.")
    relation_type: RelationType = Field(..., description="Relation type.")
    document_id: str = Field(..., description="Source document id.")
    chunk_id: str | None = Field(None, description="Source chunk id if known.")
    source_text: str = Field(..., description="Text span that grounded the relation.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphExtractionResult(BaseModel):
    """Structured output of the entity/relation extraction step."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class GraphSearchResult(BaseModel):
    """A graph neighborhood returned by the store."""

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
