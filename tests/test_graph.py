"""Tests for the knowledge graph layer."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from local_rag_stack.graph.extractor import OllamaGraphExtractor, _RawExtraction
from local_rag_stack.graph.models import Entity, EntityType, GraphExtractionResult, Relation, RelationType
from local_rag_stack.graph.store import SQLiteGraphStore


def test_canonicalise_assigns_stable_ids() -> None:
    """Canonicalisation should generate deterministic UUIDs from raw extraction."""
    raw = _RawExtraction(
        entities=[
            _RawExtraction.model_fields["entities"].annotation.__args__[0](
                name="BDV 2088 B", entity_type=EntityType.OBJECT, source_text="equipment BDV 2088 B"
            ),
            _RawExtraction.model_fields["entities"].annotation.__args__[0](
                name="G. Pitavin", entity_type=EntityType.PERSON, source_text="Inspector G. Pitavin"
            ),
        ],
        relations=[
            _RawExtraction.model_fields["relations"].annotation.__args__[0](
                source_name="G. Pitavin",
                target_name="BDV 2088 B",
                relation_type=RelationType.AUTHORED_BY,
                source_text="Inspector G. Pitavin reviewed BDV 2088 B",
            )
        ],
    )
    extractor = OllamaGraphExtractor.__new__(OllamaGraphExtractor)
    result = extractor._canonicalise(raw, document_id="doc1", chunk_id="c1")

    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert result.entities[0].name == "BDV 2088 B"
    assert result.entities[0].entity_type == EntityType.OBJECT
    # Re-running with the same inputs should produce the same IDs.
    result2 = extractor._canonicalise(raw, document_id="doc1", chunk_id="c1")
    assert result.entities[0].id == result2.entities[0].id
    assert result.relations[0].id == result2.relations[0].id


def test_sqlite_graph_store_roundtrip(tmp_path: Path) -> None:
    """Entities and relations should be persisted and queryable."""
    db_path = tmp_path / "graph.sqlite"
    store = SQLiteGraphStore(db_path)

    entity_a = Entity(
        id=str(uuid.uuid4()),
        name="BDV 2088 B",
        entity_type=EntityType.OBJECT,
        document_id="doc1",
        chunk_id="c1",
        source_text="equipment BDV 2088 B",
    )
    entity_b = Entity(
        id=str(uuid.uuid4()),
        name="2.7 mm",
        entity_type=EntityType.METRIC,
        document_id="doc1",
        chunk_id="c1",
        source_text="wall thickness 2.7 mm",
    )
    relation = Relation(
        id=str(uuid.uuid4()),
        source_entity_id=entity_a.id,
        target_entity_id=entity_b.id,
        relation_type=RelationType.HAS_METRIC,
        document_id="doc1",
        chunk_id="c1",
        source_text="BDV 2088 B has wall thickness 2.7 mm",
    )

    store.add([entity_a, entity_b], [relation])

    result = store.search(query="BDV")
    assert any(e.name == "BDV 2088 B" for e in result.entities)

    result = store.search(entity_types=[EntityType.METRIC])
    assert any(e.name == "2.7 mm" for e in result.entities)

    result = store.search(relation_type=RelationType.HAS_METRIC)
    assert any(r.source_entity_id == entity_a.id for r in result.relations)

    neighborhood = store.get_neighborhood(entity_a.id, hops=1)
    assert any(e.id == entity_b.id for e in neighborhood.entities)
    assert any(r.id == relation.id for r in neighborhood.relations)

    store.delete_document("doc1")
    result = store.search(query="BDV")
    assert not result.entities
    store.close()
