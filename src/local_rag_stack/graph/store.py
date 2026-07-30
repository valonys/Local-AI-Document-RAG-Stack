"""SQLite-backed graph store for typed entities and relations.

A plain SQLite database is used so the stack remains self-contained and
deployable on the local appliance without an additional graph server.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..exceptions import StorageError
from .models import Entity, EntityType, GraphSearchResult, Relation, RelationType

logger = logging.getLogger(__name__)


class GraphStore(ABC):
    """Abstract graph store."""

    @abstractmethod
    def add(self, entities: list[Entity], relations: list[Relation]) -> None:
        """Upsert entities and relations into the graph."""

    @abstractmethod
    def search(
        self,
        query: str | None = None,
        *,
        entity_types: list[EntityType] | None = None,
        document_id: str | None = None,
        entity_name: str | None = None,
        relation_type: RelationType | None = None,
        top_k: int = 20,
    ) -> GraphSearchResult:
        """Return entities and relations matching the filters."""

    @abstractmethod
    def get_neighborhood(
        self,
        entity_id: str,
        *,
        hops: int = 1,
    ) -> GraphSearchResult:
        """Return the n-hop neighborhood around an entity."""

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """Remove all graph nodes and edges derived from a document."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""


class SQLiteGraphStore(GraphStore):
    """Graph store backed by a single SQLite file."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT,
                source_text TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_document_id ON entities(document_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)"
        )

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                document_id TEXT NOT NULL,
                chunk_id TEXT,
                source_text TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
                FOREIGN KEY (target_entity_id) REFERENCES entities(id) ON DELETE CASCADE
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relations_document_id ON relations(document_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_entity_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_entity_id)"
        )
        self._conn.commit()

    def add(self, entities: list[Entity], relations: list[Relation]) -> None:
        if not entities and not relations:
            return
        try:
            for entity in entities:
                self._conn.execute(
                    """
                    INSERT INTO entities (id, name, entity_type, document_id, chunk_id, source_text, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name,
                        entity_type=excluded.entity_type,
                        document_id=excluded.document_id,
                        chunk_id=excluded.chunk_id,
                        source_text=excluded.source_text,
                        metadata=excluded.metadata
                    """,
                    (
                        entity.id,
                        entity.name,
                        entity.entity_type.value,
                        entity.document_id,
                        entity.chunk_id,
                        entity.source_text,
                        json.dumps(entity.metadata),
                    ),
                )
            for relation in relations:
                self._conn.execute(
                    """
                    INSERT INTO relations (id, source_entity_id, target_entity_id, relation_type,
                                           document_id, chunk_id, source_text, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        source_entity_id=excluded.source_entity_id,
                        target_entity_id=excluded.target_entity_id,
                        relation_type=excluded.relation_type,
                        document_id=excluded.document_id,
                        chunk_id=excluded.chunk_id,
                        source_text=excluded.source_text,
                        metadata=excluded.metadata
                    """,
                    (
                        relation.id,
                        relation.source_entity_id,
                        relation.target_entity_id,
                        relation.relation_type.value,
                        relation.document_id,
                        relation.chunk_id,
                        relation.source_text,
                        json.dumps(relation.metadata),
                    ),
                )
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            raise StorageError(f"Failed to add graph data: {exc}") from exc

    def search(
        self,
        query: str | None = None,
        *,
        entity_types: list[EntityType] | None = None,
        document_id: str | None = None,
        entity_name: str | None = None,
        relation_type: RelationType | None = None,
        top_k: int = 20,
    ) -> GraphSearchResult:
        try:
            entity_rows = self._fetch_entities(
                query=query,
                entity_types=entity_types,
                document_id=document_id,
                entity_name=entity_name,
                limit=top_k,
            )
            entity_ids = {row["id"] for row in entity_rows}

            relation_rows: list[sqlite3.Row] = []
            if entity_ids:
                relation_rows = self._fetch_relations(
                    entity_ids=entity_ids,
                    relation_type=relation_type,
                    document_id=document_id,
                    limit=top_k * 2,
                )
                for row in relation_rows:
                    entity_ids.add(row["source_entity_id"])
                    entity_ids.add(row["target_entity_id"])

            # Make sure endpoints are materialised even if they did not match the entity query.
            if entity_ids:
                placeholders = ",".join("?" * len(entity_ids))
                endpoint_rows = self._conn.execute(
                    f"SELECT * FROM entities WHERE id IN ({placeholders})",
                    tuple(entity_ids),
                ).fetchall()
                endpoint_ids = {row["id"] for row in endpoint_rows}
                for row in endpoint_rows:
                    if row["id"] not in {r["id"] for r in entity_rows}:
                        entity_rows.append(row)
                # Drop dangling relations whose endpoints are missing.
                relation_rows = [
                    row
                    for row in relation_rows
                    if row["source_entity_id"] in endpoint_ids and row["target_entity_id"] in endpoint_ids
                ]

            return GraphSearchResult(
                entities=[self._row_to_entity(row) for row in entity_rows],
                relations=[self._row_to_relation(row) for row in relation_rows],
            )
        except Exception as exc:
            raise StorageError(f"Graph search failed: {exc}") from exc

    def _fetch_entities(
        self,
        query: str | None,
        entity_types: list[EntityType] | None,
        document_id: str | None,
        entity_name: str | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        conditions: list[str] = []
        params: list[Any] = []
        if query:
            conditions.append("(name LIKE ? OR source_text LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like])
        if entity_types:
            placeholders = ",".join("?" * len(entity_types))
            conditions.append(f"entity_type IN ({placeholders})")
            params.extend([t.value for t in entity_types])
        if document_id:
            conditions.append("document_id = ?")
            params.append(document_id)
        if entity_name:
            conditions.append("name LIKE ?")
            params.append(f"%{entity_name}%")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM entities {where} ORDER BY name LIMIT ?"
        params.append(limit)
        return self._conn.execute(sql, tuple(params)).fetchall()

    def _fetch_relations(
        self,
        entity_ids: set[str],
        relation_type: RelationType | None,
        document_id: str | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        placeholders = ",".join("?" * len(entity_ids))
        conditions = [f"(source_entity_id IN ({placeholders}) OR target_entity_id IN ({placeholders}))"]
        params: list[Any] = list(entity_ids) + list(entity_ids)
        if relation_type:
            conditions.append("relation_type = ?")
            params.append(relation_type.value)
        if document_id:
            conditions.append("document_id = ?")
            params.append(document_id)
        where = f"WHERE {' AND '.join(conditions)}"
        sql = f"SELECT * FROM relations {where} ORDER BY relation_type LIMIT ?"
        params.append(limit)
        return self._conn.execute(sql, tuple(params)).fetchall()

    def get_neighborhood(
        self,
        entity_id: str,
        *,
        hops: int = 1,
    ) -> GraphSearchResult:
        if hops < 1:
            hops = 1
        try:
            frontier = {entity_id}
            seen = set(frontier)
            relation_rows: list[sqlite3.Row] = []
            for _ in range(hops):
                if not frontier:
                    break
                placeholders = ",".join("?" * len(frontier))
                rows = self._conn.execute(
                    f"""
                    SELECT * FROM relations
                    WHERE source_entity_id IN ({placeholders}) OR target_entity_id IN ({placeholders})
                    """,
                    tuple(frontier) + tuple(frontier),
                ).fetchall()
                frontier = set()
                for row in rows:
                    if row["id"] in {r["id"] for r in relation_rows}:
                        continue
                    relation_rows.append(row)
                    for col in ("source_entity_id", "target_entity_id"):
                        if row[col] not in seen:
                            seen.add(row[col])
                            frontier.add(row[col])

            if not seen:
                return GraphSearchResult()

            placeholders = ",".join("?" * len(seen))
            entity_rows = self._conn.execute(
                f"SELECT * FROM entities WHERE id IN ({placeholders})",
                tuple(seen),
            ).fetchall()
            return GraphSearchResult(
                entities=[self._row_to_entity(row) for row in entity_rows],
                relations=[self._row_to_relation(row) for row in relation_rows],
            )
        except Exception as exc:
            raise StorageError(f"Graph neighborhood failed: {exc}") from exc

    def delete_document(self, document_id: str) -> None:
        try:
            self._conn.execute("DELETE FROM relations WHERE document_id = ?", (document_id,))
            self._conn.execute("DELETE FROM entities WHERE document_id = ?", (document_id,))
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            raise StorageError(f"Failed to delete graph document {document_id}: {exc}") from exc

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            document_id=row["document_id"],
            chunk_id=row["chunk_id"],
            source_text=row["source_text"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _row_to_relation(row: sqlite3.Row) -> Relation:
        return Relation(
            id=row["id"],
            source_entity_id=row["source_entity_id"],
            target_entity_id=row["target_entity_id"],
            relation_type=RelationType(row["relation_type"]),
            document_id=row["document_id"],
            chunk_id=row["chunk_id"],
            source_text=row["source_text"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
        )
