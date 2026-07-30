"""sqlite-vec based vector store."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec

from ..exceptions import StorageError
from ..models import EmbeddedItem, Modality, SearchResult
from .base import VectorStore

logger = logging.getLogger(__name__)


def _serialize(vector: list[float]) -> bytes:
    return np.array(vector, dtype=np.float32).tobytes()


def _deserialize(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()


class SQLiteVecStore(VectorStore):
    """Vector store backed by sqlite-vec.

    Stores text and visual items in separate virtual tables because their
    embedding dimensions may differ.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
        except Exception as exc:
            logger.warning("Could not load sqlite-vec extension: %s", exc)
            raise StorageError(
                "sqlite-vec extension is required. Install with: pip install sqlite-vec"
            ) from exc

        # Items table for metadata and raw content.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_name TEXT NOT NULL,
                modality TEXT NOT NULL,
                page_number INTEGER,
                text TEXT,
                image_path TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_document_id ON items(document_id)"
        )

        # Virtual vec tables will be created lazily once we know the dimensions.
        self._text_dim: int | None = None
        self._visual_dim: int | None = None
        self._conn.commit()

    def _ensure_vec_table(self, modality: Modality, dim: int) -> None:
        if modality == Modality.TEXT:
            if self._text_dim is None:
                self._text_dim = dim
                self._conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_text USING vec0(
                        item_id TEXT PRIMARY KEY,
                        embedding FLOAT[{dim}]
                    )
                    """
                )
            elif self._text_dim != dim:
                raise StorageError(
                    f"Text embedding dimension mismatch: {self._text_dim} != {dim}"
                )
        elif modality == Modality.VISUAL:
            if self._visual_dim is None:
                self._visual_dim = dim
                self._conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_visual USING vec0(
                        item_id TEXT PRIMARY KEY,
                        embedding FLOAT[{dim}]
                    )
                    """
                )
            elif self._visual_dim != dim:
                raise StorageError(
                    f"Visual embedding dimension mismatch: {self._visual_dim} != {dim}"
                )
        self._conn.commit()

    def _vec_table(self, modality: Modality) -> str:
        if modality == Modality.TEXT:
            return "vec_text"
        if modality == Modality.VISUAL:
            return "vec_visual"
        raise StorageError(f"Unsupported modality: {modality}")

    def add(self, items: list[EmbeddedItem]) -> None:
        if not items:
            return
        try:
            for item in items:
                self._ensure_vec_table(item.modality, len(item.embedding))
                table = self._vec_table(item.modality)
                # Upsert metadata.
                self._conn.execute(
                    """
                    INSERT INTO items (id, document_id, document_name, modality, page_number, text, image_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        document_id=excluded.document_id,
                        document_name=excluded.document_name,
                        modality=excluded.modality,
                        page_number=excluded.page_number,
                        text=excluded.text,
                        image_path=excluded.image_path,
                        metadata=excluded.metadata
                    """,
                    (
                        item.id,
                        item.document_id,
                        item.document_name,
                        item.modality.value,
                        item.page_number,
                        item.text,
                        str(item.image_path) if item.image_path else None,
                        json.dumps(item.metadata),
                    ),
                )
                # Virtual tables do not support UPSERT or REPLACE; delete first.
                self._conn.execute(f"DELETE FROM {table} WHERE item_id = ?", (item.id,))
                self._conn.execute(
                    f"""
                    INSERT INTO {table} (item_id, embedding)
                    VALUES (?, ?)
                    """,
                    (item.id, _serialize(item.embedding)),
                )
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            raise StorageError(f"Failed to add items: {exc}") from exc

    def search(
        self,
        query_vector: list[float],
        *,
        modality: str | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        modalities = []
        if modality:
            try:
                modalities = [Modality(modality)]
            except ValueError as exc:
                raise StorageError(f"Invalid modality: {modality}") from exc
        else:
            modalities = [Modality.TEXT, Modality.VISUAL]

        query_blob = _serialize(query_vector)
        results: list[SearchResult] = []
        for mod in modalities:
            table = self._vec_table(mod)
            rows = self._conn.execute(
                f"""
                SELECT v.item_id, distance
                FROM {table} AS v
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (query_blob, top_k),
            ).fetchall()
            for rank, row in enumerate(rows, start=1):
                item_row = self._conn.execute(
                    "SELECT * FROM items WHERE id = ?", (row["item_id"],)
                ).fetchone()
                if not item_row:
                    continue
                results.append(
                    SearchResult(
                        id=item_row["id"],
                        document_id=item_row["document_id"],
                        document_name=item_row["document_name"],
                        modality=Modality(item_row["modality"]),
                        page_number=item_row["page_number"],
                        text=item_row["text"],
                        image_path=Path(item_row["image_path"]) if item_row["image_path"] else None,
                        score=1.0 / (1.0 + row["distance"]),
                        rank=rank,
                        metadata=json.loads(item_row["metadata"] or "{}"),
                    )
                )
        # Sort by score descending and slice.
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def delete_document(self, document_id: str) -> None:
        try:
            rows = self._conn.execute(
                "SELECT id, modality FROM items WHERE document_id = ?", (document_id,)
            ).fetchall()
            for row in rows:
                table = self._vec_table(Modality(row["modality"]))
                self._conn.execute(f"DELETE FROM {table} WHERE item_id = ?", (row["id"],))
            self._conn.execute("DELETE FROM items WHERE document_id = ?", (document_id,))
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            raise StorageError(f"Failed to delete document {document_id}: {exc}") from exc

    def list_documents(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT document_id, document_name, modality, COUNT(*) as count
            FROM items
            GROUP BY document_id, modality
            """
        ).fetchall()
        docs: dict[str, dict[str, Any]] = {}
        for row in rows:
            doc_id = row["document_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_name": row["document_name"],
                    "chunk_count": 0,
                    "page_count": 0,
                }
            if row["modality"] == Modality.TEXT.value:
                docs[doc_id]["chunk_count"] = row["count"]
            elif row["modality"] == Modality.VISUAL.value:
                docs[doc_id]["page_count"] = row["count"]
        return docs

    def close(self) -> None:
        self._conn.close()

    def health(self) -> dict[str, Any]:
        try:
            count = self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            return {"status": "ok", "items": count, "path": str(self.db_path)}
        except Exception as exc:
            return {"status": "error", "error": str(exc)}
