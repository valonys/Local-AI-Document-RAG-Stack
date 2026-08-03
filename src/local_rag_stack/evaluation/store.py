"""SQLite-backed store for experiments, eval results, and keep/revert decisions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    Decision,
    DecisionOutcome,
    EvalResult,
    Experiment,
    ExperimentStatus,
    StrategyConfig,
)


class ExperimentStore:
    """Persistent store for evaluator-optimizer experiments.

    Uses a dedicated SQLite database (not the graph store) so eval metadata can
    be inspected without touching production vector/graph data.
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                parent_id TEXT,
                strategy_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                decided_at TEXT
            );

            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                question_id TEXT,
                question TEXT NOT NULL,
                expected_answer TEXT,
                expected_document_ids TEXT NOT NULL,
                retrieved_document_ids TEXT NOT NULL,
                precision_at_k REAL NOT NULL,
                answer_correctness REAL NOT NULL,
                generated_answer TEXT,
                elapsed_seconds REAL NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );

            CREATE TABLE IF NOT EXISTS decisions (
                experiment_id TEXT PRIMARY KEY,
                outcome TEXT NOT NULL,
                reason TEXT,
                decided_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            );
            """
        )
        self._conn.commit()

    def create_experiment(
        self,
        name: str,
        strategy_config: StrategyConfig | None = None,
        parent_id: str | None = None,
    ) -> Experiment:
        """Create a new experiment record."""
        experiment = Experiment(
            id=str(uuid.uuid4()),
            name=name,
            parent_id=parent_id,
            strategy_config=strategy_config or StrategyConfig(),
            status=ExperimentStatus.PENDING,
        )
        self._conn.execute(
            """
            INSERT INTO experiments (id, name, parent_id, strategy_json, status, created_at, decided_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment.id,
                experiment.name,
                experiment.parent_id,
                experiment.strategy_config.model_dump_json(),
                experiment.status.value,
                experiment.created_at.isoformat(),
                experiment.decided_at.isoformat() if experiment.decided_at else None,
            ),
        )
        self._conn.commit()
        return experiment

    def save_result(self, experiment_id: str, result: EvalResult) -> int:
        """Persist one eval result and return its row id."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            """
            INSERT INTO eval_results (
                experiment_id, question_id, question, expected_answer,
                expected_document_ids, retrieved_document_ids, precision_at_k,
                answer_correctness, generated_answer, elapsed_seconds, metadata, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                result.question_id,
                result.question,
                result.expected_answer,
                json.dumps(result.expected_document_ids),
                json.dumps(result.retrieved_document_ids),
                result.precision_at_k,
                result.answer_correctness,
                result.generated_answer,
                result.elapsed_seconds,
                json.dumps(result.metadata),
                now,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    def update_status(self, experiment_id: str, status: ExperimentStatus) -> None:
        """Update the lifecycle status of an experiment."""
        self._conn.execute(
            "UPDATE experiments SET status = ? WHERE id = ?",
            (status.value, experiment_id),
        )
        self._conn.commit()

    def record_decision(
        self, experiment_id: str, outcome: DecisionOutcome, reason: str | None = None
    ) -> Decision:
        """Record a keep/revert decision and update the experiment status."""
        decision = Decision(experiment_id=experiment_id, outcome=outcome, reason=reason)
        self._conn.execute(
            """
            INSERT INTO decisions (experiment_id, outcome, reason, decided_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                outcome=excluded.outcome,
                reason=excluded.reason,
                decided_at=excluded.decided_at
            """,
            (
                decision.experiment_id,
                decision.outcome.value,
                decision.reason,
                decision.decided_at.isoformat(),
            ),
        )
        self._conn.execute(
            """
            UPDATE experiments
            SET status = ?, decided_at = ?
            WHERE id = ?
            """,
            (ExperimentStatus.DECIDED.value, decision.decided_at.isoformat(), experiment_id),
        )
        self._conn.commit()
        return decision

    def list_experiments(self) -> list[Experiment]:
        """Return all experiments ordered by creation time."""
        rows = self._conn.execute(
            "SELECT * FROM experiments ORDER BY created_at"
        ).fetchall()
        return [self._row_to_experiment(row) for row in rows]

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Fetch one experiment including its eval results and decision."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        experiment = self._row_to_experiment(row)
        experiment.results = self._load_results(experiment_id)
        experiment.decision = self._load_decision(experiment_id)
        return experiment

    def _load_results(self, experiment_id: str) -> list[EvalResult]:
        rows = self._conn.execute(
            "SELECT * FROM eval_results WHERE experiment_id = ? ORDER BY id",
            (experiment_id,),
        ).fetchall()
        return [self._row_to_result(row) for row in rows]

    def _load_decision(self, experiment_id: str) -> Decision | None:
        row = self._conn.execute(
            "SELECT * FROM decisions WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        return Decision(
            experiment_id=row["experiment_id"],
            outcome=DecisionOutcome(row["outcome"]),
            reason=row["reason"],
            decided_at=datetime.fromisoformat(row["decided_at"]),
        )

    def _row_to_experiment(self, row: sqlite3.Row) -> Experiment:
        return Experiment(
            id=row["id"],
            name=row["name"],
            parent_id=row["parent_id"],
            strategy_config=StrategyConfig.model_validate_json(row["strategy_json"]),
            status=ExperimentStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            decided_at=datetime.fromisoformat(row["decided_at"]) if row["decided_at"] else None,
        )

    def _row_to_result(self, row: sqlite3.Row) -> EvalResult:
        return EvalResult(
            experiment_id=row["experiment_id"],
            question_id=row["question_id"],
            question=row["question"],
            expected_answer=row["expected_answer"],
            expected_document_ids=json.loads(row["expected_document_ids"]),
            retrieved_document_ids=json.loads(row["retrieved_document_ids"]),
            precision_at_k=row["precision_at_k"],
            answer_correctness=row["answer_correctness"],
            generated_answer=row["generated_answer"],
            elapsed_seconds=row["elapsed_seconds"],
            metadata=json.loads(row["metadata"]),
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ExperimentStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
