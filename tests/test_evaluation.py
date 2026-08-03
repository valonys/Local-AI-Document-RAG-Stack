"""Tests for the evaluator-optimizer scaffolding."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from local_rag_stack.cli import main
from local_rag_stack.evaluation.evaluator import (
    Evaluator,
    exact_or_contains_judge,
    llm_judge,
)
from local_rag_stack.evaluation.models import (
    DecisionOutcome,
    EvalQuestion,
    EvalResult,
    ExperimentStatus,
    StrategyConfig,
)
from local_rag_stack.evaluation.store import ExperimentStore
from local_rag_stack.models import Citation, Modality, QueryResponse, SearchResult


def _make_query_response(
    query: str,
    answer: str,
    retrieved: list[SearchResult],
) -> QueryResponse:
    return QueryResponse(
        query=query,
        answer=answer,
        citations=[
            Citation(
                source_id=r.id,
                document_name=r.document_name,
                page_number=r.page_number,
                modality=r.modality,
                snippet=(r.text or "")[:300],
                score=r.score,
            )
            for r in retrieved
        ],
        retrieved=retrieved,
        model="mock",
        elapsed_seconds=0.1,
    )


def test_evaluator_precision_at_k() -> None:
    """Evaluator should compute document-level precision@K correctly."""
    retrieved = [
        SearchResult(
            id="c1",
            document_id="doc-a",
            document_name="doc-a.pdf",
            modality=Modality.TEXT,
            text="relevant",
            score=0.9,
            rank=1,
        ),
        SearchResult(
            id="c2",
            document_id="doc-b",
            document_name="doc-b.pdf",
            modality=Modality.TEXT,
            text="irrelevant",
            score=0.8,
            rank=2,
        ),
    ]
    pipeline = MagicMock()
    pipeline.query.return_value = _make_query_response(
        "What is the date?",
        "The date is 2024-01-01.",
        retrieved,
    )

    evaluator = Evaluator(pipeline)
    results = evaluator.evaluate(
        [
            EvalQuestion(
                question_id="q1",
                question="What is the date?",
                expected_answer="2024-01-01",
                expected_document_ids=["doc-a"],
                k=2,
            )
        ]
    )

    assert len(results) == 1
    result = results[0]
    assert result.precision_at_k == pytest.approx(0.5)
    assert result.answer_correctness == 1.0
    assert result.retrieved_document_ids == ["doc-a", "doc-b"]


def test_evaluator_answer_correctness_partial() -> None:
    """Contains judge should score partial matches."""
    pipeline = MagicMock()
    pipeline.query.return_value = _make_query_response(
        "Who is responsible?",
        "Acme Corp is the data processor.",
        [
            SearchResult(
                id="c1",
                document_id="doc-a",
                document_name="doc-a.pdf",
                modality=Modality.TEXT,
                text="context",
                score=0.9,
                rank=1,
            )
        ],
    )

    evaluator = Evaluator(pipeline, judge=exact_or_contains_judge)
    results = evaluator.evaluate(
        [
            EvalQuestion(
                question="Who is responsible?",
                expected_answer="Acme Corp",
                expected_document_ids=["doc-a"],
                k=1,
            )
        ]
    )

    assert results[0].answer_correctness == 1.0


def test_llm_judge_extracts_score() -> None:
    """LLM judge should parse a numeric score from the generator answer."""
    generator = MagicMock()
    generator.generate.return_value = QueryResponse(
        query="score",
        answer="The answer is mostly correct, so I give it a score of 0.75.",
        citations=[],
        retrieved=[],
        model="mock",
        elapsed_seconds=0.1,
    )
    score = llm_judge(generator, "q", "expected", "actual")
    assert score == pytest.approx(0.75)


def test_experiment_store_roundtrip(tmp_path: Path) -> None:
    """ExperimentStore should persist experiments, results, and decisions."""
    db_path = tmp_path / "experiments.sqlite"
    store = ExperimentStore(db_path)
    try:
        strategy = StrategyConfig(
            extraction_engine="plaintext",
            chunk_size=256,
            text_embed_model="BAAI/bge-small-en-v1.5",
            vision_embed_backend="none",
        )
        experiment = store.create_experiment(
            name="text-only baseline",
            strategy_config=strategy,
        )
        assert experiment.name == "text-only baseline"
        assert experiment.status == ExperimentStatus.PENDING

        result = EvalResult(
            question_id="q1",
            question="What is the date?",
            expected_answer="2024-01-01",
            expected_document_ids=["doc-a"],
            retrieved_document_ids=["doc-a", "doc-b"],
            precision_at_k=0.5,
            answer_correctness=1.0,
            generated_answer="The date is 2024-01-01.",
            elapsed_seconds=0.42,
        )
        row_id = store.save_result(experiment.id, result)
        assert row_id is not None

        experiments = store.list_experiments()
        assert len(experiments) == 1
        assert experiments[0].id == experiment.id

        loaded = store.get_experiment(experiment.id)
        assert loaded is not None
        assert loaded.name == experiment.name
        assert loaded.strategy_config.extraction_engine == "plaintext"
        assert loaded.strategy_config.chunk_size == 256
        assert len(loaded.results) == 1
        assert loaded.results[0].precision_at_k == pytest.approx(0.5)

        decision = store.record_decision(
            experiment.id, DecisionOutcome.KEEP, reason="Baseline is good enough."
        )
        assert decision.outcome == DecisionOutcome.KEEP

        loaded_after = store.get_experiment(experiment.id)
        assert loaded_after is not None
        assert loaded_after.status == ExperimentStatus.DECIDED
        assert loaded_after.decision is not None
        assert loaded_after.decision.reason == "Baseline is good enough."
    finally:
        store.close()


def test_cli_experiment_list_and_show(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI experiment list/show should read from the store."""
    runner = CliRunner()

    fake_store = MagicMock()
    fake_experiment = MagicMock()
    fake_experiment.id = "exp-123"
    fake_experiment.name = "test experiment"
    fake_experiment.status.value = "completed"
    fake_experiment.created_at = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)
    fake_experiment.parent_id = None
    fake_experiment.strategy_config.model_dump_json.return_value = "{}"
    fake_experiment.decision = None
    fake_result = MagicMock()
    fake_result.question_id = "q1"
    fake_result.precision_at_k = 0.75
    fake_result.answer_correctness = 1.0
    fake_result.elapsed_seconds = 0.5
    fake_experiment.results = [fake_result]

    fake_store.list_experiments.return_value = [fake_experiment]
    fake_store.get_experiment.return_value = fake_experiment

    monkeypatch.setattr(
        "local_rag_stack.cli.ExperimentStore", lambda _path: fake_store
    )

    list_result = runner.invoke(main, ["experiment", "list"])
    assert list_result.exit_code == 0
    assert "exp-123" in list_result.output
    assert "test experiment" in list_result.output

    show_result = runner.invoke(main, ["experiment", "show", "exp-123"])
    assert show_result.exit_code == 0
    assert "exp-123" in show_result.output
    assert "test experiment" in show_result.output
    assert "0.750" in show_result.output

    fake_store.close.assert_called()
