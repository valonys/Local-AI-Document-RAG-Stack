"""Pydantic models for evaluator-optimizer experiments."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionOutcome(str, Enum):
    """Decision taken after evaluating an experiment."""

    KEEP = "keep"
    REVERT = "revert"


class ExperimentStatus(str, Enum):
    """Lifecycle status of an experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    DECIDED = "decided"


class StrategyConfig(BaseModel):
    """Ingestion/retrieval strategy applied to an experiment.

    Fields mirror ``Settings`` knobs where possible. ``chunk_size`` is recorded
    for the experiment but is not applied by ``RAGPipeline`` because chunking is
    configured at extraction time.
    """

    model_config = ConfigDict(extra="allow")

    extraction_engine: str | None = Field(default=None, description="docling, marker, or plaintext")
    chunk_size: int | None = Field(default=None, description="Target chunk size in tokens")
    chunk_overlap: int | None = Field(default=None, description="Chunk overlap in tokens")
    text_embed_model: str | None = Field(default=None, description="Text embedding model name")
    vision_embed_backend: str | None = Field(
        default=None, description="none, ollama, hf, or colqwen"
    )
    vision_embed_model: str | None = Field(default=None, description="Visual embedding model name")
    rerank_backend: str | None = Field(default=None, description="none, ollama, or st")
    rerank_model: str | None = Field(default=None, description="Reranker model name")
    llm_model: str | None = Field(default=None, description="Answer generation model")
    text_search_k: int | None = Field(default=None, description="Top-K text retrieval")
    final_top_k: int | None = Field(default=None, description="Top-K results sent to generator")


class EvalQuestion(BaseModel):
    """A single labeled evaluation question."""

    question_id: str | None = Field(default=None, description="Optional stable question id")
    question: str
    expected_answer: str | None = Field(default=None, description="Reference answer text")
    expected_document_ids: list[str] = Field(default_factory=list)
    k: int = Field(default=5, description="Cutoff for precision@K")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Result of evaluating one question."""

    experiment_id: str | None = None
    question_id: str | None = None
    question: str
    expected_answer: str | None = None
    expected_document_ids: list[str] = Field(default_factory=list)
    retrieved_document_ids: list[str] = Field(default_factory=list)
    precision_at_k: float = 0.0
    answer_correctness: float = 0.0
    generated_answer: str | None = None
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """Keep/revert decision for an experiment."""

    experiment_id: str
    outcome: DecisionOutcome
    reason: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Experiment(BaseModel):
    """An evaluator-optimizer experiment."""

    id: str
    name: str
    parent_id: str | None = None
    strategy_config: StrategyConfig = Field(default_factory=StrategyConfig)
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decided_at: datetime | None = None
    results: list[EvalResult] = Field(default_factory=list)
    decision: Decision | None = None
