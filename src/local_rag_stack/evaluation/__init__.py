"""Evaluation scaffolding for the Local RAG Stack."""

from __future__ import annotations

from .evaluator import Evaluator, exact_or_contains_judge, llm_judge
from .models import Decision, DecisionOutcome, EvalQuestion, EvalResult, Experiment, StrategyConfig
from .store import ExperimentStore

__all__ = [
    "Decision",
    "DecisionOutcome",
    "EvalQuestion",
    "EvalResult",
    "Evaluator",
    "Experiment",
    "ExperimentStore",
    "StrategyConfig",
    "exact_or_contains_judge",
    "llm_judge",
]
