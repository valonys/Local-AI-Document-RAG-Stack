"""Evaluate a RAGPipeline against a labeled eval set."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Protocol

from ..generation.generator import AnswerGenerator
from ..pipeline import RAGPipeline
from .models import EvalQuestion, EvalResult

logger = logging.getLogger(__name__)


class Judge(Protocol):
    """Callable that scores answer correctness between 0 and 1."""

    def __call__(self, question: str, expected: str | None, actual: str) -> float:
        ...


def exact_or_contains_judge(
    question: str, expected: str | None, actual: str, *, case_sensitive: bool = False
) -> float:
    """Simple judge: 1.0 if the expected answer is present in the actual answer.

    Falls back to exact match when ``expected`` is a single token or short phrase.
    """
    if expected is None or not expected.strip():
        return 0.0
    if not case_sensitive:
        expected_norm = expected.strip().lower()
        actual_norm = actual.strip().lower()
    else:
        expected_norm = expected.strip()
        actual_norm = actual.strip()
    if expected_norm == actual_norm or expected_norm in actual_norm:
        return 1.0
    return 0.0


def llm_judge(
    generator: AnswerGenerator,
    question: str,
    expected: str | None,
    actual: str,
) -> float:
    """Score answer correctness using an LLM via the answer generator.

    Returns 1.0 if the model responds with a clear "yes" or a numeric score >= 0.5.
    Returns 0.0 otherwise. This avoids adding a heavy dependency; it reuses the
    existing Ollama-backed ``AnswerGenerator``.
    """
    if expected is None or not expected.strip():
        return 0.0
    prompt = (
        "You are a strict evaluator. Respond with ONLY a number from 0 to 1 "
        "where 1 means the actual answer fully and correctly answers the question, "
        "and 0 means it is completely wrong or unsupported.\n\n"
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"Actual answer: {actual}\n\n"
        "Score (0-1):"
    )
    try:
        response = generator.generate(
            query=prompt,
            results=[],
            include_images=False,
        )
        text = response.answer.strip()
        # Extract the first numeric token.
        import re

        match = re.search(r"[0-9]*\.?[0-9]+", text)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))
        if "yes" in text.lower():
            return 1.0
        return 0.0
    except Exception as exc:
        logger.warning("LLM judge failed: %s", exc)
        return 0.0


class Evaluator:
    """Run a labeled eval set through a ``RAGPipeline`` and produce metrics."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        judge: Judge | None = None,
        llm_judge_generator: AnswerGenerator | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.judge = judge or exact_or_contains_judge
        self.llm_judge_generator = llm_judge_generator

    def evaluate(self, eval_set: list[EvalQuestion]) -> list[EvalResult]:
        """Evaluate every question and return one ``EvalResult`` per question."""
        return [self._evaluate_question(q) for q in eval_set]

    def _evaluate_question(self, question: EvalQuestion) -> EvalResult:
        start = time.perf_counter()
        response = self.pipeline.query(question.question, top_k=question.k, include_images=False)
        elapsed = time.perf_counter() - start

        retrieved_doc_ids = [r.document_id for r in response.retrieved[: question.k]]
        precision = self._precision_at_k(
            retrieved_doc_ids, question.expected_document_ids, question.k
        )

        generated = response.answer
        if self.llm_judge_generator is not None:
            correctness = llm_judge(
                self.llm_judge_generator,
                question.question,
                question.expected_answer,
                generated,
            )
        else:
            correctness = self.judge(question.question, question.expected_answer, generated)

        return EvalResult(
            experiment_id=None,
            question_id=question.question_id or str(uuid.uuid4()),
            question=question.question,
            expected_answer=question.expected_answer,
            expected_document_ids=question.expected_document_ids,
            retrieved_document_ids=retrieved_doc_ids,
            precision_at_k=precision,
            answer_correctness=correctness,
            generated_answer=generated,
            elapsed_seconds=elapsed,
            metadata=question.metadata,
        )

    @staticmethod
    def _precision_at_k(
        retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int
    ) -> float:
        """Compute document-level precision at K."""
        if not retrieved_doc_ids or k <= 0:
            return 0.0
        expected = set(expected_doc_ids)
        if not expected:
            return 0.0
        relevant_count = sum(1 for doc_id in retrieved_doc_ids[:k] if doc_id in expected)
        return relevant_count / k
