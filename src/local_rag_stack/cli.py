"""Command-line interface for the Local RAG Stack."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from .config import Settings
from .evaluation.evaluator import Evaluator, exact_or_contains_judge
from .evaluation.models import DecisionOutcome, EvalQuestion, ExperimentStatus, StrategyConfig
from .evaluation.store import ExperimentStore
from .pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


pass_settings = click.make_pass_decorator(Settings)


@click.group()
@click.option("--config-file", type=click.Path(), help="Path to .env config file")
@click.pass_context
def main(ctx: click.Context, config_file: str | None) -> None:
    """Local AI Document RAG Stack CLI."""
    settings = Settings(_env_file=config_file) if config_file else Settings()  # type: ignore[call-arg]
    ctx.obj = settings


@main.command()
@click.option("--host", default=None, help="Bind host")
@click.option("--port", default=None, type=int, help="Bind port")
@click.pass_obj
def serve(settings: Settings, host: str | None, port: int | None) -> None:
    """Run the FastAPI orchestrator."""
    import uvicorn

    from .api.app import create_app

    app = create_app(settings)
    uvicorn.run(
        app,
        host=host or settings.api_host,
        port=port or settings.api_port,
        workers=settings.api_workers,
    )


@main.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.pass_obj
def ingest(settings: Settings, file_path: Path) -> None:
    """Ingest a document."""
    pipeline = RAGPipeline(settings)
    try:
        doc = pipeline.ingest(file_path)
        click.echo(f"Ingested: {doc.document_name}")
        click.echo(f"  document_id: {doc.document_id}")
        click.echo(f"  chunks: {len(doc.chunks)}")
        click.echo(f"  pages: {len(doc.pages)}")
    finally:
        pipeline.close()


@main.command()
@click.argument("query")
@click.option("--top-k", default=None, type=int, help="Number of sources to use")
@click.option("--no-images", is_flag=True, help="Do not include page images in answer prompt")
@click.pass_obj
def query(settings: Settings, query: str, top_k: int | None, no_images: bool) -> None:
    """Ask a question over ingested documents."""
    pipeline = RAGPipeline(settings)
    try:
        response = pipeline.query(query, top_k=top_k, include_images=not no_images)
        click.echo(response.answer)
        click.echo("\nSources:")
        for citation in response.citations:
            page = f" page {citation.page_number}" if citation.page_number else ""
            click.echo(
                f"  [{citation.source_id}] {citation.document_name}{page} "
                f"(score: {citation.score:.3f})"
            )
        click.echo(f"\nModel: {response.model} | elapsed: {response.elapsed_seconds:.2f}s")
    finally:
        pipeline.close()


@main.command()
@click.pass_obj
def list_documents(settings: Settings) -> None:
    """List ingested documents."""
    pipeline = RAGPipeline(settings)
    try:
        docs = pipeline.list_documents()
        if not docs:
            click.echo("No documents ingested.")
            return
        for doc in docs:
            click.echo(
                f"{doc.document_id}  {doc.document_name}  "
                f"chunks={doc.chunk_count} pages={doc.page_count}"
            )
    finally:
        pipeline.close()


@main.command()
@click.argument("document_id")
@click.pass_obj
def delete(settings: Settings, document_id: str) -> None:
    """Delete an ingested document."""
    pipeline = RAGPipeline(settings)
    try:
        pipeline.delete_document(document_id)
        click.echo(f"Deleted {document_id}")
    finally:
        pipeline.close()


@main.command()
@click.pass_obj
def health(settings: Settings) -> None:
    """Check service health."""
    pipeline = RAGPipeline(settings)
    try:
        h = pipeline.health()
        click.echo(f"status: {h.status}")
        click.echo(f"ollama_ready: {h.ollama_ready}")
        click.echo(f"vector_store: {h.vector_store}")
        click.echo(f"graph_store: {h.graph_store}")
        for name, model in h.models.items():
            click.echo(f"  {name}: {model}")
    finally:
        pipeline.close()


@main.command("graph-query")
@click.option("--query", default=None, help="Free-text search over entity names/source text")
@click.option("--entity-type", default=None, help="Filter by entity type")
@click.option("--entity-name", default=None, help="Filter by entity name substring")
@click.option("--relation-type", default=None, help="Filter by relation type")
@click.option("--document-id", default=None, help="Filter by source document id")
@click.option("--top-k", default=20, type=int, help="Maximum entities to return")
@click.pass_obj
def graph_query(
    settings: Settings,
    query: str | None,
    entity_type: str | None,
    entity_name: str | None,
    relation_type: str | None,
    document_id: str | None,
    top_k: int,
) -> None:
    """Query the knowledge graph."""
    from .models import GraphQueryRequest

    pipeline = RAGPipeline(settings)
    try:
        request = GraphQueryRequest(
            query=query,
            entity_types=[entity_type] if entity_type else None,
            entity_name=entity_name,
            relation_type=relation_type,
            document_id=document_id,
            top_k=top_k,
        )
        response = pipeline.graph_query(request)
        click.echo(f"Entities: {len(response.entities)} | Relations: {len(response.relations)}")
        for entity in response.entities:
            click.echo(
                f"  [{entity.entity_type}] {entity.name} ({entity.id})"
            )
        for relation in response.relations:
            click.echo(
                f"  ({relation.relation_type}) {relation.source_entity_id} -> {relation.target_entity_id}"
            )
    finally:
        pipeline.close()


@main.command("graph-neighborhood")
@click.argument("entity_id")
@click.option("--hops", default=1, type=int, help="Number of hops to expand")
@click.pass_obj
def graph_neighborhood(settings: Settings, entity_id: str, hops: int) -> None:
    """Expand the graph neighborhood around an entity id."""
    from .models import GraphNeighborhoodRequest

    pipeline = RAGPipeline(settings)
    try:
        request = GraphNeighborhoodRequest(entity_id=entity_id, hops=hops)
        response = pipeline.graph_neighborhood(request)
        click.echo(f"Entities: {len(response.entities)} | Relations: {len(response.relations)}")
        for entity in response.entities:
            marker = " *" if entity.id == entity_id else ""
            click.echo(
                f"  [{entity.entity_type}] {entity.name} ({entity.id}){marker}"
            )
        for relation in response.relations:
            click.echo(
                f"  ({relation.relation_type}) {relation.source_entity_id} -> {relation.target_entity_id}"
            )
    finally:
        pipeline.close()


def _experiment_db_path(settings: Settings) -> Path:
    return settings.data_dir / "lrs.experiments.sqlite"


def _load_eval_set(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(EvalQuestion.model_validate_json(line))
    return questions


def _settings_with_strategy(settings: Settings, strategy: StrategyConfig) -> Settings:
    """Return a copy of ``settings`` overridden by the experiment strategy."""
    overrides = strategy.model_dump(exclude_none=True)
    # ``Settings`` does not yet expose chunking knobs, so keep them for the
    # experiment record but do not pass them to the pipeline constructor.
    overrides.pop("chunk_size", None)
    overrides.pop("chunk_overlap", None)
    return Settings(**{**settings.model_dump(), **overrides})


@main.group()
@click.pass_obj
def experiment(settings: Settings) -> None:
    """Run and manage evaluator-optimizer experiments."""


@experiment.command("run")
@click.option(
    "--eval-set",
    "eval_set_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--name", required=True, help="Human-readable experiment name")
@click.option("--parent-id", default=None, help="Parent experiment id")
@click.option("--strategy-json", default=None, help="JSON object of strategy overrides")
@click.pass_obj
def experiment_run(
    settings: Settings,
    eval_set_path: Path,
    name: str,
    parent_id: str | None,
    strategy_json: str | None,
) -> None:
    """Run an eval set and store the result as an experiment."""
    strategy = StrategyConfig()
    if strategy_json:
        strategy = StrategyConfig.model_validate_json(strategy_json)

    eval_set = _load_eval_set(eval_set_path)
    if not eval_set:
        click.echo("Eval set is empty.")
        return

    store = ExperimentStore(_experiment_db_path(settings))
    experiment_record = store.create_experiment(name, strategy_config=strategy, parent_id=parent_id)

    experiment_settings = _settings_with_strategy(settings, strategy)
    pipeline = RAGPipeline(experiment_settings)
    try:
        store.update_status(experiment_record.id, ExperimentStatus.RUNNING)

        evaluator = Evaluator(pipeline, judge=exact_or_contains_judge)
        results = evaluator.evaluate(eval_set)
        for result in results:
            store.save_result(experiment_record.id, result)

        store.update_status(experiment_record.id, ExperimentStatus.COMPLETED)

        avg_precision = sum(r.precision_at_k for r in results) / len(results)
        avg_correctness = sum(r.answer_correctness for r in results) / len(results)
        click.echo(f"Experiment: {experiment_record.id}")
        click.echo(f"Name: {name}")
        click.echo(f"Questions: {len(results)}")
        click.echo(f"Avg precision@K: {avg_precision:.3f}")
        click.echo(f"Avg correctness: {avg_correctness:.3f}")
    finally:
        pipeline.close()
        store.close()


@experiment.command("list")
@click.pass_obj
def experiment_list(settings: Settings) -> None:
    """List all experiments."""
    store = ExperimentStore(_experiment_db_path(settings))
    try:
        experiments = store.list_experiments()
        if not experiments:
            click.echo("No experiments found.")
            return
        for exp in experiments:
            click.echo(
                f"{exp.id}  {exp.name}  {exp.status.value}  {exp.created_at.isoformat()}"
            )
    finally:
        store.close()


@experiment.command("show")
@click.argument("experiment_id")
@click.pass_obj
def experiment_show(settings: Settings, experiment_id: str) -> None:
    """Show one experiment including its eval results and decision."""
    store = ExperimentStore(_experiment_db_path(settings))
    try:
        exp = store.get_experiment(experiment_id)
        if exp is None:
            raise click.ClickException(f"Experiment not found: {experiment_id}")
        click.echo(f"id: {exp.id}")
        click.echo(f"name: {exp.name}")
        click.echo(f"status: {exp.status.value}")
        click.echo(f"parent_id: {exp.parent_id}")
        click.echo(f"created_at: {exp.created_at.isoformat()}")
        click.echo(f"strategy: {exp.strategy_config.model_dump_json()}")
        if exp.decision:
            click.echo(
                f"decision: {exp.decision.outcome.value} "
                f"({exp.decision.decided_at.isoformat()})"
            )
            if exp.decision.reason:
                click.echo(f"decision reason: {exp.decision.reason}")
        click.echo(f"results: {len(exp.results)}")
        for result in exp.results:
            click.echo(f"  - {result.question_id}: p@k={result.precision_at_k:.3f}, "
                       f"correct={result.answer_correctness:.3f}, "
                       f"elapsed={result.elapsed_seconds:.3f}s")
    finally:
        store.close()


@experiment.command("decide")
@click.argument("experiment_id")
@click.option("--outcome", type=click.Choice([o.value for o in DecisionOutcome]), required=True)
@click.option("--reason", default=None, help="Why this decision was made")
@click.pass_obj
def experiment_decide(
    settings: Settings, experiment_id: str, outcome: str, reason: str | None
) -> None:
    """Record a keep/revert decision for an experiment."""
    store = ExperimentStore(_experiment_db_path(settings))
    try:
        decision = store.record_decision(
            experiment_id, DecisionOutcome(outcome), reason=reason
        )
        click.echo(
            f"Recorded {decision.outcome.value} for {decision.experiment_id}"
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
