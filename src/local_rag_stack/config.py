"""Pydantic settings for the Local RAG Stack."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables.

    All variables are prefixed with ``LRS_``.
    """

    model_config = SettingsConfigDict(
        env_prefix="LRS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_timeout: float = Field(default=120.0, description="Ollama request timeout in seconds")

    # Models
    text_embed_model: str = Field(
        default="qwen3-embedding:4b",
        description="Ollama text embedding model name",
    )
    vision_embed_model: str = Field(
        default="qwen3-vl",
        description="Visual embedding model (Ollama or local MLX/HF name)",
    )
    vision_embed_backend: str = Field(
        default="ollama",
        description="Backend for visual embeddings: 'ollama', 'mlx', or 'hf'",
    )
    rerank_model: str = Field(
        default="bge-reranker-v2-m3",
        description="Cross-encoder reranker model name",
    )
    rerank_backend: str = Field(
        default="ollama",
        description="Backend for reranking: 'ollama', 'sentence_transformers'",
    )
    llm_model: str = Field(
        default="qwen3-vl",
        description="LLM/VLM used for final answer generation",
    )

    # Extraction
    extraction_engine: str = Field(
        default="docling",
        description="Document extraction engine: 'docling' or 'marker'",
    )
    extraction_dpi: int = Field(
        default=150,
        description="DPI used when rendering pages to images",
    )
    force_full_page_ocr: bool = Field(
        default=False,
        description="Run OCR across whole PDF pages even when a native text layer exists",
    )
    ocr_engine: str = Field(
        default="auto",
        description="OCR engine: auto, ocrmac, or rapidocr. Auto prefers macOS Vision when available.",
    )

    # Storage
    vector_store: str = Field(
        default="sqlite",
        description="Vector store backend: 'sqlite' or 'qdrant'",
    )
    sqlite_vec_path: Path = Field(
        default=Path("./data/lrs.vec.sqlite"),
        description="Path to sqlite-vec database file",
    )
    graph_store_path: Path = Field(
        default=Path("./data/lrs.graph.sqlite"),
        description="Path to SQLite graph database file",
    )
    graph_extractor_model: str = Field(
        default="gemma4:latest",
        description="LLM used for entity/relation extraction",
    )
    graph_extract_on_ingest: bool = Field(
        default=True,
        description="Run graph extraction during document ingestion",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL",
    )
    qdrant_api_key: str | None = Field(
        default=None,
        description="Qdrant API key",
    )
    qdrant_collection: str = Field(
        default="local_rag_stack",
        description="Qdrant collection name",
    )

    # Retrieval
    text_search_k: int = Field(default=10, description="Top-K text results to retrieve")
    visual_search_k: int = Field(default=10, description="Top-K visual results to retrieve")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion constant")
    rerank_top_k: int = Field(default=5, description="Top-K results to send to reranker")
    final_top_k: int = Field(
        default=5,
        description="Top-K results to send to the answer generator",
    )

    # API
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind host")
    api_port: int = Field(default=8000, description="FastAPI bind port")
    api_workers: int = Field(default=1, description="Number of uvicorn workers")

    # Appliance
    data_dir: Path = Field(default=Path("./data"), description="Local data directory")

    # Tenant / auth
    auth_required: bool = Field(
        default=False,
        description="Require X-API-Key header for all API requests",
    )
    default_tenant_id: str = Field(
        default="default",
        description="Tenant id used when auth is disabled or no key is supplied",
    )
    tenant_keys_json: str | None = Field(
        default=None,
        description="JSON object mapping api_key -> tenant_id",
    )
    tenant_keys_file: Path | None = Field(
        default=None,
        description="Path to JSON file mapping api_key -> tenant_id",
    )
    tenants_file: Path | None = Field(
        default=None,
        description="Path to JSON file with full Tenant objects",
    )

    def ensure_paths(self) -> None:
        """Create required directories on disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_vec_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()  # type: ignore[call-arg]
