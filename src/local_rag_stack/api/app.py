"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from ..config import Settings
from ..pipeline import RAGPipeline
from .routes import router

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or Settings()  # type: ignore[call-arg]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.pipeline = RAGPipeline(settings)
        logger.info("RAG pipeline initialized")
        yield
        app.state.pipeline.close()
        logger.info("RAG pipeline closed")

    app = FastAPI(
        title="Local AI Document RAG Stack",
        description="On-premise hybrid document RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
