"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..config import Settings
from ..tenant import TenantManager
from .routes import router

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or Settings()  # type: ignore[call-arg]

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        app.state.tenant_manager = TenantManager(settings)
        app.state.tenant_pipelines: dict[str, object] = {}
        logger.info("Tenant manager initialized (%d tenants)", len(app.state.tenant_manager.tenants))
        yield
        for pipeline in app.state.tenant_pipelines.values():
            pipeline.close()
        logger.info("Tenant pipelines closed")

    app = FastAPI(
        title="Local AI Document RAG Stack",
        description="On-premise hybrid document RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
