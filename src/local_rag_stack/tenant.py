"""Tenant-aware auth and per-tenant pipeline isolation.

The API wrapper maps an ``X-API-Key`` (or ``Authorization: Bearer``) header to
a tenant. Each tenant gets its own data directory, sqlite-vec database,
graph store, and (when Qdrant is used) a prefixed collection. The model objects
(embedders, generators) are still shared because a per-tenant pipeline is built
from a tenant-scoped copy of the global Settings.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field

from .config import Settings
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class Tenant(BaseModel):
    """A single tenant / customer namespace."""

    id: str = Field(..., description="Unique tenant identifier used for namespacing storage.")
    name: str = Field(default="", description="Human-readable tenant name.")
    api_key: str = Field(..., description="API key that maps to this tenant.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class TenantManager:
    """Loads tenant/API-key mappings and creates per-tenant pipelines."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.auth_required = settings.auth_required
        self.default_tenant_id = settings.default_tenant_id
        self._tenants_by_key: dict[str, Tenant] = {}
        self._load_tenants()

    def _load_tenants(self) -> None:
        raw_mappings: dict[str, str] = {}

        if self.settings.tenant_keys_json:
            try:
                parsed = json.loads(self.settings.tenant_keys_json)
                if isinstance(parsed, dict):
                    raw_mappings.update(parsed)
                else:
                    logger.warning("LRS_TENANT_KEYS_JSON is not a JSON object; ignoring")
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse LRS_TENANT_KEYS_JSON: %s", exc)

        if self.settings.tenant_keys_file and self.settings.tenant_keys_file.exists():
            try:
                data = json.loads(self.settings.tenant_keys_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    raw_mappings.update(data)
                else:
                    logger.warning("tenant_keys_file does not contain a JSON object; ignoring")
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load tenant_keys_file: %s", exc)

        if self.settings.tenants_file and self.settings.tenants_file.exists():
            try:
                data = json.loads(self.settings.tenants_file.read_text(encoding="utf-8"))
                tenants = data if isinstance(data, list) else [data]
                for t in tenants:
                    tenant = Tenant.model_validate(t)
                    self._tenants_by_key[tenant.api_key] = tenant
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load tenants_file: %s", exc)

        # Simple key->tenant_id mappings take lowest precedence (overwritten by full objects).
        for key, tenant_id in raw_mappings.items():
            if key not in self._tenants_by_key:
                self._tenants_by_key[key] = Tenant(
                    id=str(tenant_id),
                    name=str(tenant_id),
                    api_key=str(key),
                )

        if self.auth_required and not self._tenants_by_key:
            logger.warning(
                "auth_required is True but no tenant keys are configured; "
                "all API requests will be rejected"
            )

    @property
    def tenants(self) -> list[Tenant]:
        return list(self._tenants_by_key.values())

    def get_tenant_for_key(self, api_key: str | None) -> Tenant | None:
        if not api_key:
            return None
        return self._tenants_by_key.get(api_key)

    def resolve_tenant(self, request: Request) -> Tenant:
        """Return the tenant for the current request, raising 401 if required."""
        api_key = self._extract_api_key(request)
        tenant = self.get_tenant_for_key(api_key)

        if tenant is not None:
            return tenant

        if not self.auth_required:
            return Tenant(
                id=self.default_tenant_id,
                name=self.default_tenant_id,
                api_key="",
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    @staticmethod
    def _extract_api_key(request: Request) -> str | None:
        if key := request.headers.get("X-API-Key"):
            return key.strip()
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return None

    def settings_for_tenant(self, tenant: Tenant) -> Settings:
        """Create a tenant-scoped copy of the global settings."""
        tenant_data_dir = self.settings.data_dir / "tenants" / tenant.id
        tenant_sqlite_path = tenant_data_dir / "lrs.vec.sqlite"
        tenant_graph_path = tenant_data_dir / "lrs.graph.sqlite"

        qdrant_collection = self.settings.qdrant_collection
        if tenant.id != self.default_tenant_id:
            safe_id = re.sub(r"[^a-z0-9]+", "-", tenant.id.lower()).strip("-")
            qdrant_collection = f"{qdrant_collection}-{safe_id}"

        return self.settings.model_copy(
            update={
                "data_dir": tenant_data_dir,
                "sqlite_vec_path": tenant_sqlite_path,
                "graph_store_path": tenant_graph_path,
                "qdrant_collection": qdrant_collection,
            },
            deep=True,
        )

    def build_pipeline(self, tenant: Tenant) -> RAGPipeline:
        """Build (or rebuild) a pipeline for a tenant."""
        tenant_settings = self.settings_for_tenant(tenant)
        return RAGPipeline(tenant_settings)


def get_current_tenant(request: Request) -> Tenant:
    """FastAPI dependency that resolves the tenant for the incoming request."""
    manager: TenantManager = request.app.state.tenant_manager
    return manager.resolve_tenant(request)


def get_tenant_pipeline(request: Request, tenant: Tenant) -> RAGPipeline:
    """Return a cached or newly created per-tenant pipeline."""
    pipelines: dict[str, RAGPipeline] = request.app.state.tenant_pipelines
    if tenant.id not in pipelines:
        manager: TenantManager = request.app.state.tenant_manager
        pipelines[tenant.id] = manager.build_pipeline(tenant)
    return pipelines[tenant.id]
