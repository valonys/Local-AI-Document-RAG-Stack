"""Tenant manager and per-tenant settings tests."""

from __future__ import annotations

import json

import pytest
from fastapi import Request

from local_rag_stack.config import Settings
from local_rag_stack.tenant import TenantManager


def test_manager_loads_keys_from_json(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        sqlite_vec_path=tmp_path / "vec.sqlite",
        auth_required=True,
        tenant_keys_json=json.dumps({"key-1": "tenant-1", "key-2": "tenant-2"}),
    )  # type: ignore[call-arg]
    manager = TenantManager(settings)
    assert len(manager.tenants) == 2
    assert manager.get_tenant_for_key("key-1").id == "tenant-1"
    assert manager.get_tenant_for_key("key-2").id == "tenant-2"


def test_manager_loads_keys_from_file(tmp_path) -> None:
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"file-key": "tenant-file"}))
    settings = Settings(
        data_dir=tmp_path,
        sqlite_vec_path=tmp_path / "vec.sqlite",
        auth_required=True,
        tenant_keys_file=keys_file,
    )  # type: ignore[call-arg]
    manager = TenantManager(settings)
    assert manager.get_tenant_for_key("file-key").id == "tenant-file"


def test_settings_for_tenant_namespaces_storage(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        sqlite_vec_path=tmp_path / "vec.sqlite",
        graph_store_path=tmp_path / "graph.sqlite",
        auth_required=True,
        tenant_keys_json=json.dumps({"key": "acme-corp"}),
    )  # type: ignore[call-arg]
    manager = TenantManager(settings)
    tenant = manager.get_tenant_for_key("key")
    tenant_settings = manager.settings_for_tenant(tenant)

    assert tenant_settings.data_dir == tmp_path / "tenants" / "acme-corp"
    assert tenant_settings.sqlite_vec_path == tmp_path / "tenants" / "acme-corp" / "lrs.vec.sqlite"
    assert tenant_settings.graph_store_path == tmp_path / "tenants" / "acme-corp" / "lrs.graph.sqlite"
    assert tenant_settings.qdrant_collection == "local_rag_stack-acme-corp"


def test_default_tenant_when_auth_disabled(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        sqlite_vec_path=tmp_path / "vec.sqlite",
        auth_required=False,
        default_tenant_id="single",
    )  # type: ignore[call-arg]
    manager = TenantManager(settings)
    request = Request(scope={"type": "http", "headers": [], "method": "GET", "path": "/health"})
    tenant = manager.resolve_tenant(request)
    assert tenant.id == "single"


def test_resolve_tenant_raises_when_auth_required_and_key_missing(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        sqlite_vec_path=tmp_path / "vec.sqlite",
        auth_required=True,
        tenant_keys_json=json.dumps({"key": "tenant"}),
    )  # type: ignore[call-arg]
    manager = TenantManager(settings)
    request = Request(scope={"type": "http", "headers": [], "method": "GET", "path": "/health"})
    with pytest.raises(Exception) as exc_info:
        manager.resolve_tenant(request)
    assert exc_info.value.status_code == 401


def test_invalid_key_raises_401(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        sqlite_vec_path=tmp_path / "vec.sqlite",
        auth_required=True,
        tenant_keys_json=json.dumps({"key": "tenant"}),
    )  # type: ignore[call-arg]
    manager = TenantManager(settings)
    headers = [(b"x-api-key", b"wrong-key")]
    request = Request(
        scope={"type": "http", "headers": headers, "method": "GET", "path": "/health"}
    )
    with pytest.raises(Exception) as exc_info:
        manager.resolve_tenant(request)
    assert exc_info.value.status_code == 401
