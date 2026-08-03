"""FastAPI endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from local_rag_stack.api.app import create_app
from local_rag_stack.config import Settings

API_KEY_A = "test-key-a"
API_KEY_B = "test-key-b"


@pytest.fixture
def settings(tmp_path_factory) -> Settings:
    data_dir = tmp_path_factory.mktemp("lrs-data")
    return Settings(
        data_dir=data_dir,
        sqlite_vec_path=data_dir / "test.vec.sqlite",
        extraction_engine="plaintext",
        text_embed_model="BAAI/bge-small-en-v1.5",
        vision_embed_backend="none",
        rerank_backend="none",
        auth_required=True,
        tenant_keys_json=f'{{"{API_KEY_A}": "tenant-a", "{API_KEY_B}": "tenant-b"}}',
    )  # type: ignore[call-arg]


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health_requires_auth(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 401


def test_health_with_api_key(client: TestClient) -> None:
    response = client.get("/health", headers={"X-API-Key": API_KEY_A})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vector_store" in data
    assert data["tenant_id"] == "tenant-a"


def test_health_with_bearer_token(client: TestClient) -> None:
    response = client.get("/health", headers={"Authorization": f"Bearer {API_KEY_B}"})
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "tenant-b"


def test_ingest_text_file(client: TestClient, tmp_path) -> None:
    path = tmp_path / "readme.md"
    path.write_text("# Local RAG Stack\nThis is a test document.")
    with path.open("rb") as f:
        response = client.post(
            "/ingest",
            files={"file": ("readme.md", f, "text/markdown")},
            data={"metadata": '{"test": true}'},
            headers={"X-API-Key": API_KEY_A},
        )
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["document_name"] == "readme.md"


def test_list_documents(client: TestClient, tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("Hello world.")
    with path.open("rb") as f:
        client.post("/ingest", files={"file": ("doc.md", f, "text/markdown")}, headers={"X-API-Key": API_KEY_A})
    response = client.get("/documents", headers={"X-API-Key": API_KEY_A})
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_tenant_isolation(client: TestClient, tmp_path) -> None:
    """Documents ingested for tenant-a must not be visible to tenant-b."""
    path = tmp_path / "tenant-doc.md"
    path.write_text("Tenant-specific content.")

    with path.open("rb") as f:
        ingest_response = client.post(
            "/ingest",
            files={"file": ("tenant-doc.md", f, "text/markdown")},
            headers={"X-API-Key": API_KEY_A},
        )
    assert ingest_response.status_code == 200

    list_a = client.get("/documents", headers={"X-API-Key": API_KEY_A})
    assert list_a.status_code == 200
    assert len(list_a.json()) == 1

    list_b = client.get("/documents", headers={"X-API-Key": API_KEY_B})
    assert list_b.status_code == 200
    assert len(list_b.json()) == 0


def test_auth_disabled_uses_default_tenant(tmp_path_factory) -> None:
    data_dir = tmp_path_factory.mktemp("lrs-data-default")
    settings = Settings(
        data_dir=data_dir,
        sqlite_vec_path=data_dir / "test.vec.sqlite",
        extraction_engine="plaintext",
        text_embed_model="BAAI/bge-small-en-v1.5",
        vision_embed_backend="none",
        rerank_backend="none",
        auth_required=False,
    )  # type: ignore[call-arg]
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["tenant_id"] == "default"
