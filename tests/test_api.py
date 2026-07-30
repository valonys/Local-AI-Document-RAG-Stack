"""FastAPI endpoint tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from local_rag_stack.api.app import create_app
from local_rag_stack.config import Settings


@pytest.fixture
def client(tmp_path_factory) -> TestClient:
    data_dir = tmp_path_factory.mktemp("lrs-data")
    settings = Settings(
        data_dir=data_dir,
        sqlite_vec_path=data_dir / "test.vec.sqlite",
        extraction_engine="plaintext",
        text_embed_model="BAAI/bge-small-en-v1.5",
        vision_embed_backend="none",
        rerank_backend="none",
    )  # type: ignore[call-arg]
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vector_store" in data


def test_ingest_text_file(client: TestClient, tmp_path) -> None:
    path = tmp_path / "readme.md"
    path.write_text("# Local RAG Stack\nThis is a test document.")
    with path.open("rb") as f:
        response = client.post(
            "/ingest",
            files={"file": ("readme.md", f, "text/markdown")},
            data={"metadata": '{"test": true}'},
        )
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["document_name"] == "readme.md"


def test_list_documents(client: TestClient, tmp_path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("Hello world.")
    with path.open("rb") as f:
        client.post("/ingest", files={"file": ("doc.md", f, "text/markdown")})
    response = client.get("/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
