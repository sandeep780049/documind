"""Test fixtures.

Integration tests run against a real PostgreSQL database with the pgvector
extension. Point ``TEST_DATABASE_URL`` at a disposable database. When it is
not set, database-backed tests are skipped instead of silently passing.
"""

from __future__ import annotations

import io
import os
import uuid

import pytest

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")

if TEST_DATABASE_URL:
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ["RATE_LIMIT_DEFAULT"] = "100000/minute"
    os.environ["RATE_LIMIT_AUTH"] = "100000/minute"
    os.environ["RATE_LIMIT_CHAT"] = "100000/minute"
    os.environ["RATE_LIMIT_UPLOAD"] = "100000/minute"
    os.environ["CORS_ORIGINS"] = "[]"

requires_db = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="TEST_DATABASE_URL is not configured; skipping PostgreSQL integration tests",
)

pytestmark = requires_db


@pytest.fixture(scope="session")
def database() -> None:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    from app.core.database import engine

    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture()
def db_session(database):
    from sqlalchemy import text

    from app.core.database import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        with session.bind.begin() as connection:
            connection.execute(text("TRUNCATE users, documents, conversations CASCADE"))


@pytest.fixture()
def fake_embedding():
    from tests.fakes import FakeEmbeddingService

    return FakeEmbeddingService()


@pytest.fixture()
def fake_llm():
    from tests.fakes import FakeLLMService

    return FakeLLMService()


@pytest.fixture()
def client(database, fake_embedding, fake_llm, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.api.dependencies import (
        provide_embedding_service,
        provide_llm_service,
    )
    from app.core.config import settings
    from app.core.database import get_db
    from app.main import app

    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)

    def _get_db():
        from app.core.database import SessionLocal

        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[provide_embedding_service] = lambda: fake_embedding
    app.dependency_overrides[provide_llm_service] = lambda: fake_llm

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def register_user(client, email: str | None = None) -> tuple[dict, str]:
    email = email or f"user-{uuid.uuid4().hex[:8]}@example.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "StrongPass123"},
    )
    assert response.status_code == 201, response.text
    payload = response.json()
    return payload["user"], payload["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def make_txt_upload(content: str, name: str = "handbook.txt") -> dict:
    return {"file": (name, io.BytesIO(content.encode("utf-8")), "text/plain")}


def make_pdf_upload(lines: list[str], name: str = "manual.pdf") -> dict:
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "\n".join(lines))
    buffer = io.BytesIO(document.tobytes())
    document.close()
    return {"file": (name, buffer, "application/pdf")}
