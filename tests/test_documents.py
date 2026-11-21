"""Document endpoint tests backed by real pgvector storage."""

from __future__ import annotations

import io

from sqlalchemy import func, select

from app.models.chunk import DocumentChunk
from tests.conftest import (
    auth_headers,
    make_pdf_upload,
    make_txt_upload,
    register_user,
    requires_db,
)

pytestmark = requires_db

TEXT = (
    "DocuMind employee handbook.\n\n"
    "Remote work policy: employees may work remotely three days per week. "
    "The remaining days should be spent in the office for collaboration.\n\n"
    "Expense policy: receipts must be submitted within 30 days. "
    "Travel expenses above 500 USD require manager approval.\n\n"
    "Security policy: all laptops must use full disk encryption and MFA."
)


def test_upload_txt_processes_and_chunks_document(client):
    user, token = register_user(client)
    response = client.post(
        "/api/v1/documents/upload", files=make_txt_upload(TEXT), headers=auth_headers(token)
    )
    assert response.status_code == 201, response.text
    document = response.json()
    assert document["processing_status"] == "completed"
    assert document["file_type"] == ".txt"
    assert document["chunk_count"] > 0
    assert document["page_count"] == 1
    assert document["file_size"] == len(TEXT.encode())
    assert document["error_message"] is None


def test_upload_pdf_extracts_pages(client):
    _, token = register_user(client)
    files = make_pdf_upload(["Quarterly report", "Revenue grew by 12 percent in Q3."])
    response = client.post(
        "/api/v1/documents/upload", files=files, headers=auth_headers(token)
    )
    assert response.status_code == 201, response.text
    assert response.json()["processing_status"] == "completed"
    assert response.json()["page_count"] == 1


def test_upload_rejects_unsupported_extension(client):
    _, token = register_user(client)
    files = {"file": ("malware.exe", io.BytesIO(b"MZ..."), "application/octet-stream")}
    response = client.post(
        "/api/v1/documents/upload", files=files, headers=auth_headers(token)
    )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media_type"


def test_upload_rejects_fake_pdf(client):
    _, token = register_user(client)
    files = {"file": ("fake.pdf", io.BytesIO(b"not really a pdf"), "application/pdf")}
    response = client.post(
        "/api/v1/documents/upload", files=files, headers=auth_headers(token)
    )
    assert response.status_code == 415


def test_upload_rejects_empty_file(client):
    _, token = register_user(client)
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    response = client.post(
        "/api/v1/documents/upload", files=files, headers=auth_headers(token)
    )
    assert response.status_code == 422
    assert response.json()["code"] == "document_processing_failed"


def test_upload_marks_unreadable_pdf_as_failed(client, db_session):
    _, token = register_user(client)
    files = {"file": ("broken.pdf", io.BytesIO(b"%PDF-1.4 broken"), "application/pdf")}
    response = client.post(
        "/api/v1/documents/upload", files=files, headers=auth_headers(token)
    )
    assert response.status_code == 422

    listed = client.get("/api/v1/documents", headers=auth_headers(token)).json()
    assert listed["total"] == 1
    assert listed["items"][0]["processing_status"] == "failed"
    assert listed["items"][0]["error_message"]


def test_document_listing_pagination(client):
    _, token = register_user(client)
    for index in range(3):
        client.post(
            "/api/v1/documents/upload",
            files=make_txt_upload(f"Policy number {index}. " + TEXT, f"policy-{index}.txt"),
            headers=auth_headers(token),
        )
    page_one = client.get(
        "/api/v1/documents?page=1&limit=2", headers=auth_headers(token)
    ).json()
    page_two = client.get(
        "/api/v1/documents?page=2&limit=2", headers=auth_headers(token)
    ).json()
    assert page_one["total"] == 3
    assert page_one["pages"] == 2
    assert len(page_one["items"]) == 2
    assert len(page_two["items"]) == 1
    assert {item["id"] for item in page_one["items"]}.isdisjoint(
        {item["id"] for item in page_two["items"]}
    )


def test_document_ownership_isolation(client):
    _, owner_token = register_user(client)
    _, other_token = register_user(client)
    created = client.post(
        "/api/v1/documents/upload", files=make_txt_upload(TEXT), headers=auth_headers(owner_token)
    ).json()

    assert (
        client.get(
            f"/api/v1/documents/{created['id']}", headers=auth_headers(other_token)
        ).status_code
        == 404
    )
    assert (
        client.delete(
            f"/api/v1/documents/{created['id']}", headers=auth_headers(other_token)
        ).status_code
        == 404
    )
    assert client.get("/api/v1/documents", headers=auth_headers(other_token)).json()["total"] == 0
    assert (
        client.get(
            f"/api/v1/documents/{created['id']}", headers=auth_headers(owner_token)
        ).status_code
        == 200
    )


def test_delete_document_removes_chunks(client, db_session):
    _, token = register_user(client)
    created = client.post(
        "/api/v1/documents/upload", files=make_txt_upload(TEXT), headers=auth_headers(token)
    ).json()
    assert db_session.scalar(select(func.count()).select_from(DocumentChunk)) > 0

    deleted = client.delete(
        f"/api/v1/documents/{created['id']}", headers=auth_headers(token)
    )
    assert deleted.status_code == 200
    assert deleted.json()["document_id"] == created["id"]

    missing = client.get(f"/api/v1/documents/{created['id']}", headers=auth_headers(token))
    assert missing.status_code == 404
    assert db_session.scalar(select(func.count()).select_from(DocumentChunk)) == 0
