"""Chat / RAG endpoint tests."""

from __future__ import annotations

from tests.conftest import (
    auth_headers,
    make_txt_upload,
    register_user,
    requires_db,
)

pytestmark = requires_db

REMOTE_POLICY = (
    "Remote work policy: employees may work remotely three days per week. "
    "Managers approve remote schedules monthly."
)
EXPENSE_POLICY = (
    "Expense policy: receipts must be submitted within 30 days. "
    "Travel expenses above 500 USD require manager approval."
)


def _upload(client, token: str, text: str, name: str) -> dict:
    response = client.post(
        "/api/v1/documents/upload",
        files=make_txt_upload(text, name),
        headers=auth_headers(token),
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_ask_without_documents_returns_no_context_answer(client):
    _, token = register_user(client)
    response = client.post(
        "/api/v1/chat/ask",
        json={"question": "What is the remote work policy?"},
        headers=auth_headers(token),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sources"] == []
    assert "could not find" in body["answer"].lower()
    assert body["conversation_id"]


def test_ask_retrieves_relevant_chunks_and_calls_llm(client, fake_llm):
    _, token = register_user(client)
    document = _upload(client, token, REMOTE_POLICY, "remote-policy.txt")
    _upload(client, token, EXPENSE_POLICY, "expenses.txt")

    response = client.post(
        "/api/v1/chat/ask",
        json={"question": "How many days per week can I work remotely?", "top_k": 3},
        headers=auth_headers(token),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == (
        "According to the uploaded document, the answer is stated in source [1]."
    )
    assert body["sources"], "expected retrieved sources"
    top_source = body["sources"][0]
    assert top_source["document_id"] == document["id"]
    assert top_source["document_name"] == "remote-policy.txt"
    assert 0.0 <= top_source["similarity"] <= 1.0
    assert body["retrieved_chunks"] >= 1
    assert body["model"] == "fake-chat-model"
    assert body["embedding_model"] == "fake-bow-embedding"

    assert len(fake_llm.calls) == 1
    system_message = fake_llm.calls[0][0]
    assert "ONLY the information" in system_message.content
    assert "Never invent" in system_message.content


def test_multi_turn_conversation_and_history(client):
    _, token = register_user(client)
    _upload(client, token, REMOTE_POLICY, "remote-policy.txt")

    first = client.post(
        "/api/v1/chat/ask",
        json={"question": "What is the remote work policy?", "top_k": 2},
        headers=auth_headers(token),
    ).json()
    conversation_id = first["conversation_id"]

    second = client.post(
        "/api/v1/chat/ask",
        json={
            "question": "Who approves remote schedules?",
            "conversation_id": conversation_id,
        },
        headers=auth_headers(token),
    )
    assert second.status_code == 200, second.text
    assert second.json()["conversation_id"] == conversation_id

    history = client.get("/api/v1/chat/history", headers=auth_headers(token)).json()
    assert history["total"] == 1
    assert history["items"][0]["message_count"] == 4
    assert history["items"][0]["last_message_preview"]

    detail = client.get(
        f"/api/v1/chat/history/{conversation_id}", headers=auth_headers(token)
    ).json()
    assert detail["message_count"] == 4
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert detail["messages"][1]["sources"]


def test_conversation_history_is_user_scoped(client):
    _, owner_token = register_user(client)
    _, other_token = register_user(client)
    owner = client.post(
        "/api/v1/chat/ask",
        json={"question": "Any question about my own data?"},
        headers=auth_headers(owner_token),
    ).json()
    conversation_id = owner["conversation_id"]

    assert (
        client.get(
            f"/api/v1/chat/history/{conversation_id}", headers=auth_headers(other_token)
        ).status_code
        == 404
    )
    assert (
        client.delete(
            f"/api/v1/chat/history/{conversation_id}", headers=auth_headers(other_token)
        ).status_code
        == 404
    )
    continue_foreign = client.post(
        "/api/v1/chat/ask",
        json={"question": "Can I read someone else's history?", "conversation_id": conversation_id},
        headers=auth_headers(other_token),
    )
    assert continue_foreign.status_code == 404
    assert client.get("/api/v1/chat/history", headers=auth_headers(other_token)).json()["total"] == 0


def test_delete_conversation(client):
    _, token = register_user(client)
    created = client.post(
        "/api/v1/chat/ask",
        json={"question": "Will this conversation be deleted?"},
        headers=auth_headers(token),
    ).json()
    conversation_id = created["conversation_id"]

    deleted = client.delete(
        f"/api/v1/chat/history/{conversation_id}", headers=auth_headers(token)
    )
    assert deleted.status_code == 200
    assert (
        client.get(
            f"/api/v1/chat/history/{conversation_id}", headers=auth_headers(token)
        ).status_code
        == 404
    )
    assert client.get("/api/v1/chat/history", headers=auth_headers(token)).json()["total"] == 0


def test_ask_validates_question_length(client):
    _, token = register_user(client)
    response = client.post(
        "/api/v1/chat/ask", json={"question": "hi"}, headers=auth_headers(token)
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_ask_top_k_validation(client):
    _, token = register_user(client)
    response = client.post(
        "/api/v1/chat/ask",
        json={"question": "A perfectly valid question", "top_k": 99},
        headers=auth_headers(token),
    )
    assert response.status_code == 422
