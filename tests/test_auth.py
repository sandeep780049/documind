"""Authentication tests."""

from __future__ import annotations

from tests.conftest import auth_headers, register_user, requires_db

pytestmark = requires_db


def test_register_returns_token_and_user(client):
    response = client.post(
        "/api/v1/auth/register",
        json={"email": "Alice@Example.com", "password": "StrongPass123"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["expires_in"] > 0
    assert body["user"]["email"] == "alice@example.com"
    assert "hashed_password" not in body["user"]


def test_register_rejects_duplicate_email(client):
    payload = {"email": "duplicate@example.com", "password": "StrongPass123"}
    assert client.post("/api/v1/auth/register", json=payload).status_code == 201
    duplicate = client.post("/api/v1/auth/register", json=payload)
    assert duplicate.status_code == 409
    assert duplicate.json()["code"] == "conflict"


def test_register_validates_password_and_email(client):
    weak = client.post(
        "/api/v1/auth/register",
        json={"email": "weak@example.com", "password": "onlyletters"},
    )
    assert weak.status_code == 422
    assert weak.json()["code"] == "validation_error"

    invalid_email = client.post(
        "/api/v1/auth/register",
        json={"email": "not-an-email", "password": "StrongPass123"},
    )
    assert invalid_email.status_code == 422


def test_login_and_me(client):
    _, token = register_user(client, "login@example.com")
    login = client.post(
        "/api/v1/auth/login",
        json={"email": "login@example.com", "password": "StrongPass123"},
    )
    assert login.status_code == 200
    new_token = login.json()["access_token"]

    me = client.get("/api/v1/auth/me", headers=auth_headers(new_token))
    assert me.status_code == 200
    assert me.json()["email"] == "login@example.com"

    original = client.get("/api/v1/auth/me", headers=auth_headers(token))
    assert original.status_code == 200


def test_login_with_wrong_password_is_rejected(client):
    register_user(client, "wrongpass@example.com")
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "wrongpass@example.com", "password": "WrongPass123"},
    )
    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_login_unknown_user_is_rejected(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "ghost@example.com", "password": "StrongPass123"},
    )
    assert response.status_code == 401


def test_protected_endpoints_require_auth(client):
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/documents").status_code == 401
    assert client.get("/api/v1/chat/history").status_code == 401
    bad_token = client.get("/api/v1/auth/me", headers=auth_headers("not-a-jwt"))
    assert bad_token.status_code == 401
