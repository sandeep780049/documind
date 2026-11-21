"""Password hashing and JWT helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app.core.config import settings

_password_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=64 * 1024,
    parallelism=4,
    hash_len=32,
    salt_len=16,
)

# Verified when the user does not exist so that login latency is comparable.
_DUMMY_HASH = _password_hasher.hash("documind-dummy-password")


class TokenError(Exception):
    """Raised when a JWT cannot be trusted."""


def hash_password(password: str) -> str:
    return _password_hasher.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    try:
        return _password_hasher.verify(hashed_password, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def verify_password_dummy(password: str) -> None:
    """Burn the same amount of CPU as a real verification attempt."""
    try:
        _password_hasher.verify(_DUMMY_HASH, password)
    except (VerifyMismatchError, InvalidHashError):
        pass


def password_needs_rehash(hashed_password: str) -> bool:
    return _password_hasher.check_needs_rehash(hashed_password)


def create_access_token(subject: str | UUID, expires_minutes: int | None = None) -> tuple[str, int]:
    """Return ``(token, expires_in_seconds)`` for the given user id."""
    minutes = expires_minutes or settings.access_token_expire_minutes
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=minutes)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    token = jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)
    return token, minutes * 60


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Invalid authentication token") from exc
    if payload.get("type") != "access":
        raise TokenError("Invalid token type")
    return payload
