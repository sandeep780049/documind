"""Authentication business logic."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import AuthenticationError, ConflictError
from app.core.security import (
    hash_password,
    password_needs_rehash,
    verify_password,
    verify_password_dummy,
)
from app.models.user import User
from app.schemas.auth import UserCreate


class AuthService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def register(self, payload: UserCreate) -> User:
        email = payload.email.lower().strip()
        existing = self.db.scalar(select(User).where(func.lower(User.email) == email))
        if existing is not None:
            raise ConflictError("An account with this email already exists")
        user = User(email=email, hashed_password=hash_password(payload.password))
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def authenticate(self, email: str, password: str) -> User:
        normalized = email.lower().strip()
        user = self.db.scalar(select(User).where(func.lower(User.email) == normalized))
        if user is None:
            verify_password_dummy(password)
            raise AuthenticationError("Incorrect email or password")
        if not verify_password(password, user.hashed_password):
            raise AuthenticationError("Incorrect email or password")
        if password_needs_rehash(user.hashed_password):
            user.hashed_password = hash_password(password)
            self.db.commit()
        return user

    def get_user(self, user_id) -> User | None:
        return self.db.get(User, user_id)
