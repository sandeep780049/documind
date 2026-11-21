"""Authentication schemas."""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

_PASSWORD_RULES = (
    (re.compile(r"[A-Za-z]"), "Password must contain at least one letter"),
    (re.compile(r"\d"), "Password must contain at least one digit"),
)


class UserCreate(BaseModel):
    email: EmailStr = Field(description="Unique account email")
    password: str = Field(min_length=8, max_length=128, description="At least 8 characters")

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("password")
    @classmethod
    def _password_policy(cls, value: str) -> str:
        for pattern, message in _PASSWORD_RULES:
            if not pattern.search(value):
                raise ValueError(message)
        return value


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def _normalise_email(cls, value: str) -> str:
        return value.strip().lower()


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds")
    user: UserRead
