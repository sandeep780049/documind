"""Reusable FastAPI dependencies."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Query
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import AuthenticationError
from app.core.security import TokenError, decode_access_token
from app.models.user import User
from app.services.embedding_service import BaseEmbeddingService, get_embedding_service
from app.services.llm_service import BaseLLMService, get_llm_service
from app.services.vector_service import VectorService

bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")

DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db: DbSession,
) -> User:
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Authentication required")
    if credentials.scheme.lower() != "bearer":
        raise AuthenticationError("Unsupported authentication scheme")
    try:
        payload = decode_access_token(credentials.credentials)
    except TokenError as exc:
        raise AuthenticationError(str(exc)) from exc

    try:
        user_id = uuid.UUID(str(payload.get("sub")))
    except (ValueError, TypeError) as exc:
        raise AuthenticationError("Invalid authentication token") from exc

    user = db.get(User, user_id)
    if user is None:
        raise AuthenticationError("The account no longer exists")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def provide_embedding_service() -> BaseEmbeddingService:
    """Indirection so tests can override the provider."""
    return get_embedding_service()


def provide_llm_service() -> BaseLLMService:
    return get_llm_service()


def provide_vector_service() -> VectorService:
    return VectorService()


EmbeddingProviderDep = Annotated[BaseEmbeddingService, Depends(provide_embedding_service)]
LLMProviderDep = Annotated[BaseLLMService, Depends(provide_llm_service)]
VectorServiceDep = Annotated[VectorService, Depends(provide_vector_service)]


class Pagination:
    def __init__(
        self,
        page: int = Query(1, ge=1, description="1-based page number"),
        limit: int = Query(20, ge=1, le=100, description="Items per page"),
    ) -> None:
        self.page = page
        self.limit = limit
        self.offset = (page - 1) * limit


PaginationDep = Annotated[Pagination, Depends(Pagination)]
