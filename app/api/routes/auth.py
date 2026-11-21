"""Authentication routes."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from app.api.dependencies import CurrentUser, DbSession
from app.core.config import settings
from app.core.rate_limit import limiter
from app.core.security import create_access_token
from app.schemas.auth import TokenResponse, UserCreate, UserLogin, UserRead
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _token_response(user_id, user: UserRead) -> TokenResponse:
    token, expires_in = create_access_token(user_id)
    return TokenResponse(access_token=token, expires_in=expires_in, user=user)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
)
@limiter.limit(settings.rate_limit_auth)
def register(
    request: Request, response: Response, payload: UserCreate, db: DbSession
) -> TokenResponse:
    user = AuthService(db).register(payload)
    return _token_response(user.id, UserRead.model_validate(user))


@router.post("/login", response_model=TokenResponse, summary="Obtain an access token")
@limiter.limit(settings.rate_limit_auth)
def login(
    request: Request, response: Response, payload: UserLogin, db: DbSession
) -> TokenResponse:
    user = AuthService(db).authenticate(payload.email, payload.password)
    return _token_response(user.id, UserRead.model_validate(user))


@router.get("/me", response_model=UserRead, summary="Current account")
def me(current_user: CurrentUser) -> UserRead:
    return UserRead.model_validate(current_user)
