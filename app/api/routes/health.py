"""Health endpoints used by Docker, Render and uptime monitors."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status

from app.core.config import settings
from app.core.database import check_database
from app.core.exceptions import AppError
from app.schemas.health import ComponentStatus, HealthResponse
from app.services.embedding_service import get_embedding_service
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])

VERSION = "1.0.0"


def _build_health() -> HealthResponse:
    components: dict[str, ComponentStatus] = {}

    try:
        check_database()
        components["database"] = ComponentStatus(status="ok", detail="PostgreSQL reachable")
    except Exception as exc:
        logger.warning("Health check: database unavailable: %s", exc)
        components["database"] = ComponentStatus(
            status="unavailable", detail="PostgreSQL unreachable"
        )

    try:
        ok, detail = get_embedding_service().health()
        components["embedding"] = ComponentStatus(status="ok" if ok else "degraded", detail=detail)
    except AppError as exc:
        components["embedding"] = ComponentStatus(status="unavailable", detail=exc.message)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Health check: embedding unavailable: %s", exc)
        components["embedding"] = ComponentStatus(
            status="unavailable", detail="Embedding provider unavailable"
        )

    try:
        ok, detail = get_llm_service().health()
        components["llm"] = ComponentStatus(status="ok" if ok else "degraded", detail=detail)
    except AppError as exc:
        components["llm"] = ComponentStatus(status="unavailable", detail=exc.message)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Health check: LLM unavailable: %s", exc)
        components["llm"] = ComponentStatus(status="unavailable", detail="LLM provider unavailable")

    healthy = all(component.status == "ok" for component in components.values())
    return HealthResponse(
        status="ok" if healthy else "degraded",
        app=settings.app_name,
        version=VERSION,
        environment=settings.environment,
        **components,
    )


def _health(request_response: Response) -> HealthResponse:
    payload = _build_health()
    # The service stays reachable while dependencies degrade so that the
    # platform health check does not kill the container during an outage.
    request_response.status_code = status.HTTP_200_OK
    return payload


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health(response: Response) -> HealthResponse:
    return _health(response)


@router.get(
    f"{settings.api_v1_prefix}/health",
    response_model=HealthResponse,
    summary="Health check (versioned)",
)
def api_health(response: Response) -> HealthResponse:
    return _health(response)
