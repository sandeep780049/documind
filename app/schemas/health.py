"""Health check schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ComponentStatus(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    version: str
    environment: str
    database: ComponentStatus
    embedding: ComponentStatus
    llm: ComponentStatus
