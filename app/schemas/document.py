"""Document schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from math import ceil

from pydantic import BaseModel, ConfigDict, Field

from app.models.document import DocumentStatus


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: str
    file_size: int
    processing_status: DocumentStatus
    page_count: int
    chunk_count: int
    error_message: str | None = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentRead]
    total: int
    page: int
    limit: int
    pages: int

    @classmethod
    def build(
        cls, items: list[DocumentRead], total: int, page: int, limit: int
    ) -> "DocumentListResponse":
        return cls(
            items=items,
            total=total,
            page=page,
            limit=limit,
            pages=ceil(total / limit) if limit else 0,
        )


class DocumentDeleteResponse(BaseModel):
    message: str = "Document deleted successfully"
    document_id: uuid.UUID


class ChunkPreview(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_index: int
    page_number: int | None
    chunk_text: str = Field(description="Chunk contents truncated for preview")
