"""Chat, retrieval and conversation schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from math import ceil

from pydantic import BaseModel, ConfigDict, Field

from app.models.message import MessageRole


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    conversation_id: uuid.UUID | None = Field(
        default=None, description="Existing conversation to continue"
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Override retrieval depth")

    def normalized_question(self) -> str:
        return " ".join(self.question.split())


class SourceReference(BaseModel):
    document_id: uuid.UUID
    document_name: str
    page_number: int | None = None
    chunk_index: int
    similarity: float = Field(ge=0.0, le=1.0)
    snippet: str


class AskResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    answer: str
    sources: list[SourceReference]
    model: str
    embedding_model: str
    retrieved_chunks: int
    latency_ms: int


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: MessageRole
    content: str
    sources: list[dict] | None = None
    created_at: datetime


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    message_count: int = 0
    last_message_preview: str | None = None


class ConversationDetail(ConversationSummary):
    messages: list[MessageRead] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    items: list[ConversationSummary]
    total: int
    page: int
    limit: int
    pages: int

    @classmethod
    def build(
        cls, items: list[ConversationSummary], total: int, page: int, limit: int
    ) -> "ConversationListResponse":
        return cls(
            items=items,
            total=total,
            page=page,
            limit=limit,
            pages=ceil(total / limit) if limit else 0,
        )


class ConversationDeleteResponse(BaseModel):
    message: str = "Conversation deleted successfully"
    conversation_id: uuid.UUID
