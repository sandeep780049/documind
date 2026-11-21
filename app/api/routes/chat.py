"""Chat and conversation history routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request, Response

from app.api.dependencies import (
    CurrentUser,
    DbSession,
    EmbeddingProviderDep,
    LLMProviderDep,
    PaginationDep,
    VectorServiceDep,
)
from app.core.config import settings
from app.core.exceptions import NotFoundError
from app.core.rate_limit import limiter
from app.repositories.chat_repository import ChatRepository
from app.schemas.chat import (
    AskRequest,
    AskResponse,
    ConversationDeleteResponse,
    ConversationDetail,
    ConversationListResponse,
    ConversationSummary,
    MessageRead,
)
from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/ask", response_model=AskResponse, summary="Ask a question about your documents")
@limiter.limit(settings.rate_limit_chat)
def ask_question(
    request: Request,
    response: Response,
    payload: AskRequest,
    current_user: CurrentUser,
    db: DbSession,
    embedding_service: EmbeddingProviderDep,
    llm_service: LLMProviderDep,
    vector_service: VectorServiceDep,
) -> AskResponse:
    service = RAGService(db, embedding_service, llm_service, vector_service)
    return service.ask(user=current_user, payload=payload)


@router.get(
    "/history",
    response_model=ConversationListResponse,
    summary="List conversations",
)
def list_history(
    current_user: CurrentUser, db: DbSession, pagination: PaginationDep
) -> ConversationListResponse:
    rows, total = ChatRepository(db).list_owned(
        current_user.id, page=pagination.page, limit=pagination.limit
    )
    items = [
        ConversationSummary(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            message_count=message_count,
            last_message_preview=(preview[:160] if preview else None),
        )
        for conversation, message_count, preview in rows
    ]
    return ConversationListResponse.build(
        items, total=total, page=pagination.page, limit=pagination.limit
    )


@router.get(
    "/history/{conversation_id}",
    response_model=ConversationDetail,
    summary="Get one conversation with messages",
)
def get_conversation(
    conversation_id: uuid.UUID, current_user: CurrentUser, db: DbSession
) -> ConversationDetail:
    conversation = ChatRepository(db).get_owned(conversation_id, current_user.id)
    if conversation is None:
        raise NotFoundError("Conversation not found")
    messages = [MessageRead.model_validate(message) for message in conversation.messages]
    return ConversationDetail(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        message_count=len(messages),
        last_message_preview=(messages[-1].content[:160] if messages else None),
        messages=messages,
    )


@router.delete(
    "/history/{conversation_id}",
    response_model=ConversationDeleteResponse,
    summary="Delete a conversation",
)
def delete_conversation(
    conversation_id: uuid.UUID, current_user: CurrentUser, db: DbSession
) -> ConversationDeleteResponse:
    repository = ChatRepository(db)
    conversation = repository.get_owned(conversation_id, current_user.id)
    if conversation is None:
        raise NotFoundError("Conversation not found")
    repository.delete_conversation(conversation)
    db.commit()
    return ConversationDeleteResponse(conversation_id=conversation_id)
