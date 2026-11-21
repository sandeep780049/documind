"""Persistence layer for conversations and messages."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.models.conversation import Conversation
from app.models.message import Message, MessageRole


class ChatRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------ conversations
    def create_conversation(self, user_id: uuid.UUID, title: str) -> Conversation:
        conversation = Conversation(user_id=user_id, title=title[:255])
        self.db.add(conversation)
        self.db.flush()
        return conversation

    def get_owned(self, conversation_id: uuid.UUID, user_id: uuid.UUID) -> Conversation | None:
        return self.db.scalar(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        )

    def list_owned(
        self, user_id: uuid.UUID, *, page: int, limit: int
    ) -> tuple[list[tuple[Conversation, int, str | None]], int]:
        message_count = (
            select(func.count(Message.id))
            .where(Message.conversation_id == Conversation.id)
            .correlate(Conversation)
            .scalar_subquery()
        )
        last_message = (
            select(Message.content)
            .where(Message.conversation_id == Conversation.id)
            .order_by(Message.created_at.desc())
            .limit(1)
            .correlate(Conversation)
            .scalar_subquery()
        )
        total = int(
            self.db.scalar(
                select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
            )
            or 0
        )
        rows = self.db.execute(
            select(Conversation, message_count, last_message)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.created_at.desc())
            .offset((page - 1) * limit)
            .limit(limit)
        ).all()
        items = [(row[0], int(row[1] or 0), row[2]) for row in rows]
        return items, total

    def delete_conversation(self, conversation: Conversation) -> None:
        self.db.delete(conversation)
        self.db.flush()

    # ---------------------------------------------------------------- messages
    def add_message(
        self,
        conversation_id: uuid.UUID,
        role: MessageRole,
        content: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources,
        )
        self.db.add(message)
        self.db.flush()
        return message
