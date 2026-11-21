"""Persistence layer for documents and their chunks."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, insert, select
from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk
from app.models.document import Document, DocumentStatus


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------ write
    def create(
        self,
        *,
        user_id: uuid.UUID,
        filename: str,
        stored_path: str,
        file_type: str,
        file_size: int,
        status: DocumentStatus = DocumentStatus.PENDING,
    ) -> Document:
        document = Document(
            user_id=user_id,
            filename=filename,
            stored_path=stored_path,
            file_type=file_type,
            file_size=file_size,
            processing_status=status,
        )
        self.db.add(document)
        self.db.flush()
        return document

    def set_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
        page_count: int | None = None,
        chunk_count: int | None = None,
    ) -> Document:
        document.processing_status = status
        document.error_message = error_message
        if page_count is not None:
            document.page_count = page_count
        if chunk_count is not None:
            document.chunk_count = chunk_count
        self.db.flush()
        return document

    def replace_chunks(self, document: Document, chunks: Sequence[dict[str, Any]]) -> int:
        """Idempotently replace every chunk of a document."""
        self.db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
        if not chunks:
            self.db.flush()
            return 0
        rows = [{**chunk, "document_id": document.id} for chunk in chunks]
        self.db.execute(insert(DocumentChunk), rows)
        self.db.flush()
        return len(rows)

    def delete(self, document: Document) -> None:
        self.db.delete(document)
        self.db.flush()

    # ------------------------------------------------------------------- read
    def get_owned(self, document_id: uuid.UUID, user_id: uuid.UUID) -> Document | None:
        return self.db.scalar(
            select(Document).where(Document.id == document_id, Document.user_id == user_id)
        )

    def list_owned(
        self,
        user_id: uuid.UUID,
        *,
        page: int,
        limit: int,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], int]:
        filters = [Document.user_id == user_id]
        if status is not None:
            filters.append(Document.processing_status == status)
        total = self.db.scalar(select(func.count()).select_from(Document).where(*filters)) or 0
        items = list(
            self.db.scalars(
                select(Document)
                .where(*filters)
                .order_by(Document.created_at.desc())
                .offset((page - 1) * limit)
                .limit(limit)
            )
        )
        return items, int(total)

    def count_chunks(self, document_id: uuid.UUID) -> int:
        return int(
            self.db.scalar(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
            )
            or 0
        )
