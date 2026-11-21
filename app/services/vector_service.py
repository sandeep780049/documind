"""Vector similarity search backed by pgvector."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk
from app.models.document import Document, DocumentStatus


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    page_number: int | None
    chunk_index: int
    chunk_text: str
    similarity: float


class VectorService:
    """Read-only retrieval service.

    Ownership is enforced in SQL by joining ``documents`` and filtering on
    ``documents.user_id``, so a query can never surface another user's chunks.
    """

    def search(
        self,
        db: Session,
        *,
        user_id: uuid.UUID,
        query_embedding: Sequence[float],
        top_k: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        distance = DocumentChunk.embedding.cosine_distance(list(query_embedding)).label("distance")
        filters = [
            Document.user_id == user_id,
            Document.processing_status == DocumentStatus.COMPLETED,
        ]
        if document_ids:
            filters.append(Document.id.in_(document_ids))

        statement = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.page_number,
                DocumentChunk.chunk_index,
                DocumentChunk.chunk_text,
                Document.filename,
                distance,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(*filters)
            .order_by(distance)
            .limit(top_k)
        )

        results: list[RetrievedChunk] = []
        for row in db.execute(statement).all():
            similarity = max(0.0, min(1.0, 1.0 - float(row.distance)))
            results.append(
                RetrievedChunk(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    document_name=row.filename,
                    page_number=row.page_number,
                    chunk_index=row.chunk_index,
                    chunk_text=row.chunk_text,
                    similarity=round(similarity, 4),
                )
            )
        return results
