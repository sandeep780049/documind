"""Document routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status

from app.api.dependencies import CurrentUser, DbSession, EmbeddingProviderDep, PaginationDep
from app.core.config import settings
from app.core.exceptions import NotFoundError
from app.core.rate_limit import limiter
from app.models.document import DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentRead,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF or TXT document",
)
@limiter.limit(settings.rate_limit_upload)
def upload_document(
    request: Request,
    response: Response,
    current_user: CurrentUser,
    db: DbSession,
    embedding_service: EmbeddingProviderDep,
    file: Annotated[UploadFile, File(description="PDF or TXT file")],
) -> DocumentRead:
    document = DocumentService(db, embedding_service).ingest_upload(
        user_id=current_user.id,
        filename=file.filename,
        stream=file.file,
    )
    return DocumentRead.model_validate(document)


@router.get("", response_model=DocumentListResponse, summary="List uploaded documents")
def list_documents(
    current_user: CurrentUser,
    db: DbSession,
    pagination: PaginationDep,
    processing_status: Annotated[
        DocumentStatus | None, Query(description="Filter by processing status")
    ] = None,
) -> DocumentListResponse:
    items, total = DocumentRepository(db).list_owned(
        current_user.id,
        page=pagination.page,
        limit=pagination.limit,
        status=processing_status,
    )
    return DocumentListResponse.build(
        [DocumentRead.model_validate(item) for item in items],
        total=total,
        page=pagination.page,
        limit=pagination.limit,
    )


@router.get("/{document_id}", response_model=DocumentRead, summary="Get a document")
def get_document(
    document_id: uuid.UUID, current_user: CurrentUser, db: DbSession
) -> DocumentRead:
    document = DocumentRepository(db).get_owned(document_id, current_user.id)
    if document is None:
        raise NotFoundError("Document not found")
    return DocumentRead.model_validate(document)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and its chunks",
)
def delete_document(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    db: DbSession,
    embedding_service: EmbeddingProviderDep,
) -> DocumentDeleteResponse:
    document = DocumentRepository(db).get_owned(document_id, current_user.id)
    if document is None:
        raise NotFoundError("Document not found")
    DocumentService(db, embedding_service).delete(document)
    return DocumentDeleteResponse(document_id=document_id)
