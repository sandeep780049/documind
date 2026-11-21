"""Document ingestion pipeline.

Upload -> validate -> extract -> clean -> chunk -> embed -> store (pgvector).
Processing status is persisted so failed uploads can be inspected and retried
without creating duplicate document rows.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import BinaryIO

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import (
    DocumentProcessingError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)
from app.models.document import Document, DocumentStatus
from app.repositories.document_repository import DocumentRepository
from app.services.embedding_service import BaseEmbeddingService, EmbeddingError
from app.utils.pdf_extractor import ExtractionError, extract_pages
from app.utils.text_chunker import chunk_pages

logger = logging.getLogger(__name__)

_READ_BLOCK = 1024 * 1024


class DocumentService:
    def __init__(self, db: Session, embedding_service: BaseEmbeddingService) -> None:
        self.db = db
        self.embedding_service = embedding_service
        self.repository = DocumentRepository(db)

    # ------------------------------------------------------------------ upload
    def ingest_upload(
        self, *, user_id: uuid.UUID, filename: str | None, stream: BinaryIO
    ) -> Document:
        safe_name = Path(filename or "upload").name
        extension = Path(safe_name).suffix.lower()
        if extension not in settings.allowed_extension_set:
            allowed = ", ".join(sorted(settings.allowed_extension_set))
            raise UnsupportedMediaTypeError(
                f"Unsupported file type '{extension or 'unknown'}'. Allowed: {allowed}"
            )

        destination = self._destination_for(user_id, extension)
        size = self._store_stream(stream, destination)

        if extension == ".pdf":
            with destination.open("rb") as handle:
                header = handle.read(5)
            if header != b"%PDF-":
                _safe_unlink(destination)
                raise UnsupportedMediaTypeError("The uploaded file is not a valid PDF")

        document = self.repository.create(
            user_id=user_id,
            filename=safe_name,
            stored_path=str(destination),
            file_type=extension,
            file_size=size,
            status=DocumentStatus.PROCESSING,
        )
        self.db.commit()
        self.db.refresh(document)

        return self._process(document)

    def _destination_for(self, user_id: uuid.UUID, extension: str) -> Path:
        directory = Path(settings.upload_dir) / str(user_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{uuid.uuid4().hex}{extension}"

    def _store_stream(self, stream: BinaryIO, destination: Path) -> int:
        total = 0
        try:
            with destination.open("wb") as output:
                while True:
                    block = stream.read(_READ_BLOCK)
                    if not block:
                        break
                    total += len(block)
                    if total > settings.max_upload_size_bytes:
                        raise PayloadTooLargeError(
                            f"File exceeds the {settings.max_upload_size_mb} MB limit"
                        )
                    output.write(block)
        except BaseException:
            _safe_unlink(destination)
            raise
        if total == 0:
            _safe_unlink(destination)
            raise DocumentProcessingError("The uploaded file is empty")
        return total

    # --------------------------------------------------------------- processing
    def _process(self, document: Document) -> Document:
        path = Path(document.stored_path) if document.stored_path else None
        try:
            if path is None:
                raise ExtractionError("The stored file path is missing")

            pages = extract_pages(path, document.file_type)
            drafts = chunk_pages(
                pages,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            if not drafts:
                raise ExtractionError("The document did not produce any text chunks")

            vectors = self.embedding_service.embed_documents([draft.chunk_text for draft in drafts])
            if len(vectors) != len(drafts):
                raise EmbeddingError("Embedding count did not match the number of chunks")

            rows = [
                {
                    "id": uuid.uuid4(),
                    "chunk_index": draft.chunk_index,
                    "chunk_text": draft.chunk_text,
                    "embedding": vector,
                    "page_number": draft.page_number,
                    "chunk_metadata": draft.metadata,
                }
                for draft, vector in zip(drafts, vectors, strict=True)
            ]
            stored = self.repository.replace_chunks(document, rows)
            self.repository.set_status(
                document,
                DocumentStatus.COMPLETED,
                error_message=None,
                page_count=len(pages),
                chunk_count=stored,
            )
            self.db.commit()
            self.db.refresh(document)
            logger.info(
                "Processed document %s (%s chunks, %s pages)",
                document.id,
                stored,
                len(pages),
            )
            return document
        except (ExtractionError, EmbeddingError, DocumentProcessingError, PayloadTooLargeError) as exc:
            self._fail(document, path, str(exc))
            raise DocumentProcessingError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected failure processing document %s", document.id)
            self._fail(document, path, "Unexpected error while processing the document")
            raise DocumentProcessingError(
                "The document could not be processed. Please try again."
            ) from exc

    def _fail(self, document: Document, path: Path | None, message: str) -> None:
        if path is not None:
            _safe_unlink(path)
            document.stored_path = None
        document.error_message = message[:1000]
        try:
            self.repository.replace_chunks(document, [])
            self.repository.set_status(document, DocumentStatus.FAILED, error_message=message[:1000])
            self.db.commit()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Could not persist failure state for document %s", document.id)
            self.db.rollback()

    # ----------------------------------------------------------------- delete
    def delete(self, document: Document) -> None:
        if document.stored_path:
            _safe_unlink(Path(document.stored_path))
        self.repository.delete(document)
        self.db.commit()


def _safe_unlink(path: Path, attempts: int = 20, delay: float = 0.1) -> None:
    """Best-effort file removal that tolerates transient Windows locks."""
    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(delay)
        except OSError:
            # Fall through and retry once more; ignore unrecoverable errors.
            time.sleep(delay)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logging.getLogger(__name__).warning(
            "Could not remove uploaded file %s; leaving it on disk", path
        )
