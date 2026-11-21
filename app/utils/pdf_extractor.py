"""Text extraction for supported upload formats."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = {".pdf", ".txt"}


class ExtractionError(Exception):
    """Raised when a document cannot be turned into text."""


@dataclass(slots=True)
class ExtractedPage:
    page_number: int
    text: str


def _extract_pdf(path: Path) -> list[ExtractedPage]:
    try:
        document = fitz.open(path)
    except Exception as exc:  # PyMuPDF raises several low-level error types
        raise ExtractionError("The PDF file is corrupted or unreadable") from exc

    try:
        if document.needs_pass:
            raise ExtractionError("Password-protected PDFs are not supported")
        if document.page_count == 0:
            raise ExtractionError("The PDF contains no pages")
        pages: list[ExtractedPage] = []
        for index, page in enumerate(document, start=1):
            try:
                text = page.get_text("text") or ""
            except Exception as exc:
                raise ExtractionError(f"Failed to read page {index} of the PDF") from exc
            pages.append(ExtractedPage(page_number=index, text=text))
        return pages
    finally:
        document.close()


def _extract_txt(path: Path) -> list[ExtractedPage]:
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise ExtractionError("The TXT file appears to be binary, not plain text")
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return [ExtractedPage(page_number=1, text=raw.decode(encoding))]
        except UnicodeDecodeError:
            continue
    raise ExtractionError("The TXT file encoding could not be detected")


def extract_pages(path: Path, file_type: str) -> list[ExtractedPage]:
    """Extract text per page. ``file_type`` is a lowercase extension like ``.pdf``."""
    suffix = file_type.lower()
    if suffix not in SUPPORTED_TYPES:
        raise ExtractionError(f"Unsupported file type: {suffix}")
    if not path.exists():
        raise ExtractionError("The uploaded file could not be found on disk")
    if path.stat().st_size == 0:
        raise ExtractionError("The uploaded file is empty")

    pages = _extract_pdf(path) if suffix == ".pdf" else _extract_txt(path)
    if not any(page.text.strip() for page in pages):
        raise ExtractionError(
            "No extractable text was found in the document "
            "(scanned PDFs require OCR, which is not supported)"
        )
    return pages
