"""Deterministic, word-boundary-aware text chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.utils.pdf_extractor import ExtractedPage

_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalise whitespace while preserving paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


@dataclass(slots=True)
class ChunkDraft:
    chunk_index: int
    chunk_text: str
    page_number: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    """Split text into ``(chunk, start, end)`` triples.

    Boundaries prefer whitespace so words are never cut in half. The overlap
    is applied between consecutive chunks and the function always advances,
    so it cannot loop forever.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            break_at = window.rfind(" ")
            newline_at = window.rfind("\n")
            break_at = max(break_at, newline_at)
            # Only accept a boundary in the second half of the window so that
            # chunks stay reasonably large.
            if break_at >= chunk_size // 2:
                end = start + break_at
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        if end >= length:
            break
        next_start = max(end - chunk_overlap, start + 1)
        # Snap the overlap start to the beginning of a word for readability.
        if next_start > start and next_start < length and text[next_start - 1].isalnum():
            space = text.find(" ", next_start)
            if 0 <= space < end:
                next_start = space + 1
        start = next_start
    return chunks


def chunk_pages(
    pages: list[ExtractedPage],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkDraft]:
    """Clean and chunk extracted pages, preserving page numbers and offsets."""
    drafts: list[ChunkDraft] = []
    chunk_index = 0
    for page in pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue
        for chunk, start, end in split_text(cleaned, chunk_size, chunk_overlap):
            drafts.append(
                ChunkDraft(
                    chunk_index=chunk_index,
                    chunk_text=chunk,
                    page_number=page.page_number,
                    metadata={
                        "page_number": page.page_number,
                        "chunk_index": chunk_index,
                        "char_start": start,
                        "char_end": end,
                        "word_count": len(chunk.split()),
                    },
                )
            )
            chunk_index += 1
    return drafts
