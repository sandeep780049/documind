"""Unit tests for the ingestion pipeline (no database required)."""

from __future__ import annotations

import io

import pytest

from app.utils.pdf_extractor import ExtractionError, extract_pages
from app.utils.text_chunker import chunk_pages, clean_text, split_text
from tests.fakes import FakeEmbeddingService


def test_clean_text_normalises_whitespace():
    dirty = "Heading\r\n\r\n\r\n\r\nBody   text\twith\tspaces\x00and control"
    cleaned = clean_text(dirty)
    assert "\r" not in cleaned
    assert "\x00" not in cleaned
    assert "Body text with spaces and control" in cleaned


def test_split_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = split_text(text, chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 1
    for chunk, start, end in chunks:
        assert len(chunk) <= 400
        assert start < end
    # Consecutive chunks share content.
    assert chunks[0][1] < chunks[1][1] < chunks[0][2]


def test_split_text_rejects_bad_configuration():
    with pytest.raises(ValueError):
        split_text("text", chunk_size=100, chunk_overlap=100)


def test_chunk_pages_preserves_page_numbers():
    from app.utils.pdf_extractor import ExtractedPage

    pages = [
        ExtractedPage(page_number=1, text="Page one content. " * 50),
        ExtractedPage(page_number=2, text="Page two content. " * 50),
    ]
    drafts = chunk_pages(pages, chunk_size=300, chunk_overlap=50)
    assert {draft.page_number for draft in drafts} == {1, 2}
    assert [draft.chunk_index for draft in drafts] == list(range(len(drafts)))
    assert all(draft.metadata["page_number"] == draft.page_number for draft in drafts)


def test_extract_txt_and_pdf(tmp_path):
    text_file = tmp_path / "doc.txt"
    text_file.write_text("Hello DocuMind", encoding="utf-8")
    pages = extract_pages(text_file, ".txt")
    assert pages[0].text == "Hello DocuMind"
    assert pages[0].page_number == 1

    import fitz

    pdf_path = tmp_path / "doc.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "PDF body text")
    document.save(pdf_path)
    document.close()
    pdf_pages = extract_pages(pdf_path, ".pdf")
    assert "PDF body text" in pdf_pages[0].text


def test_extract_rejects_empty_and_corrupt_files(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with pytest.raises(ExtractionError):
        extract_pages(empty, ".txt")

    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"%PDF-1.4 definitely broken")
    with pytest.raises(ExtractionError):
        extract_pages(corrupt, ".pdf")


def test_fake_embedding_ranks_relevant_text_higher():
    service = FakeEmbeddingService(dimension=384)
    query = service.embed_query("remote work policy")
    relevant = service.embed_documents(["remote work policy allows three days"])[0]
    unrelated = service.embed_documents(["annual tax filing deadlines"])[0]
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert dot(query, relevant) > dot(query, unrelated)
