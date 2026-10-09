import re

import pytest

from reportlens.ingest.chunking import chunk_document, embedding_tokenizer
from reportlens.ingest.models import ParsedBlock, ParsedDocument


def words(text: str) -> list[tuple[int, int]]:
    """Test tokenizer: one token per whitespace-separated word."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _doc(*blocks: tuple[str, int, str]) -> ParsedDocument:
    return ParsedDocument(
        company="rbc",
        year=2023,
        source_path="data/raw/rbc/2023.pdf",
        sha256="x",
        converter_version="t",
        extract_version="t",
        num_pages=max((page for _, page, _ in blocks), default=0),
        blocks=[ParsedBlock(text=t, page=p, section=s, block_type="text") for t, p, s in blocks],
    )


def _numbered_words(first: int, last: int) -> str:
    return " ".join(f"w{i}" for i in range(first, last + 1))


def test_windows_have_fixed_size_and_overlap() -> None:
    doc = _doc((_numbered_words(1, 25), 1, "Intro"))
    chunks = chunk_document(doc, tokenize=words, chunk_tokens=10, overlap=3)

    assert [c.text for c in chunks] == [
        _numbered_words(1, 10),
        _numbered_words(8, 17),
        _numbered_words(15, 24),
        _numbered_words(22, 25),
    ]
    assert [c.num_tokens for c in chunks] == [10, 10, 10, 4]


def test_no_tail_chunk_when_last_window_reaches_the_end() -> None:
    doc = _doc((_numbered_words(1, 17), 1, "Intro"))
    chunks = chunk_document(doc, tokenize=words, chunk_tokens=10, overlap=3)
    assert [c.text for c in chunks] == [_numbered_words(1, 10), _numbered_words(8, 17)]


def test_chunk_spanning_blocks_lists_all_their_pages() -> None:
    doc = _doc(
        ("a b c", 4, "Revenue"),
        ("d e f", 5, "Revenue"),
        ("g h i", 7, "Expenses"),
    )
    chunks = chunk_document(doc, tokenize=words, chunk_tokens=5, overlap=1)

    assert [(c.text, c.pages, c.section) for c in chunks] == [
        ("a b c\n\nd e", [4, 5], "Revenue"),
        ("e f\n\ng h i", [5, 7], "Revenue"),
    ]


def test_section_comes_from_the_block_the_chunk_starts_in() -> None:
    doc = _doc(("a b", 1, "Revenue"), ("c d e f", 2, "Expenses"))
    chunks = chunk_document(doc, tokenize=words, chunk_tokens=3, overlap=0)
    assert [c.section for c in chunks] == ["Revenue", "Expenses"]


def test_chunk_ids_are_stable_and_ordered() -> None:
    doc = _doc((_numbered_words(1, 9), 1, "Intro"))
    ids = [c.chunk_id for c in chunk_document(doc, tokenize=words, chunk_tokens=4, overlap=1)]
    assert ids == ["rbc-2023-00000", "rbc-2023-00001", "rbc-2023-00002"]


def test_empty_document_gives_no_chunks() -> None:
    assert chunk_document(_doc(), tokenize=words, chunk_tokens=10, overlap=2) == []


@pytest.mark.parametrize("overlap", [-1, 10, 11])
def test_rejects_overlap_outside_window(overlap: int) -> None:
    doc = _doc(("a b c", 1, "Intro"))
    with pytest.raises(ValueError):
        chunk_document(doc, tokenize=words, chunk_tokens=10, overlap=overlap)


def test_defaults_come_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from reportlens.core.config import get_settings

    monkeypatch.setenv("CHUNK_TOKENS", "4")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    get_settings.cache_clear()
    chunks = chunk_document(_doc((_numbered_words(1, 8), 1, "Intro")), tokenize=words)
    assert [c.num_tokens for c in chunks] == [4, 4]


@pytest.mark.slow
def test_real_tokenizer_keeps_chunks_within_model_limit() -> None:
    """Uses the bge-small tokenizer (downloads on first run)."""
    text = "Net income was $16.2 billion, up 3% from last year. " * 200
    chunks = chunk_document(_doc((text, 1, "Results")), tokenize=embedding_tokenizer())

    assert len(chunks) > 1
    assert all(c.num_tokens <= 450 for c in chunks)
    # Re-tokenizing the chunk text must also fit the model's 512-token window.
    assert all(len(embedding_tokenizer()(c.text)) <= 510 for c in chunks)
