import re

import pytest

from reportlens.ingest.chunking import (
    chunk,
    chunk_contextual,
    chunk_document,
    chunk_tables,
    context_label,
    embedding_tokenizer,
    split_table_header,
    table_pieces,
)
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


# --- Contextual chunking -----------------------------------------------------------------------


def _blocks_doc(*blocks: tuple[str, int, str | None, str]) -> ParsedDocument:
    """Blocks as (text, page, section, block_type)."""
    return ParsedDocument(
        company="bce",
        year=2024,
        source_path="data/raw/bce/2024.pdf",
        sha256="x",
        converter_version="t",
        extract_version="t",
        num_pages=max(page for _, page, _, _ in blocks),
        blocks=[ParsedBlock(text=t, page=p, section=s, block_type=bt) for t, p, s, bt in blocks],
    )


def _contextual(doc: ParsedDocument, budget: int, min_tokens: int):
    # label_reserve=0 so `budget` is the content budget, in whitespace "tokens".
    return chunk_contextual(
        doc, tokenize=words, chunk_tokens=budget, min_tokens=min_tokens, label_reserve=0
    )


def test_heading_starts_a_new_chunk_once_the_current_one_is_big_enough() -> None:
    doc = _blocks_doc(
        ("Revenue", 1, "Revenue", "heading"),
        ("revenue grew five percent", 1, "Revenue", "text"),
        ("Expenses", 2, "Expenses", "heading"),
        ("costs fell", 2, "Expenses", "text"),
    )
    chunks = _contextual(doc, budget=50, min_tokens=3)

    assert [(c.text, c.section, c.pages) for c in chunks] == [
        ("Revenue\n\nrevenue grew five percent", "Revenue", [1]),
        ("Expenses\n\ncosts fell", "Expenses", [2]),
    ]


def test_small_sections_are_merged_instead_of_making_tiny_chunks() -> None:
    doc = _blocks_doc(
        ("Table 18", 1, "Table 18", "heading"),
        ("a b", 1, "Table 18", "text"),
        ("Table 19", 1, "Table 19", "heading"),
        ("c d", 1, "Table 19", "text"),
    )
    [only] = _contextual(doc, budget=50, min_tokens=20)
    assert only.text == "Table 18\n\na b\n\nTable 19\n\nc d"
    assert only.section == "Table 18"


def test_blocks_are_never_cut_and_chunks_fill_up_to_the_budget() -> None:
    doc = _blocks_doc(
        ("one two three", 1, "S", "text"),
        ("four five six", 1, "S", "text"),
        ("seven eight", 2, "S", "text"),
    )
    chunks = _contextual(doc, budget=6, min_tokens=0)
    assert [c.text for c in chunks] == ["one two three\n\nfour five six", "seven eight"]
    assert [c.num_tokens for c in chunks] == [6, 2]


def test_oversized_table_is_split_at_line_breaks() -> None:
    table = "Metric 2024 2023\nRevenue 10 9\nNet income 3 2\nEPS 1.1 0.9"
    doc = _blocks_doc((table, 8, "Results", "table"))
    chunks = _contextual(doc, budget=7, min_tokens=0)

    assert [c.text for c in chunks] == [
        "Metric 2024 2023\nRevenue 10 9",
        "Net income 3 2\nEPS 1.1 0.9",
    ]
    assert all(c.pages == [8] for c in chunks)


def test_label_names_company_year_section_and_pages() -> None:
    doc = _blocks_doc(("x", 3, "Wireless", "text"), ("y", 4, "Wireless", "text"))
    [c] = _contextual(doc, budget=50, min_tokens=0)

    assert c.context == "BCE 2024 annual report | Section: Wireless | Pages 3-4"
    assert c.embedding_text() == "BCE 2024 annual report | Section: Wireless | Pages 3-4\nx\n\ny"
    assert context_label(doc, None, [7]).endswith("Section: unknown | Page 7")


def test_label_reserve_shrinks_the_content_budget() -> None:
    doc = _blocks_doc(("a b c d e f", 1, "S", "text"))
    chunks = chunk_contextual(doc, tokenize=words, chunk_tokens=8, min_tokens=0, label_reserve=4)
    assert [c.num_tokens for c in chunks] == [4, 2]


def test_naive_chunks_have_no_label() -> None:
    [c] = chunk_document(_doc(("a b c", 1, "Intro")), tokenize=words, chunk_tokens=10, overlap=0)
    assert c.context == "" and c.embedding_text() == "a b c"


def test_chunk_picks_strategy_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from reportlens.core.config import get_settings
    from reportlens.ingest import chunking

    seen: list[str] = []
    monkeypatch.setattr(chunking, "chunk_document", lambda doc: seen.append("naive"))
    monkeypatch.setattr(chunking, "chunk_contextual", lambda doc: seen.append("contextual"))
    monkeypatch.setattr(chunking, "chunk_tables", lambda doc: seen.append("tables"))
    for strategy in ("naive", "contextual", "tables"):
        monkeypatch.setenv("CHUNKING", strategy)
        get_settings.cache_clear()
        chunk(_doc(("a", 1, "S")))
    assert seen == ["naive", "contextual", "tables"]


@pytest.mark.slow
def test_contextual_label_and_text_fit_the_embedding_model() -> None:
    """Real bge tokenizer: label + content must stay under the model's 512-token window."""
    long_section = "Management's discussion and analysis of financial condition " * 4
    text = "Net income was $16.2 billion, up 3% from last year. " * 120
    doc = _blocks_doc((text, 5, long_section, "text"))
    chunks = chunk_contextual(doc, tokenize=embedding_tokenizer())

    assert len(chunks) > 1
    assert all(len(embedding_tokenizer()(c.embedding_text())) <= 510 for c in chunks)


# --- Table-aware chunking ---------------------------------------------------------------------

TABLE = "\n".join(
    [
        "| ($ millions) | 2023 | 2022 |",
        "| --- | --- | --- |",
        "| Net income | 7,528 | 10,174 |",
        "| NCI | 118 | 258 |",
        "| Equity holders | 7,410 | 9,916 |",
    ]
)


def test_split_table_header_ends_at_the_divider_row() -> None:
    header, body = split_table_header(TABLE)
    assert header == "| ($ millions) | 2023 | 2022 |\n| --- | --- | --- |"
    assert body.splitlines()[0] == "| Net income | 7,528 | 10,174 |"


def test_table_without_divider_has_no_header() -> None:
    assert split_table_header("Revenue 10 9\nEPS 1 2") == ("", "Revenue 10 9\nEPS 1 2")


def test_small_table_is_one_piece_with_its_section_heading() -> None:
    block = ParsedBlock(text=TABLE, page=22, section="T3 Financial results", block_type="table")
    assert table_pieces(block, words, budget=100) == [f"T3 Financial results\n{TABLE}"]


def test_split_table_repeats_heading_and_header_in_every_piece() -> None:
    block = ParsedBlock(text=TABLE, page=22, section="Results", block_type="table")
    # Prefix ("Results" + header) is 16 words, leaving 16 for rows of 8, 7 and 8 words.
    pieces = table_pieces(block, words, budget=32)

    assert len(pieces) == 2
    header = "Results\n| ($ millions) | 2023 | 2022 |\n| --- | --- | --- |"
    assert all(p.startswith(header + "\n| ") for p in pieces)
    assert [p.splitlines()[-1] for p in pieces] == TABLE.splitlines()[3:]
    assert all(len(words(p)) <= 32 for p in pieces)


def test_header_too_big_to_repeat_falls_back_to_line_splits() -> None:
    block = ParsedBlock(text=TABLE, page=22, section="Results", block_type="table")
    pieces = table_pieces(block, words, budget=30)  # prefix (16) > half the budget
    assert "\n".join(pieces) == TABLE


def test_chunk_tables_keeps_text_windows_and_adds_one_chunk_per_table() -> None:
    doc = _blocks_doc(
        ("intro words here", 1, "Intro", "text"),
        (TABLE, 22, "Results", "table"),
        ("closing words", 23, "Outlook", "text"),
    )
    chunks = chunk_tables(doc, tokenize=words, chunk_tokens=100, overlap=0)

    assert [c.chunk_id for c in chunks] == ["bce-2024-00000", "bce-2024-00001"]
    text_chunk, table_chunk = chunks
    assert "|" not in text_chunk.text  # the table isn't repeated inside the text windows
    assert text_chunk.text == "intro words here\n\nclosing words"
    assert table_chunk.text == f"Results\n{TABLE}"
    assert table_chunk.pages == [22] and table_chunk.section == "Results"
    assert table_chunk.num_tokens == len(words(table_chunk.text))
