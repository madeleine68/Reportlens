"""Chunking: split a parsed report into the pieces we embed, search and cite.

- Naive (Phase 1 baseline): the report's blocks are joined into one text and cut every
  `chunk_tokens` tokens, ignoring structure. Chunks may start mid-paragraph or straddle sections.
- Contextual (Phase 2): chunks start at headings and never cut a block, and each carries a short
  label (company, year, section, pages) that is embedded with it.

`chunk()` picks one based on the CHUNKING setting.
"""

import bisect
from collections.abc import Callable
from functools import lru_cache

from transformers import AutoTokenizer

from reportlens.core.config import get_settings
from reportlens.ingest.models import Chunk, ParsedBlock, ParsedDocument

# Maps text to the (start, end) character span of each token.
Tokenizer = Callable[[str], list[tuple[int, int]]]

BLOCK_SEPARATOR = "\n\n"


@lru_cache
def embedding_tokenizer() -> Tokenizer:
    """Token spans from the embedding model's own tokenizer, so sizes match what it reads."""
    hf_tokenizer = AutoTokenizer.from_pretrained(get_settings().embedding_model)
    hf_tokenizer.model_max_length = 10**9  # we tokenize whole reports; silence the length warning

    def tokenize(text: str) -> list[tuple[int, int]]:
        encoded = hf_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return encoded["offset_mapping"]

    return tokenize


def chunk_document(
    doc: ParsedDocument,
    tokenize: Tokenizer | None = None,
    chunk_tokens: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Split a parsed report into overlapping fixed-size token windows, tagged with pages."""
    settings = get_settings()
    tokenize = tokenize or embedding_tokenizer()
    chunk_tokens = chunk_tokens or settings.chunk_tokens
    overlap = settings.chunk_overlap if overlap is None else overlap
    if not 0 <= overlap < chunk_tokens:
        raise ValueError(f"overlap must be in [0, chunk_tokens), got {overlap}")

    # Join blocks, remembering where each one starts so a character offset maps back to a block.
    texts = [block.text for block in doc.blocks]
    full_text = BLOCK_SEPARATOR.join(texts)
    block_starts: list[int] = []
    offset = 0
    for text in texts:
        block_starts.append(offset)
        offset += len(text) + len(BLOCK_SEPARATOR)

    spans = tokenize(full_text)
    chunks: list[Chunk] = []
    step = chunk_tokens - overlap
    for start in range(0, len(spans), step):
        end = min(start + chunk_tokens, len(spans))
        char_start, char_end = spans[start][0], spans[end - 1][1]

        first = bisect.bisect_right(block_starts, char_start) - 1
        last = bisect.bisect_right(block_starts, char_end - 1) - 1
        covered = doc.blocks[first : last + 1]

        chunks.append(
            Chunk(
                chunk_id=f"{doc.company}-{doc.year}-{len(chunks):05d}",
                company=doc.company,
                year=doc.year,
                text=full_text[char_start:char_end],
                pages=sorted({block.page for block in covered}),
                section=covered[0].section,
                num_tokens=end - start,
            )
        )
        if end == len(spans):
            break
    return chunks


# --- Contextual chunking (Phase 2) ------------------------------------------------------------

LABEL_SECTION_CHARS = 80  # long headings are cut in the label so it stays within its reserve


def _page_label(pages: list[int]) -> str:
    return f"Page {pages[0]}" if len(pages) == 1 else f"Pages {pages[0]}-{pages[-1]}"


def context_label(doc: ParsedDocument, section: str | None, pages: list[int]) -> str:
    """Short header embedded with a chunk, e.g. 'RBC 2023 annual report | Section: X | Page 26'."""
    section_text = (section or "unknown")[:LABEL_SECTION_CHARS]
    report = f"{doc.company.upper()} {doc.year} annual report"
    return f"{report} | Section: {section_text} | {_page_label(pages)}"


def _split_oversized(block: ParsedBlock, tokenize: Tokenizer, budget: int) -> list[ParsedBlock]:
    """Split a block bigger than the budget at line breaks (keeping table rows whole).

    A single line that is still too long is cut into fixed token windows as a last resort.
    """
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for line in block.text.split("\n"):
        n = len(tokenize(line))
        if n > budget:  # one enormous line: hard-split it by tokens
            spans = tokenize(line)
            for start in range(0, len(spans), budget):
                window = spans[start : start + budget]
                pieces.append(line[window[0][0] : window[-1][1]])
            continue
        if current and current_tokens + n > budget:
            pieces.append("\n".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += n
    if current:
        pieces.append("\n".join(current))
    return [block.model_copy(update={"text": p}) for p in pieces if p.strip()]


def chunk_contextual(
    doc: ParsedDocument,
    tokenize: Tokenizer | None = None,
    chunk_tokens: int | None = None,
    min_tokens: int | None = None,
    label_reserve: int | None = None,
) -> list[Chunk]:
    """Section-aware chunks: start a new chunk at a heading, never cut a block, add a label.

    A heading only starts a new chunk once the current one has `min_tokens`, so the many small
    headings in annual reports (often just table titles) don't create tiny chunks. Content gets
    `chunk_tokens - label_reserve` tokens, so label + content fit the embedding model.
    """
    settings = get_settings()
    tokenize = tokenize or embedding_tokenizer()
    chunk_tokens = chunk_tokens or settings.chunk_tokens
    min_tokens = settings.chunk_min_tokens if min_tokens is None else min_tokens
    label_reserve = settings.context_reserve_tokens if label_reserve is None else label_reserve
    budget = chunk_tokens - label_reserve
    if budget <= 0:
        raise ValueError(f"label_reserve ({label_reserve}) must be below chunk_tokens")

    # 1. Make every block fit the budget.
    pieces: list[tuple[ParsedBlock, int]] = []
    for block in doc.blocks:
        n = len(tokenize(block.text))
        if n <= budget:
            pieces.append((block, n))
        else:
            pieces.extend(
                (p, len(tokenize(p.text))) for p in _split_oversized(block, tokenize, budget)
            )

    # 2. Pack pieces into chunks, starting a new one at headings or when the budget is full.
    groups: list[list[ParsedBlock]] = []
    current: list[ParsedBlock] = []
    current_tokens = 0
    for block, n in pieces:
        is_heading = block.block_type == "heading"
        if current and (
            current_tokens + n > budget or (is_heading and current_tokens >= min_tokens)
        ):
            groups.append(current)
            current, current_tokens = [], 0
        current.append(block)
        current_tokens += n
    if current:
        groups.append(current)

    # 3. Turn each group into a chunk with its label.
    chunks = []
    for i, group in enumerate(groups):
        text = BLOCK_SEPARATOR.join(b.text for b in group)
        pages = sorted({b.page for b in group})
        chunks.append(
            Chunk(
                chunk_id=f"{doc.company}-{doc.year}-{i:05d}",
                company=doc.company,
                year=doc.year,
                text=text,
                context=context_label(doc, group[0].section, pages),
                pages=pages,
                section=group[0].section,
                num_tokens=len(tokenize(text)),
            )
        )
    return chunks


def chunk(doc: ParsedDocument) -> list[Chunk]:
    """Chunk a report with the strategy chosen by the CHUNKING setting."""
    if get_settings().chunking == "contextual":
        return chunk_contextual(doc)
    return chunk_document(doc)
