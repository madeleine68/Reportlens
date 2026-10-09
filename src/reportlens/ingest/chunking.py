"""Naive chunking (Phase 1 baseline): fixed token windows with overlap, ignoring structure.

A report's blocks are joined into one text and cut every `chunk_tokens` tokens. Chunks may start
mid-paragraph or straddle sections; Phase 2's contextual chunking is measured against this.
"""

import bisect
from collections.abc import Callable
from functools import lru_cache

from transformers import AutoTokenizer

from reportlens.core.config import get_settings
from reportlens.ingest.models import Chunk, ParsedDocument

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
