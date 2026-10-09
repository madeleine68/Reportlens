"""Data models passed between ingest steps (parse -> chunk -> embed -> load)."""

from typing import Literal

from pydantic import BaseModel

BlockType = Literal["heading", "text", "list_item", "table", "caption", "footnote"]


class ParsedBlock(BaseModel):
    """One piece of document content, tagged with where it came from."""

    text: str
    page: int  # 1-based page number in the PDF
    section: str | None  # nearest heading above this block, if any
    block_type: BlockType


class ParsedDocument(BaseModel):
    """A whole annual report as an ordered list of blocks."""

    company: str
    year: int
    source_path: str
    sha256: str  # fingerprint of the PDF bytes, used to skip unchanged files
    converter_version: str  # Docling settings used; see parse.CONVERTER_VERSION
    extract_version: str  # block extraction logic used; see parse.EXTRACT_VERSION
    num_pages: int
    blocks: list[ParsedBlock]


class Chunk(BaseModel):
    """A retrievable piece of a report: the unit we embed, search and cite."""

    chunk_id: str  # e.g. "rbc-2023-00042"
    company: str
    year: int
    text: str
    pages: list[int]  # every page the chunk's text comes from, ascending
    section: str | None  # section of the block the chunk starts in
    num_tokens: int
