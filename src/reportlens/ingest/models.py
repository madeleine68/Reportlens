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
    num_tokens: int  # tokens in `text` (the label is not counted)
    context: str = ""  # contextual chunking's label, embedded with the text; "" for naive chunks

    def embedding_text(self) -> str:
        """What gets embedded: the label (if any) followed by the chunk text."""
        return f"{self.context}\n{self.text}" if self.context else self.text


PeriodKind = Literal["year", "quarter", "change", "unknown"]


class TableCell(BaseModel):
    """One numeric cell of a report table, with its row and column labels (Phase 3)."""

    company: str
    report_year: int  # fiscal year of the report the table is in
    table_id: str  # e.g. "rbc-2023-t0042": groups the cells of one table
    table_title: str  # the table's section heading, e.g. "T3 Impact of adjustments"
    page: int
    row_index: int  # 0 = first row below the header
    row_label: str
    row_group: str  # latest label-only row above, e.g. "Adjusted Results"; "" if none
    column_index: int  # 1 = first value column (column 0 holds the row labels)
    column_label: str
    raw_text: str  # the cell as printed, e.g. "$ (367)"
    value: float
    unit: str | None  # "millions", "billions", "thousands", "percent", "bps" or None
    period_year: int | None  # year the column covers, if it names one
    period_kind: PeriodKind
