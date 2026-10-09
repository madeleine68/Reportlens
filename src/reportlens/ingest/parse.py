"""Parse annual report PDFs into page-tagged blocks with Docling, caching the result as JSON.

Phase 1 runs Docling in a light mode (no OCR, no table-structure model) because the reports are
digital PDFs and the dev machine is a CPU-only Intel Mac. See DECISIONS.md.

Two cache files per PDF, in data/parsed/{company}/:
- {year}.docling.json: Docling's raw output (slow to produce: ~30 min per report on CPU)
- {year}.json: our ParsedDocument, extracted from the Docling output (fast)
So changing extraction logic only re-runs extraction, not Docling.
"""

import hashlib
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ContentLayer, DocItem, DocItemLabel, DoclingDocument, TableItem

from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.ingest.models import BlockType, ParsedBlock, ParsedDocument

log = get_logger(__name__)

# Bump when Docling converter options change (forces the slow Docling step to re-run).
CONVERTER_VERSION = "1-light"
# Bump when blocks_from_docling changes (re-extracts from the cached Docling output).
EXTRACT_VERSION = "2"

# A text item counts as inside a table if at least this share of its box overlaps the table's.
_TABLE_OVERLAP = 0.5

_LABEL_TO_TYPE: dict[DocItemLabel, BlockType] = {
    DocItemLabel.TITLE: "heading",
    DocItemLabel.SECTION_HEADER: "heading",
    DocItemLabel.TEXT: "text",
    DocItemLabel.PARAGRAPH: "text",
    DocItemLabel.REFERENCE: "text",
    DocItemLabel.LIST_ITEM: "list_item",
    DocItemLabel.TABLE: "table",
    DocItemLabel.CAPTION: "caption",
    DocItemLabel.FOOTNOTE: "footnote",
}


def company_year_from_path(pdf_path: Path) -> tuple[str, int]:
    """Read company and fiscal year from a data/raw/{company}/{year}.pdf path."""
    company, stem = pdf_path.parent.name, pdf_path.stem
    if not company or not stem.isdigit() or len(stem) != 4:
        raise ValueError(f"Expected data/raw/{{company}}/{{year}}.pdf, got {pdf_path}")
    return company, int(stem)


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, read in chunks so large PDFs stay off the heap."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _item_text(item: DocItem) -> str:
    return getattr(item, "text", "").strip()


def _containing_table(item: DocItem, tables_on_page: list[TableItem]) -> TableItem | None:
    box = item.prov[0].bbox
    for table in tables_on_page:
        if box.intersection_over_self(table.prov[0].bbox) >= _TABLE_OVERLAP:
            return table
    return None


def _table_text(table: TableItem, member_lines: list[str], doc: DoclingDocument) -> str:
    """Markdown when Docling recovered rows/columns, else the table's lines in reading order.

    Without the table-structure model Docling returns each table as one cell holding all its
    text, and also emits every line inside it as a loose text item. The loose lines keep their
    line breaks, so they read better than the single squashed cell.
    """
    if table.data.num_cols > 1 or not member_lines:
        return table.export_to_markdown(doc=doc).strip()
    return "\n".join(member_lines)


def blocks_from_docling(doc: DoclingDocument) -> list[ParsedBlock]:
    """Flatten a Docling document into blocks, tracking the nearest heading as the section.

    Only the body layer is read, so Docling-detected page headers and footers are dropped.
    Text lying inside a table's area is folded into that table's block instead of repeated.
    """
    items = [
        item
        for item, _level in doc.iterate_items(included_content_layers={ContentLayer.BODY})
        if isinstance(item, DocItem) and item.prov and item.label in _LABEL_TO_TYPE
    ]

    tables_by_page: dict[int, list[TableItem]] = defaultdict(list)
    for item in items:
        if isinstance(item, TableItem):
            tables_by_page[item.prov[0].page_no].append(item)

    # Assign each non-table item to the table it sits in, if any.
    table_lines: dict[str, list[str]] = defaultdict(list)
    in_table: set[str] = set()
    for item in items:
        if isinstance(item, TableItem):
            continue
        table = _containing_table(item, tables_by_page.get(item.prov[0].page_no, []))
        if table is not None:
            in_table.add(item.self_ref)
            if text := _item_text(item):
                table_lines[table.self_ref].append(text)

    blocks: list[ParsedBlock] = []
    section: str | None = None
    for item in items:
        if item.self_ref in in_table:
            continue
        if isinstance(item, TableItem):
            text = _table_text(item, table_lines[item.self_ref], doc)
        else:
            text = _item_text(item)
        if not text:
            continue

        block_type = _LABEL_TO_TYPE[item.label]
        if block_type == "heading":
            section = text
        blocks.append(
            ParsedBlock(
                text=text, page=item.prov[0].page_no, section=section, block_type=block_type
            )
        )
    return blocks


@lru_cache
def _converter() -> DocumentConverter:
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def convert_pdf(pdf_path: Path) -> DoclingDocument:
    """Run Docling on one PDF (the slow step)."""
    start = time.perf_counter()
    doc = _converter().convert(pdf_path).document
    log.info(
        "parse.convert path=%s pages=%d latency=%.1fs",
        pdf_path,
        doc.num_pages(),
        time.perf_counter() - start,
    )
    return doc


def to_parsed_document(doc: DoclingDocument, pdf_path: Path, sha256: str) -> ParsedDocument:
    """Wrap extracted blocks with the metadata needed to cite and cache them."""
    company, year = company_year_from_path(pdf_path)
    return ParsedDocument(
        company=company,
        year=year,
        source_path=str(pdf_path),
        sha256=sha256,
        converter_version=CONVERTER_VERSION,
        extract_version=EXTRACT_VERSION,
        num_pages=doc.num_pages(),
        blocks=blocks_from_docling(doc),
    )


def cache_paths_for(pdf_path: Path) -> tuple[Path, Path]:
    """(parsed JSON, raw Docling JSON) paths for a PDF, under data/parsed/{company}/."""
    company, year = company_year_from_path(pdf_path)
    folder = get_settings().parsed_dir / company
    return folder / f"{year}.json", folder / f"{year}.docling.json"


def parse_cached(pdf_path: Path) -> ParsedDocument:
    """Parse a PDF, reusing whichever cache layers are still valid for its bytes and versions."""
    parsed_path, docling_path = cache_paths_for(pdf_path)
    sha256 = file_sha256(pdf_path)

    cached = None
    if parsed_path.exists():
        cached = ParsedDocument.model_validate_json(parsed_path.read_text())
    docling_valid = (
        cached is not None
        and cached.sha256 == sha256
        and cached.converter_version == CONVERTER_VERSION
        and docling_path.exists()
    )

    if docling_valid and cached.extract_version == EXTRACT_VERSION:
        log.info("parse.cache_hit path=%s", pdf_path)
        return cached

    if docling_valid:
        log.info("parse.reextract path=%s", pdf_path)
        doc = DoclingDocument.load_from_json(docling_path)
    else:
        doc = convert_pdf(pdf_path)
        docling_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save_as_json(docling_path)

    parsed = to_parsed_document(doc, pdf_path, sha256)
    parsed_path.write_text(parsed.model_dump_json())
    return parsed
