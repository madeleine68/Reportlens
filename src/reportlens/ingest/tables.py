"""Turn parsed tables into numeric cells for SQL (Phase 3), with rules only: no LLM.

Reads the table-mode parses (data/parsed/{company}/{year}.tables.json), where each table is a
Markdown grid whose first line is the header row. Every cell that holds a number becomes one
TableCell, labelled with its row (first column) and its column header. Unit and period come from
the cell, the labels, the table's corner cell or its title, in that order.
"""

import re
import time

import psycopg

from reportlens.core.logging import get_logger
from reportlens.ingest.chunking import split_table_header
from reportlens.ingest.models import ParsedBlock, ParsedDocument, PeriodKind, TableCell

log = get_logger(__name__)

# "$ (367)", "(6 bps)", "(26) %", "12.6 %", "-1,234.5", "$ 4,019". A lone "-" is nil, not 0.
_NUMBER = re.compile(
    r"\$?\s*(?P<open>\()?\s*\$?\s*(?P<minus>[-−–])?\s*"
    r"(?P<digits>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"\s*(?P<unit>%|bps)?\s*(?P<close>\))?\s*(?P<unit_after>%|bps)?"
)
_SCALE = re.compile(r"\b(millions|billions|thousands)\b", re.IGNORECASE)
_PER_SHARE = re.compile(r"per (common )?share|\bEPS\b|\bdividends? declared per\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_QUARTER = re.compile(r"three months|quarter|\bQ[1-4]\b", re.IGNORECASE)
_CHANGE = re.compile(r"change|\bvs\.?\b|increase|decrease", re.IGNORECASE)
# Footnote markers: LaTeX-style superscripts from the parser, and trailing "(2)", "(6), (7)",
# "(Note 22)" or "*". Labels such as "(%)", "(in dollars)" or "(FTE)" are kept.
_SUPERSCRIPT = re.compile(r"(?:\$?\^\{[^}]*\})+\$?")
_TRAILING_NOTES = re.compile(r"(?:\s*,?\s*(?:\((?:\d{1,2}|note \d+)\)|\*+))+\s*$", re.IGNORECASE)


def clean_label(label: str) -> str:
    """A row or column label without footnote markers, on one line of single spaces."""
    label = _TRAILING_NOTES.sub("", _SUPERSCRIPT.sub("", label))
    return " ".join(label.split())


def split_row(line: str) -> list[str]:
    """Cells of one Markdown table row, stripped."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_value(text: str) -> tuple[float, str | None] | None:
    """(value, unit printed in the cell) if the cell is a number, else None.

    Parentheses mean negative, as in financial statements. The unit is "percent", "bps" or None.
    """
    match = _NUMBER.fullmatch(text.strip())
    if not match or bool(match["open"]) != bool(match["close"]):
        return None
    value = float(match["digits"].replace(",", ""))
    if match["open"] or match["minus"]:
        value = -value
    if match["unit"] and match["unit_after"]:
        return None
    unit = {"%": "percent", "bps": "bps"}.get(match["unit"] or match["unit_after"] or "")
    return value, unit


def scale_unit(text: str) -> str | None:
    """'millions', 'billions' or 'thousands' if the text names a scale, e.g. '($ millions)'."""
    match = _SCALE.search(text)
    return match[1].lower() if match else None


def detect_period(label: str) -> tuple[int | None, PeriodKind]:
    """(year, kind) a column header describes, e.g. 'For the year ended - 2023' -> (2023, year).

    The first year named wins. '% change' columns are 'change', three-month columns 'quarter'.
    """
    years = _YEAR.findall(label)
    year = int(years[0]) if years else None
    if _CHANGE.search(label):
        return year, "change"
    if _QUARTER.search(label):
        return year, "quarter"
    return year, "year" if year else "unknown"


def cell_unit(
    cell: str | None, row_label: str, column_label: str, table_scale: str | None
) -> str | None:
    """A cell's unit: its own %/bps, else per-share dollars, else the row, column or table scale."""
    if cell:
        return cell
    if _PER_SHARE.search(row_label):
        return "dollars"
    if "(%)" in row_label or row_label.rstrip().endswith("%"):
        return "percent"
    return scale_unit(row_label) or scale_unit(column_label) or table_scale


def _is_header_text(cell: str) -> bool:
    """True for a header-like cell: words or a bare year, not a figure."""
    label = clean_label(cell)
    return parse_value(label) is None or bool(_YEAR.fullmatch(label))


def merge_header_rows(columns: list[str], rows: list[list[str]]) -> tuple[list[str], int]:
    """(column labels, header rows used): leading body rows that hold only words or years are
    more header (Docling sometimes takes a title row as the header), so they extend the labels.
    """
    used = 0
    for row in rows:
        values = [cell for cell in row[1:] if cell]
        if not values or not all(_is_header_text(cell) for cell in values):
            break
        width = max(len(columns), len(row))
        columns = [
            " - ".join(
                part
                for part in (
                    columns[i] if i < len(columns) else "",
                    clean_label(row[i]) if i < len(row) else "",
                )
                if part
            )
            for i in range(width)
        ]
        used += 1
    return columns, used


def cells_from_table(
    block: ParsedBlock, company: str, report_year: int, table_id: str
) -> list[TableCell]:
    """Every numeric cell of one Markdown table, labelled by its row, row group and column."""
    header, body = split_table_header(block.text)
    if not header:
        return []  # no column labels to attach (a table the table model didn't structure)
    rows = [split_row(line) for line in body.split("\n")]
    columns = [clean_label(c) for c in split_row(header.split("\n")[0])]
    columns, header_rows = merge_header_rows(columns, rows)
    title = " ".join((block.section or "").split())
    table_scale = scale_unit(columns[0]) or scale_unit(title)
    table_year, _ = detect_period(f"{columns[0]} {title}")

    cells = []
    row_group = ""
    for row_index, row in enumerate(rows):
        if row_index < header_rows:
            continue
        row_label = clean_label(row[0])
        if row_label and not any(row[1:]):
            row_group = row_label  # a label-only row heads the rows below it
            continue
        for column_index, raw in enumerate(row[1:], start=1):
            parsed = parse_value(raw)
            if parsed is None:
                continue
            value, unit_in_cell = parsed
            column_label = columns[column_index] if column_index < len(columns) else ""
            year, kind = detect_period(column_label)
            if year is None and kind != "change":
                year, kind = table_year, ("year" if table_year else kind)
            cells.append(
                TableCell(
                    company=company,
                    report_year=report_year,
                    table_id=table_id,
                    table_title=title,
                    page=block.page,
                    row_index=row_index,
                    row_label=row_label,
                    row_group=row_group,
                    column_index=column_index,
                    column_label=column_label,
                    raw_text=raw,
                    value=value,
                    unit=cell_unit(unit_in_cell, row_label, column_label, table_scale),
                    period_year=year,
                    period_kind=kind,
                )
            )
    return cells


def cells_from_document(doc: ParsedDocument) -> list[TableCell]:
    """Numeric cells of every table in a report; tables are numbered in reading order."""
    tables = [b for b in doc.blocks if b.block_type == "table"]
    return [
        cell
        for i, block in enumerate(tables)
        for cell in cells_from_table(
            block, doc.company, doc.year, f"{doc.company}-{doc.year}-t{i:04d}"
        )
    ]


_COLUMNS = list(TableCell.model_fields)


def load_table_cells(
    conn: psycopg.Connection, company: str, year: int, cells: list[TableCell]
) -> None:
    """Replace one report's cells in a single transaction (all or nothing)."""
    start = time.perf_counter()
    with conn.transaction():
        conn.execute(
            "DELETE FROM table_cells WHERE company = %s AND report_year = %s", (company, year)
        )
        with (
            conn.cursor() as cur,
            cur.copy(f"COPY table_cells ({', '.join(_COLUMNS)}) FROM STDIN") as copy,
        ):
            for cell in cells:
                copy.write_row([getattr(cell, name) for name in _COLUMNS])
    log.info(
        "tables.load company=%s year=%d cells=%d latency=%.1fs",
        company,
        year,
        len(cells),
        time.perf_counter() - start,
    )
