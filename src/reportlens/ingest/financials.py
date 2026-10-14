"""Build the `financials` table: a few headline metrics picked from `table_cells` (Phase 3).

Rules only, no LLM. A metric is a list of row-label patterns plus the unit it is stored in. For
every report, each matching full-year cell is a candidate, tagged with:
- scope: "total", or a segment name read from the column header or the table title
- basis: "adjusted" if the row, row group or column says so, else "reported"
The same figure is usually printed in several tables, so the most common value wins and the
number of agreeing cells is kept as a confidence signal.

A report also restates the prior year, sometimes differently (e.g. after an accounting change), so
rows are kept per report: `report_year` says which report a figure comes from.
"""

import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable

import psycopg
from pydantic import BaseModel

from reportlens.core.logging import get_logger
from reportlens.ingest.metrics import METRICS, SEGMENTS, Metric
from reportlens.ingest.tables import scale_unit

log = get_logger(__name__)


_TOTAL_COLUMN = re.compile(r"(total|consolidated|bce|rbc|scotiabank)\b", re.IGNORECASE)
_HAS_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_ADJUSTED = re.compile(r"adjust", re.IGNORECASE)
# Scale of a cell's unit relative to the base unit of a metric.
_SCALES = {
    "millions": {None: 1, "millions": 1, "billions": 1000, "thousands": 0.001},
    "count": {None: 1, "thousands": 1000, "millions": 1_000_000},
    "percent": {None: 1, "percent": 1},
    "dollars": {None: 1, "dollars": 1},
}


class Cell(BaseModel):
    """The table_cells columns used here."""

    company: str
    report_year: int
    table_id: str
    table_title: str
    page: int
    row_label: str
    row_group: str
    column_label: str
    value: float
    unit: str | None
    period_year: int | None
    period_kind: str


class Financial(BaseModel):
    """One row of the financials table."""

    company: str
    year: int  # fiscal year the figure is for
    report_year: int  # report it was read from (year, or year + 1 for a restated figure)
    metric: str
    scope: str  # "total" or a segment name
    basis: str  # "reported" or "adjusted"
    value: float
    unit: str
    source_page: int
    source_table: str  # title of the table the source cell is in
    agreeing_cells: int  # cells that printed this value
    candidate_values: int  # distinct values seen; > 1 means some tables disagreed


def _matches(patterns: Iterable[str], text: str) -> bool:
    return any(re.fullmatch(p, text, re.IGNORECASE) for p in patterns)


def matches_metric(metric: Metric, row_group: str, row_label: str) -> bool:
    """True if a row is printed under one of the metric's labels."""
    if _matches(metric.labels, row_label):
        return True
    return any(
        re.fullmatch(g, row_group, re.IGNORECASE) and re.fullmatch(r, row_label, re.IGNORECASE)
        for g, r in metric.group_labels
    )


def scope_of(cell: Cell) -> str | None:
    """'total', a segment name, or None when the column is some other breakdown (e.g. 'U.S.')."""
    for segment in SEGMENTS.get(cell.company, []):
        if segment.lower() in cell.column_label.lower():
            return segment
    label = cell.column_label
    if label and not (_HAS_YEAR.search(label) or _TOTAL_COLUMN.match(label)):
        return None
    for segment in SEGMENTS.get(cell.company, []):
        if segment.lower() in cell.table_title.lower():
            return segment
    return "total"


def basis_of(cell: Cell) -> str:
    """'adjusted' if the row, its group or its column says so, else 'reported'."""
    labels = (cell.row_label, cell.row_group, cell.column_label)
    return "adjusted" if any(_ADJUSTED.search(label) for label in labels) else "reported"


def to_metric_unit(metric: Metric, cell: Cell) -> float | None:
    """The cell's value in the metric's unit, or None if the units don't fit."""
    unit = cell.unit
    if metric.unit == "count":
        # Counts take a scale only from their own row, e.g. "Number of employees (thousands)":
        # a table's "($ millions)" is about its dollar rows.
        unit = scale_unit(cell.row_label)
    scale = _SCALES[metric.unit].get(unit)
    if scale is None:
        return None
    value = cell.value * scale
    return abs(value) if metric.absolute else value


def build_financials(cells: Iterable[Cell], metrics: list[Metric] = METRICS) -> list[Financial]:
    """One row per (company, year, report, metric, scope, basis): the most common candidate."""
    candidates: dict[tuple, list[tuple[float, Cell]]] = defaultdict(list)
    for cell in cells:
        if cell.period_kind != "year" or cell.period_year is None:
            continue
        if cell.period_year not in (cell.report_year, cell.report_year - 1):
            continue  # older years in multi-year summaries are skipped
        for metric in metrics:
            if not matches_metric(metric, cell.row_group, cell.row_label):
                continue
            scope = scope_of(cell)
            value = to_metric_unit(metric, cell)
            if scope is None or value is None:
                continue
            key = (
                cell.company,
                cell.period_year,
                cell.report_year,
                metric.name,
                scope,
                basis_of(cell),
            )
            candidates[key].append((round(value, 4), cell))

    units = {metric.name: metric.unit for metric in metrics}
    rows = []
    for (company, year, report_year, metric, scope, basis), found in candidates.items():
        counts = Counter(value for value, _ in found)
        # Most cells agreeing wins; ties go to the value printed earliest in the report.
        value = max(counts, key=lambda v: (counts[v], -min(c.page for x, c in found if x == v)))
        source = min((c for x, c in found if x == value), key=lambda c: c.page)
        rows.append(
            Financial(
                company=company,
                year=year,
                report_year=report_year,
                metric=metric,
                scope=scope,
                basis=basis,
                value=value,
                unit=units[metric],
                source_page=source.page,
                source_table=source.table_title,
                agreeing_cells=counts[value],
                candidate_values=len(counts),
            )
        )
    return sorted(rows, key=lambda r: (r.company, r.metric, r.year, r.report_year, r.scope))


def read_cells(conn: psycopg.Connection) -> list[Cell]:
    """Every table cell, as the fields build_financials needs."""
    fields = list(Cell.model_fields)
    rows = conn.execute(f"SELECT {', '.join(fields)} FROM table_cells").fetchall()
    return [Cell(**dict(zip(fields, row, strict=True))) for row in rows]


def load_financials(conn: psycopg.Connection, rows: list[Financial]) -> None:
    """Replace the whole financials table in one transaction."""
    start = time.perf_counter()
    fields = list(Financial.model_fields)
    with conn.transaction():
        conn.execute("DELETE FROM financials")
        with (
            conn.cursor() as cur,
            cur.copy(f"COPY financials ({', '.join(fields)}) FROM STDIN") as copy,
        ):
            for row in rows:
                copy.write_row([getattr(row, name) for name in fields])
    log.info("financials.load rows=%d latency=%.1fs", len(rows), time.perf_counter() - start)
