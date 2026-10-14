import psycopg
import pytest

from reportlens.ingest.models import ParsedBlock, ParsedDocument
from reportlens.ingest.tables import (
    cell_unit,
    cells_from_document,
    cells_from_table,
    detect_period,
    load_table_cells,
    parse_value,
    scale_unit,
)

TABLE = "\n".join(
    [
        "| As at October 31 ($ millions) | 2023 | 2022 | % change |",
        "| --- | --- | --- | --- |",
        "| Reported Results |  |  |  |",
        "| Net income | $ 7,528 | $ 10,174 | (26) % |",
        "| Diluted earnings per share (in dollars) | $ 5.78 | $ 8.02 | n.m. |",
        "| Adjusted Results |  |  |  |",
        "| Net income | 8,438 | 10,749 | - |",
    ]
)


def _block(text: str = TABLE, section: str | None = "T2  Reconciliation") -> ParsedBlock:
    return ParsedBlock(text=text, page=22, section=section, block_type="table")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$ 4,019", (4019.0, None)),
        ("$ (367)", (-367.0, None)),
        ("(6 bps)", (-6.0, "bps")),
        ("(26) %", (-26.0, "percent")),
        ("12.6 %", (12.6, "percent")),
        ("-1,234.5", (-1234.5, None)),
        ("0.9975", (0.9975, None)),
        ("1,204", (1204.0, None)),
    ],
)
def test_parse_value_reads_financial_number_formats(
    text: str, expected: tuple[float, str | None]
) -> None:
    assert parse_value(text) == expected


@pytest.mark.parametrize("text", ["", "-", "n.m.", "Net income", "(367", "Q4 2023", "2,22,1"])
def test_parse_value_rejects_non_numbers(text: str) -> None:
    assert parse_value(text) is None


def test_scale_unit_finds_millions_billions_thousands() -> None:
    assert scale_unit("(Millions of Canadian dollars)") == "millions"
    assert scale_unit("As at October 31 ($ billions)") == "billions"
    assert scale_unit("for the year ended December 31") is None


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("2023", (2023, "year")),
        ("For the year ended - October 31 2024", (2024, "year")),
        ("For the three monthsended - October 31, 2023 - Pre-tax", (2023, "quarter")),
        ("Q4 2022", (2022, "quarter")),
        ("% change", (None, "change")),
        ("2024 vs 2023", (2024, "change")),
        ("Canadian Banking", (None, "unknown")),
    ],
)
def test_detect_period_reads_year_and_kind(label: str, expected: tuple) -> None:
    assert detect_period(label) == expected


def test_cell_unit_prefers_cell_then_row_then_column_then_table() -> None:
    assert cell_unit("percent", "Net income", "2023", "millions") == "percent"
    assert cell_unit(None, "Diluted earnings per share ($)", "2023", "millions") == "dollars"
    assert cell_unit(None, "Effective tax rate (%)", "2023", "millions") == "percent"
    assert cell_unit(None, "Net income", "2023 ($ billions)", "millions") == "billions"
    assert cell_unit(None, "Net income", "2023", "millions") == "millions"


def test_cells_from_table_labels_every_number() -> None:
    cells = cells_from_table(_block(), "scotiabank", 2023, "scotiabank-2023-t0001")

    # 3 numbers in the first net income row, 2 in EPS (n.m. skipped), 2 adjusted ("-" is nil).
    assert len(cells) == 7
    first = cells[0]
    assert (first.row_label, first.column_label, first.value) == ("Net income", "2023", 7528.0)
    assert (first.unit, first.period_year, first.period_kind) == ("millions", 2023, "year")
    assert (first.page, first.table_title, first.raw_text) == (22, "T2 Reconciliation", "$ 7,528")
    change = cells[2]
    assert (change.value, change.unit, change.period_kind) == (-26.0, "percent", "change")
    eps = cells[3]
    assert (eps.value, eps.unit) == (5.78, "dollars")


def test_label_only_rows_become_the_row_group_of_rows_below() -> None:
    cells = cells_from_table(_block(), "scotiabank", 2023, "t")
    net_income_2023 = [
        (c.row_group, c.value)
        for c in cells
        if c.row_label == "Net income" and c.column_label == "2023"
    ]
    assert net_income_2023 == [("Reported Results", 7528.0), ("Adjusted Results", 8438.0)]


def test_columns_without_a_year_take_the_tables_year() -> None:
    table = "\n".join(
        [
            "| As at October 31, 2023 ($ millions) | Canadian Banking | Total |",
            "| --- | --- | --- |",
            "| Net income | 4,019 | 7,410 |",
        ]
    )
    cells = cells_from_table(_block(table), "scotiabank", 2023, "t")
    assert [(c.column_label, c.period_year, c.period_kind) for c in cells] == [
        ("Canadian Banking", 2023, "year"),
        ("Total", 2023, "year"),
    ]


def test_table_without_a_header_row_gives_no_cells() -> None:
    assert cells_from_table(_block("Revenue 10 9\nEPS 1 2"), "bce", 2024, "t") == []


def test_cells_from_document_numbers_tables_in_reading_order() -> None:
    doc = ParsedDocument(
        company="scotiabank",
        year=2023,
        source_path="x.pdf",
        sha256="abc",
        converter_version="2-tables",
        extract_version="2",
        num_pages=30,
        blocks=[
            _block(),
            ParsedBlock(text="Some text", page=23, section="S", block_type="text"),
            _block(),
        ],
    )
    ids = {c.table_id for c in cells_from_document(doc)}
    assert ids == {"scotiabank-2023-t0000", "scotiabank-2023-t0001"}


def test_load_table_cells_replaces_a_reports_cells(conn: psycopg.Connection) -> None:
    cells = cells_from_table(_block(), "testco", 2023, "testco-2023-t0000")
    load_table_cells(conn, "testco", 2023, cells)
    load_table_cells(conn, "testco", 2023, cells[:2])  # a reload replaces, never duplicates

    rows = conn.execute(
        "SELECT row_group, row_label, value, unit, period_year FROM table_cells"
        " WHERE company = 'testco' ORDER BY row_index, column_index"
    ).fetchall()
    assert rows == [
        ("Reported Results", "Net income", 7528.0, "millions", 2023),
        ("Reported Results", "Net income", 10174.0, "millions", 2022),
    ]
