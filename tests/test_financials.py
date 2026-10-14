import psycopg

from reportlens.ingest.financials import (
    Cell,
    build_financials,
    load_financials,
    matches_metric,
    scope_of,
)
from reportlens.ingest.metrics import METRICS

METRIC = {m.name: m for m in METRICS}


def _cell(**overrides: object) -> Cell:
    fields = {
        "company": "scotiabank",
        "report_year": 2023,
        "table_id": "scotiabank-2023-t0001",
        "table_title": "T1 Financial highlights",
        "page": 21,
        "row_label": "Net income",
        "row_group": "",
        "column_label": "2023",
        "value": 7528.0,
        "unit": "millions",
        "period_year": 2023,
        "period_kind": "year",
    }
    return Cell(**{**fields, **overrides})


def test_metric_labels_match_the_reports_wordings() -> None:
    assert matches_metric(METRIC["net_income"], "", "Net earnings")
    assert matches_metric(
        METRIC["net_income_to_shareholders"], "", "Net income attributable to equity holders"
    )
    assert matches_metric(
        METRIC["net_income_to_shareholders"], "Net income attributable to:", "Shareholders"
    )
    assert not matches_metric(METRIC["net_income_to_shareholders"], "", "Shareholders")
    assert matches_metric(METRIC["diluted_eps"], "Earnings per share (in dollars)", "- diluted")
    assert matches_metric(METRIC["cet1_ratio"], "", "CommonEquity Tier 1 (CET1) capital ratio (%)")
    assert not matches_metric(METRIC["net_income"], "", "Net income attributable to NCI")


def test_scope_comes_from_the_column_then_the_title() -> None:
    assert scope_of(_cell(column_label="Canadian Banking")) == "Canadian Banking"
    assert scope_of(_cell(column_label="Total")) == "total"
    assert scope_of(_cell(table_title="T21 Canadian Banking financial performance")) == (
        "Canadian Banking"
    )
    assert scope_of(_cell()) == "total"
    assert scope_of(_cell(column_label="U.S.")) is None  # some other breakdown: skipped


def test_most_common_value_wins_and_disagreement_is_counted() -> None:
    cells = [
        _cell(page=21, value=7528.0),
        _cell(page=22, value=7528.0),
        _cell(page=150, value=1234.0, table_title="Note 15 Subsidiary"),
    ]
    [row] = build_financials(cells, [METRIC["net_income"]])
    assert (row.value, row.source_page, row.agreeing_cells, row.candidate_values) == (
        7528.0,
        21,
        2,
        2,
    )


def test_reported_and_adjusted_segments_and_restatements_stay_apart() -> None:
    cells = [
        _cell(row_group="Reported Results", value=7528.0),
        _cell(row_group="Adjusted Results", value=8438.0),
        _cell(column_label="Canadian Banking", value=4019.0),
        _cell(report_year=2024, column_label="2023", value=7450.0),  # restated next year
        _cell(column_label="2021", period_year=2021, value=9955.0),  # too old for this report
    ]
    rows = build_financials(cells, [METRIC["net_income"]])
    assert {(r.year, r.report_year, r.scope, r.basis, r.value) for r in rows} == {
        (2023, 2023, "total", "reported", 7528.0),
        (2023, 2023, "total", "adjusted", 8438.0),
        (2023, 2023, "Canadian Banking", "reported", 4019.0),
        (2023, 2024, "total", "reported", 7450.0),
    }


def test_units_are_converted_or_rejected() -> None:
    cells = [
        _cell(row_label="Total assets", value=1.4, unit="billions"),
        _cell(row_label="Total assets", value=12.3, unit="percent", page=30),
    ]
    [row] = build_financials(cells, [METRIC["total_assets"]])
    assert (row.value, row.unit) == (1400.0, "millions")


def test_counts_take_a_scale_only_from_their_own_row() -> None:
    in_dollar_table = _cell(row_label="Number of employees (FTE)", value=2788.0, unit="millions")
    in_thousands = _cell(row_label="Number of employees (thousands)", value=40.0, unit="thousands")

    [row] = build_financials([in_dollar_table], [METRIC["employees"]])
    assert row.value == 2788.0  # the table's "($ millions)" is not applied to a headcount
    [row] = build_financials([in_thousands], [METRIC["employees"]])
    assert row.value == 40000.0


def test_capex_spending_is_stored_as_a_positive_amount() -> None:
    cells = [_cell(company="bce", row_label="Capital expenditures", value=-3897.0, unit=None)]
    [row] = build_financials(cells, [METRIC["capital_expenditures"]])
    assert row.value == 3897.0


def test_load_financials_replaces_the_table(conn: psycopg.Connection) -> None:
    rows = build_financials([_cell(company="testco")], [METRIC["net_income"]])
    load_financials(conn, rows)
    load_financials(conn, rows)

    assert conn.execute(
        "SELECT company, year, metric, scope, basis, value, source_page FROM financials"
    ).fetchall() == [("testco", 2023, "net_income", "total", "reported", 7528.0, 21)]
