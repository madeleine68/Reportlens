import pytest

from reportlens.sources.sql_safety import UnsafeSQL, check_sql


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "SELECT value FROM financials WHERE company = 'rbc'",
            "SELECT value FROM financials WHERE company = 'rbc' LIMIT 50",
        ),
        ("SELECT value FROM financials LIMIT 10", "SELECT value FROM financials LIMIT 10"),
        ("SELECT value FROM financials LIMIT 1000", "SELECT value FROM financials LIMIT 50"),
        ("SELECT value FROM financials LIMIT ALL", "SELECT value FROM financials LIMIT 50"),
        (
            "WITH x AS (SELECT * FROM financials) SELECT * FROM x",
            "WITH x AS (SELECT * FROM financials) SELECT * FROM x LIMIT 50",
        ),
        (
            "SELECT page FROM financials UNION SELECT page FROM table_cells",
            "SELECT page FROM financials UNION SELECT page FROM table_cells LIMIT 50",
        ),
    ],
)
def test_safe_queries_pass_with_a_row_limit(query: str, expected: str) -> None:
    assert check_sql(query, row_limit=50) == expected


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("SELECT * FROM financials; DROP TABLE financials", "exactly one statement"),
        ("", "exactly one statement"),
        ("DELETE FROM financials", "only SELECT"),
        ("UPDATE financials SET value = 0", "only SELECT"),
        ("COPY financials TO '/tmp/x'", "only SELECT"),
        ("SELECT * FROM chunks", "table 'chunks' is not allowed"),
        ("SELECT * FROM (SELECT * FROM documents) d", "table 'documents' is not allowed"),
        ("SELECT * FROM pg_roles", "table 'pg_roles' is not allowed"),
        ("SELECT * FROM pg_catalog.pg_roles", "schema-qualified"),
        ("SELECT * FROM information_schema.tables", "schema-qualified"),
        ("SELECT pg_sleep(10)", "pg_sleep"),
        ("SELECT current_setting('is_superuser')", "current_setting"),
        ("SELECT * INTO copy FROM financials", "SELECT INTO"),
        ("SELECT * FROM financials FOR UPDATE", "locking"),
        ("SELEC value FROM financials", r"could not parse the SQL: .+ \(line 1, column \d+\)$"),
    ],
)
def test_unsafe_queries_are_rejected_with_a_reason(query: str, reason: str) -> None:
    with pytest.raises(UnsafeSQL, match=reason):
        check_sql(query, row_limit=50)
