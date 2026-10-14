"""The headline metrics the financials table holds, and each company's business segments.

Edit this file to add a metric: give the row labels it is printed under (regexes, matched in full
against the cleaned row label, ignoring case) and the unit it is stored in. Then re-run
scripts/load_tables.py and scripts/check_financials.py.
"""

from pydantic import BaseModel


class Metric(BaseModel):
    """A headline figure and the row labels it is printed under."""

    name: str
    description: str  # shown to the SQL tool's LLM
    unit: str  # millions | percent | dollars | count
    labels: list[str]  # regexes, full match on the cleaned row label (case-insensitive)
    group_labels: list[
        tuple[str, str]
    ] = []  # (row group, row label) regex pairs, e.g. RBC's "Shareholders"
    absolute: bool = False  # cash-flow tables print spending as negative


METRICS = [
    Metric(
        name="net_income",
        description="Net income (BCE: net earnings), including non-controlling interests",
        unit="millions",
        labels=[r"net (income|earnings)"],
    ),
    Metric(
        name="net_income_to_shareholders",
        description="Net income attributable to shareholders / equity holders (excludes NCI)",
        unit="millions",
        labels=[
            r"(reported )?net (income|earnings) attributable to (equity holders|shareholders)"
            r"( of the bank)?"
        ],
        group_labels=[
            (
                r"net (income|earnings) attributable to:?",
                r"(shareholders|equity holders"
                r"( of the bank)?)",
            )
        ],
    ),
    Metric(
        name="revenue",
        description="Total revenue (BCE: total operating revenues)",
        unit="millions",
        labels=[r"total (operating )?revenues?", r"total bce operating revenues", r"revenues?"],
    ),
    Metric(
        name="diluted_eps",
        description="Diluted earnings per common share, in dollars",
        unit="dollars",
        labels=[
            r"diluted earnings per (common )?share( \((\$|in dollars)\))?",
            r"net earnings per common share - basic and diluted( \(dollars\))?",
        ],
        group_labels=[(r".*(earnings per share|profitability).*", r"- ?diluted")],
    ),
    Metric(
        name="cet1_ratio",
        description="Common Equity Tier 1 (CET1) capital ratio, in percent",
        unit="percent",
        labels=[r"(cet1|common ?equity tier 1( \(cet1\))?)( capital)? ratio( \(%\))?"],
    ),
    Metric(
        name="roe",
        description="Return on (common) equity, in percent",
        unit="percent",
        labels=[r"(return on (common )?equity|roe)( ?\(%\))?"],
    ),
    Metric(
        name="total_assets",
        description="Total assets at fiscal year end",
        unit="millions",
        labels=[r"total assets"],
    ),
    Metric(
        name="capital_expenditures",
        description="Capital expenditures (capex)",
        unit="millions",
        labels=[r"capital expenditures"],
        absolute=True,
    ),
    Metric(
        name="adjusted_ebitda",
        description="Adjusted EBITDA (a non-GAAP measure, so its basis is always 'adjusted')",
        unit="millions",
        labels=[r"(total bce )?adjusted ebitda"],
    ),
    Metric(
        name="employees",
        description="Number of employees at year end (banks: full-time equivalent)",
        unit="count",
        labels=[r"(number of )?employees( \((fte|full-time equivalent|thousands)\))*"],
    ),
    Metric(
        name="mobile_phone_subscribers",
        description="Mobile phone subscribers at year end (BCE)",
        unit="count",
        labels=[r"(wireless )?mobile phone subscribers"],
    ),
    Metric(
        name="dividends_per_share",
        description="Dividends declared per common share, in dollars",
        unit="dollars",
        labels=[r"dividends (declared )?per common shares?( \((dollars|in dollars)\))?"],
    ),
]

# Business segments as the reports name them. A cell under one of these is that segment's figure.
SEGMENTS: dict[str, list[str]] = {
    "bce": ["Bell CTS", "Bell Media", "Bell Wireless", "Bell Wireline"],
    "rbc": [
        "Personal & Commercial Banking",
        "Personal Banking",
        "Commercial Banking",
        "Wealth Management",
        "Insurance",
        "Capital Markets",
        "Investor & Treasury Services",
        "Corporate Support",
    ],
    "scotiabank": [
        "Canadian Banking",
        "International Banking",
        "Global Wealth Management",
        "Global Banking and Markets",
    ],
}
