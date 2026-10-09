"""Metadata filters: the fast LLM reads which companies and fiscal years a question is about.

Values are checked against what is actually in the database, so a made-up company or year
can never filter out every chunk.
"""

from collections.abc import Callable

import psycopg
from pydantic import BaseModel

from reportlens.core import llm
from reportlens.core.logging import get_logger

log = get_logger(__name__)

Structured = Callable[..., BaseModel]


class QueryFilters(BaseModel):
    """Which reports to search. An empty list means 'no filter on this field'."""

    companies: list[str] = []
    years: list[int] = []

    def is_empty(self) -> bool:
        return not self.companies and not self.years


_SYSTEM = """You extract search filters from a question about company annual reports.
Return JSON with two lists:
- companies: ids of the companies the question is about, chosen only from: {companies}.
  Common names: RBC or Royal Bank of Canada = rbc; Scotiabank, Bank of Nova Scotia or BNS = \
scotiabank; BCE, Bell or Bell Canada = bce; TD or TD Bank = td; Shopify = shopify.
- years: the fiscal years the question asks about, chosen only from: {years}.
  A range such as "between 2022 and 2024" means every year in it.
Use an empty list when the question names no company, or no year."""


def known_values(conn: psycopg.Connection) -> tuple[list[str], list[int]]:
    """Companies and years that have chunks in the database."""
    companies = [r[0] for r in conn.execute("SELECT DISTINCT company FROM chunks ORDER BY 1")]
    years = [r[0] for r in conn.execute("SELECT DISTINCT year FROM chunks ORDER BY 1")]
    return companies, years


def extract_filters(
    question: str,
    known: tuple[list[str], list[int]],
    structured: Structured = llm.structured,
) -> QueryFilters:
    """Ask the fast model for filters, then keep only values that exist in the database."""
    companies_known, years_known = known
    system = _SYSTEM.format(companies=", ".join(companies_known), years=years_known)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    raw = structured(messages, QueryFilters, tier="fast")

    normalized = list(dict.fromkeys(c.strip().lower() for c in raw.companies))  # dedupe, keep order
    companies = [c for c in normalized if c in companies_known]
    years = [y for y in dict.fromkeys(raw.years) if y in years_known]
    dropped = [c for c in normalized if c not in companies] + [
        y for y in raw.years if y not in years
    ]
    if dropped:
        log.info("filters.dropped_unknown values=%s", sorted(map(str, dropped)))
    filters = QueryFilters(companies=companies, years=years)
    log.info("filters.extracted companies=%s years=%s", filters.companies, filters.years)
    return filters
