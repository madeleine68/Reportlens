"""Metadata filters: the fast LLM reads which companies and fiscal years a question is about.

Values are checked against what is actually in the database, so a made-up company or year
can never filter out every chunk.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import psycopg
from pydantic import BaseModel

from reportlens.core import llm
from reportlens.core.config import get_settings
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


def _system_prompt(known: tuple[list[str], list[int]]) -> str:
    companies, years = known
    return _SYSTEM.format(companies=", ".join(companies), years=years)


def extract_filters(
    question: str,
    known: tuple[list[str], list[int]],
    structured: Structured = llm.structured,
) -> QueryFilters:
    """Ask the fast model for filters, then keep only values that exist in the database."""
    companies_known, years_known = known
    messages = [
        {"role": "system", "content": _system_prompt(known)},
        {"role": "user", "content": question},
    ]
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


class FilterCache:
    """Extracted filters saved in a JSON file, so asking the same question again skips the LLM.

    Eval runs ask the same questions every time; caching saves the daily token quota and gives
    every run identical filters, so differences between runs come from the retriever alone.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def key(question: str, known: tuple[list[str], list[int]]) -> str:
        """Changes whenever the model, the prompt (incl. known companies/years) or question does."""
        payload = json.dumps([get_settings().llm_model_fast, _system_prompt(known), question])
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> QueryFilters | None:
        entry = self._entries.get(key)
        return QueryFilters.model_validate(entry) if entry else None

    def put(self, key: str, question: str, filters: QueryFilters) -> None:
        self._entries[key] = {"question": question, **filters.model_dump()}  # question: readability
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._entries, indent=1))
        tmp.replace(self.path)  # atomic, so an interrupted write never corrupts the cache


def extract_filters_cached(
    question: str,
    known: tuple[list[str], list[int]],
    cache: FilterCache,
    structured: Structured = llm.structured,
) -> QueryFilters:
    """extract_filters, but reuse a cached answer for a question already seen."""
    key = cache.key(question, known)
    if (cached := cache.get(key)) is not None:
        log.info("filters.cache_hit companies=%s years=%s", cached.companies, cached.years)
        return cached
    filters = extract_filters(question, known, structured)
    cache.put(key, question, filters)
    return filters


def filter_conditions(filters: QueryFilters | None) -> tuple[list[str], dict[str, list]]:
    """SQL conditions (to be AND-ed into a WHERE clause) and their parameters for the filters."""
    conditions: list[str] = []
    params: dict[str, list] = {}
    if filters and filters.companies:
        conditions.append("company = ANY(%(companies)s)")
        params["companies"] = filters.companies
    if filters and filters.years:
        conditions.append("year = ANY(%(years)s)")
        params["years"] = filters.years
    return conditions, params


def where_sql(conditions: list[str]) -> str:
    """'WHERE a AND b', or '' when there are no conditions."""
    return f"WHERE {' AND '.join(conditions)}" if conditions else ""
