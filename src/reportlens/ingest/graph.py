"""Knowledge-graph facts (Phase 3): each report's segments, subsidiaries and executives.

- Segments: from financials, which already knows each report's segment figures and their page.
- Subsidiaries: the fast LLM reads the report's subsidiaries table (one call per 15 rows). A rule
  can't split them: RBC prints a holding company and the companies held through it in one cell.
- Executives: the fast LLM reads the short passages that pair a name with an officer title, such
  as the signatures under management's report (one call per report).

LLM output is cached by model, prompt and input text, so rebuilding the graph costs nothing.
"""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import psycopg
from pydantic import BaseModel

from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.ingest.chunking import split_table_header
from reportlens.ingest.graph_names import (
    clean_name,
    clean_title,
    is_board_only_title,
    name_key,
    person_key,
)
from reportlens.ingest.models import ParsedBlock, ParsedDocument
from reportlens.ingest.tables import clean_label

log = get_logger(__name__)

Structured = Callable[..., BaseModel]
Role = Literal["ceo", "cfo", "other"]

COMPANY_NAMES = {
    "bce": "BCE Inc.",
    "rbc": "Royal Bank of Canada",
    "scotiabank": "The Bank of Nova Scotia (Scotiabank)",
}

_SUBSIDIARY_TABLE = re.compile(r"principal subsidiaries|^subsidiar(?:y|ies)\b", re.I | re.M)
_OFFICER_TITLE = re.compile(
    r"\bChief\b[\w ,&'-]{0,40}?\bOfficer\b|\bGroup Head\b|\bExecutive Vice[ -]President\b", re.I
)
EXECUTIVE_PASSAGE_MAX_CHARS = 300  # signatures and title lines are short; longer text is prose
MAX_EXECUTIVE_PASSAGES = 60  # keeps one call per report within a few thousand tokens
# Long tables go in batches of rows (header repeated): asked for ~40 names at once, the fast model
# sometimes reasons until it hits the output cap and returns a cut-short list.
SUBSIDIARY_ROWS_PER_CALL = 15
EXTRACT_RETRIES = 2  # long JSON lists are sometimes rejected by the provider's JSON check
# Listing ~40 subsidiaries takes the reasoning model more than Groq's default 2,048 output tokens.
EXTRACT_MAX_TOKENS = 6000  # prompt + this must stay under the 8,000 tokens-per-minute limit


# --- What the LLM returns ----------------------------------------------------------------------


class SubsidiaryFact(BaseModel):
    name: str
    location: str | None = None
    ownership_pct: float | None = None


class SubsidiaryList(BaseModel):
    subsidiaries: list[SubsidiaryFact]


class ExecutiveFact(BaseModel):
    name: str
    title: str
    role: Role
    page: int


class ExecutiveList(BaseModel):
    executives: list[ExecutiveFact]


# --- What goes in the graph: one fact per report, with the page it is printed on ---------------


class Segment(BaseModel):
    company: str
    report_year: int
    name: str
    page: int


class Subsidiary(BaseModel):
    company: str
    report_year: int
    name: str
    key: str  # name_key(name): the same company spelled differently is one node
    location: str | None
    ownership_pct: float | None
    page: int


class Executive(BaseModel):
    company: str
    report_year: int
    name: str
    key: str  # person_key(name): "David I. McKay" and "David McKay" are one node
    title: str
    role: Role
    page: int


# --- Picking the source text ------------------------------------------------------------------


def subsidiary_table(doc: ParsedDocument) -> ParsedBlock | None:
    """The report's (principal) subsidiaries table, if the parse has one."""
    for block in doc.blocks:
        section = block.section or ""
        if block.block_type != "table" or "non-controlling" in section.lower():
            continue
        if _SUBSIDIARY_TABLE.search(section) or _SUBSIDIARY_TABLE.search(block.text[:400]):
            return block
    return None


def table_batches(markdown: str, rows: int = SUBSIDIARY_ROWS_PER_CALL) -> list[str]:
    """The table in pieces of at most `rows` body rows, each starting with the header rows."""
    header, body = split_table_header(markdown)
    lines = [line for line in body.split("\n") if line.strip()]
    prefix = f"{header}\n" if header else ""
    return [prefix + "\n".join(lines[i : i + rows]) for i in range(0, len(lines), rows)] or [
        markdown
    ]


def executive_passages(doc: ParsedDocument) -> list[tuple[int, str]]:
    """(page, 'heading: text') for short passages that name an officer title, duplicates dropped.

    The heading is kept because the report often prints the person's name as the heading and
    only the title in the text below it.
    """
    passages, seen = [], set()
    for block in doc.blocks:
        text = " ".join(clean_label(block.text).split())
        if len(text) > EXECUTIVE_PASSAGE_MAX_CHARS or not _OFFICER_TITLE.search(text):
            continue
        section = " ".join((block.section or "").split())
        passage = f"{section}: {text}" if section and len(section.split()) <= 5 else text
        if passage.casefold() in seen:
            continue
        seen.add(passage.casefold())
        passages.append((block.page, passage))
    return passages[:MAX_EXECUTIVE_PASSAGES]


# --- LLM calls, cached ------------------------------------------------------------------------


class FactCache:
    """LLM extractions saved in a JSON file, keyed by model, prompt and input text."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict] = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def key(messages: list[llm.Message], model: str | None = None) -> str:
        """Hash of the model (the fast one by default) and the exact messages sent."""
        payload = json.dumps([model or get_settings().llm_model_fast, messages])
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        return self._entries.get(key)

    def put(self, key: str, value: dict) -> None:
        self._entries[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._entries, indent=1))
        tmp.replace(self.path)  # atomic, so an interrupted write never corrupts the cache


def extract[T: BaseModel](
    messages: list[llm.Message], schema: type[T], cache: FactCache, structured: Structured
) -> T:
    """structured(messages, schema), reusing a cached answer for the same input."""
    key = cache.key(messages)
    if (cached := cache.get(key)) is not None:
        return schema.model_validate(cached)
    result = structured(
        messages, schema, tier="fast", max_retries=EXTRACT_RETRIES, max_tokens=EXTRACT_MAX_TOKENS
    )
    cache.put(key, result.model_dump())
    return result


def _messages(system: str, doc: ParsedDocument, user: str) -> list[llm.Message]:
    company_name = COMPANY_NAMES.get(doc.company, doc.company)
    return [
        {"role": "system", "content": system.format(company_name=company_name, year=doc.year)},
        {"role": "user", "content": user},
    ]


def subsidiaries_for(
    doc: ParsedDocument, cache: FactCache, structured: Structured = llm.structured
) -> list[Subsidiary]:
    """The subsidiaries listed in the report's subsidiaries table."""
    table = subsidiary_table(doc)
    if table is None:
        log.warning("graph.no_subsidiary_table company=%s year=%d", doc.company, doc.year)
        return []
    found = [
        fact
        for batch in table_batches(table.text)
        for fact in extract(
            _messages(prompts.GRAPH_SUBSIDIARIES_SYSTEM, doc, batch),
            SubsidiaryList,
            cache,
            structured,
        ).subsidiaries
    ]
    subsidiaries, seen = [], set()
    for fact in found:
        name, pct_in_name = clean_name(fact.name)
        key = name_key(name)
        if (
            len(key.split()) < 2
            or key in seen
            or key == name_key(COMPANY_NAMES.get(doc.company, doc.company))
        ):
            continue  # one word is a fragment of a name the PDF table split (e.g. "Scotia")
        seen.add(key)
        subsidiaries.append(
            Subsidiary(
                company=doc.company,
                report_year=doc.year,
                name=name,
                key=key,
                location=fact.location,
                ownership_pct=fact.ownership_pct if fact.ownership_pct is not None else pct_in_name,
                page=table.page,
            )
        )
    return subsidiaries


def executives_for(
    doc: ParsedDocument, cache: FactCache, structured: Structured = llm.structured
) -> list[Executive]:
    """The company's officers named in the report, each with a page that names them."""
    passages = executive_passages(doc)
    if not passages:
        return []
    text = "\n".join(f"[p. {page}] {passage}" for page, passage in passages)
    messages = _messages(prompts.GRAPH_EXECUTIVES_SYSTEM, doc, text)
    found = extract(messages, ExecutiveList, cache, structured).executives
    executives, seen = [], set()
    for fact in found:
        name, _ = clean_name(fact.name)
        key = person_key(name)
        title = clean_title(fact.title)
        if is_board_only_title(title):
            continue  # a director signing the statements, not an officer
        page = page_naming(fact, passages)
        if page is None:
            log.warning("graph.executive_not_in_passages name=%r", fact.name)
            continue
        if not key or key in seen:
            continue
        seen.add(key)
        executives.append(
            Executive(
                company=doc.company,
                report_year=doc.year,
                name=name,
                key=key,
                title=title,
                role=fact.role,
                page=page,
            )
        )
    return executives


def page_naming(fact: ExecutiveFact, passages: list[tuple[int, str]]) -> int | None:
    """The page the LLM gave if that passage names the person, else the first one that does.

    None if no passage names them, so a made-up person never reaches the graph.
    """
    words = fact.name.split()
    surname = words[-1].casefold() if words else ""
    pages = [page for page, text in passages if surname and surname in text.casefold()]
    if not pages:
        return None
    return fact.page if fact.page in pages else pages[0]


def segments_from_financials(conn: psycopg.Connection) -> list[Segment]:
    """Each report's segments: those with figures in financials, at the first page printing one."""
    rows = conn.execute(
        """SELECT company, report_year, scope, min(source_page) FROM financials
           WHERE scope <> 'total' GROUP BY 1, 2, 3 ORDER BY 1, 2, 3"""
    ).fetchall()
    return [Segment(company=c, report_year=y, name=s, page=p) for c, y, s, p in rows]
