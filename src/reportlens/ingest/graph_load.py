"""Write the graph facts to Neo4j (Phase 3). The graph is rebuilt from scratch on every load.

    (:Company {id, name})
    (:Company)-[:HAS_SEGMENT {report_year, source_page}]->(:Segment {key, name, company})
    (:Company)-[:OWNS {report_year, source_page, ownership_pct, location}]->(:Subsidiary)
    (:Executive {key, name})-[:HOLDS_ROLE {report_year, source_page, title, role}]->(:Company)

Facts come per report, so a relationship appears once per report that prints it: filter on
report_year for "in 2023", or look across years for changes (e.g. a new CFO).
"""

from collections import Counter

from neo4j import Driver

from reportlens.core.logging import get_logger
from reportlens.ingest.graph import COMPANY_NAMES, Executive, Segment, Subsidiary

log = get_logger(__name__)

_CONSTRAINTS = [
    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT segment_key IF NOT EXISTS FOR (s:Segment) REQUIRE s.key IS UNIQUE",
    "CREATE CONSTRAINT subsidiary_key IF NOT EXISTS FOR (s:Subsidiary) REQUIRE s.key IS UNIQUE",
    "CREATE CONSTRAINT executive_key IF NOT EXISTS FOR (e:Executive) REQUIRE e.key IS UNIQUE",
]

_LOAD_COMPANIES = """
UNWIND $rows AS row
MERGE (c:Company {id: row.id}) SET c.name = row.name"""

_LOAD_SEGMENTS = """
UNWIND $rows AS row
MATCH (c:Company {id: row.company})
MERGE (s:Segment {key: row.key}) SET s.name = row.name, s.company = row.company
MERGE (c)-[r:HAS_SEGMENT {report_year: row.report_year}]->(s) SET r.source_page = row.page"""

_LOAD_SUBSIDIARIES = """
UNWIND $rows AS row
MATCH (c:Company {id: row.company})
MERGE (s:Subsidiary {key: row.key}) SET s.name = row.display_name
MERGE (c)-[r:OWNS {report_year: row.report_year}]->(s)
SET r.source_page = row.page, r.ownership_pct = row.ownership_pct, r.location = row.location"""

_LOAD_EXECUTIVES = """
UNWIND $rows AS row
MATCH (c:Company {id: row.company})
MERGE (e:Executive {key: row.key}) SET e.name = row.display_name
MERGE (e)-[r:HOLDS_ROLE {report_year: row.report_year, title: row.title}]->(c)
SET r.source_page = row.page, r.role = row.role"""


def display_names(facts: list[Subsidiary] | list[Executive]) -> dict[str, str]:
    """For each key, the spelling the reports use most (ties: the longest, e.g. with initials)."""
    spellings: dict[str, Counter[str]] = {}
    for fact in facts:
        spellings.setdefault(fact.key, Counter())[fact.name] += 1
    return {
        key: max(counts, key=lambda name: (counts[name], len(name)))
        for key, counts in spellings.items()
    }


def _rows(facts: list[Subsidiary] | list[Executive]) -> list[dict]:
    names = display_names(facts)
    return [{**f.model_dump(), "display_name": names[f.key]} for f in facts]


def load_graph(
    driver: Driver,
    segments: list[Segment],
    subsidiaries: list[Subsidiary],
    executives: list[Executive],
) -> None:
    """Replace the whole graph with these facts."""
    driver.execute_query("MATCH (n) DETACH DELETE n")
    for constraint in _CONSTRAINTS:
        driver.execute_query(constraint)
    companies = sorted({f.company for f in [*segments, *subsidiaries, *executives]})
    driver.execute_query(
        _LOAD_COMPANIES,
        rows=[{"id": c, "name": COMPANY_NAMES.get(c, c)} for c in companies],
    )
    segment_rows = [{**s.model_dump(), "key": f"{s.company}/{s.name}"} for s in segments]
    driver.execute_query(_LOAD_SEGMENTS, rows=segment_rows)
    driver.execute_query(_LOAD_SUBSIDIARIES, rows=_rows(subsidiaries))
    driver.execute_query(_LOAD_EXECUTIVES, rows=_rows(executives))
    log.info(
        "graph.loaded companies=%d segments=%d subsidiaries=%d executives=%d",
        len(companies),
        len(segments),
        len(subsidiaries),
        len(executives),
    )
