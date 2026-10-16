"""Build the Neo4j knowledge graph from the parsed reports and financials (Phase 3).

Needs Neo4j running (`docker compose up -d neo4j`), the parsed reports and the financials table
(scripts/load_tables.py). The first run makes two fast-model calls per report; later runs read
data/cache/graph_facts.json and make none.
Usage:
    uv run python scripts/load_graph.py          # build the graph
    uv run python scripts/load_graph.py --show   # then print what each company has, with pages
"""

import argparse
import sys

from reportlens.core.config import get_settings
from reportlens.core.db import connect
from reportlens.core.graph_db import connect_graph, read_query
from reportlens.core.logging import setup_logging
from reportlens.ingest.graph import (
    COMPANY_NAMES,
    FactCache,
    executives_for,
    segments_from_financials,
    subsidiaries_for,
)
from reportlens.ingest.graph_load import load_graph
from reportlens.ingest.models import ParsedDocument

_SUMMARY = {
    "segments": """MATCH (c:Company {id: $company})-[r:HAS_SEGMENT]->(s)
        RETURN s.name AS name, collect(r.report_year + ' p' + r.source_page) AS where""",
    "subsidiaries": """MATCH (c:Company {id: $company})-[r:OWNS]->(s)
        RETURN s.name AS name, collect(r.report_year + ' p' + r.source_page) AS where""",
    "executives": """MATCH (e)-[r:HOLDS_ROLE]->(c:Company {id: $company})
        RETURN e.name + ', ' + r.title AS name,
               collect(r.report_year + ' p' + r.source_page) AS where""",
}


def parsed_report(company: str, year: int) -> tuple[ParsedDocument, ParsedDocument]:
    """(light parse, table parse); the table parse falls back to the light one if missing."""
    folder = get_settings().parsed_dir / company
    light = ParsedDocument.model_validate_json((folder / f"{year}.json").read_text())
    tables_path = folder / f"{year}.tables.json"
    if not tables_path.exists():
        return light, light
    return light, ParsedDocument.model_validate_json(tables_path.read_text())


def show(driver) -> None:
    """Print each company's segments, subsidiaries and executives with report years and pages."""
    for company in COMPANY_NAMES:
        print(f"\n=== {company} ===")
        for label, cypher in _SUMMARY.items():
            rows = read_query(driver, cypher + " ORDER BY name", company=company)
            print(f"{label} ({len(rows)}):")
            for row in rows:
                print(f"  {row['name']}  [{', '.join(sorted(row['where']))}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Neo4j knowledge graph.")
    parser.add_argument("--show", action="store_true", help="print the graph after loading")
    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    cache = FactCache(settings.graph_cache_path)
    subsidiaries, executives = [], []
    with connect() as conn:
        segments = segments_from_financials(conn)
        reports = conn.execute("SELECT DISTINCT company, year FROM chunks ORDER BY 1, 2").fetchall()
    for company, year in reports:
        if company not in COMPANY_NAMES:
            continue
        light, tables = parsed_report(company, year)
        found_subsidiaries = subsidiaries_for(tables, cache)
        found_executives = executives_for(light, cache)
        subsidiaries += found_subsidiaries
        executives += found_executives
        print(
            f"{company}/{year}: {len(found_subsidiaries)} subsidiaries, "
            f"{len(found_executives)} executives",
            flush=True,
        )

    with connect_graph() as driver:
        load_graph(driver, segments, subsidiaries, executives)
        if args.show:
            show(driver)
    return 0


if __name__ == "__main__":
    sys.exit(main())
