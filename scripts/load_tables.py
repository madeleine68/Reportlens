"""Load every numeric table cell into table_cells, then rebuild financials from them (Phase 3).

Needs the table-mode parses (data/parsed/{company}/{year}.tables.json); make them with
`PARSE_TABLES=true uv run python scripts/ingest_all.py`. Rules only, no LLM: takes seconds.
Usage:
    uv run python scripts/load_tables.py                       # every parsed report
    uv run python scripts/load_tables.py --companies rbc,bce   # only these
"""

import argparse
import sys

from reportlens.core.config import get_settings
from reportlens.core.db import connect
from reportlens.core.logging import setup_logging
from reportlens.ingest.financials import build_financials, load_financials, read_cells
from reportlens.ingest.models import ParsedDocument
from reportlens.ingest.tables import cells_from_document, load_table_cells


def main() -> int:
    parser = argparse.ArgumentParser(description="Load numeric table cells into Postgres.")
    parser.add_argument("--companies", help="comma-separated company folders, e.g. rbc,bce")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    paths = sorted(settings.parsed_dir.glob("*/*.tables.json"))
    if args.companies:
        wanted = {c.strip() for c in args.companies.split(",")}
        paths = [p for p in paths if p.parent.name in wanted]
    if not paths:
        print(
            f"No table parses found under {settings.parsed_dir}. Run ingest with PARSE_TABLES=true."
        )
        return 1

    with connect() as conn:
        for path in paths:
            doc = ParsedDocument.model_validate_json(path.read_text())
            cells = cells_from_document(doc)
            load_table_cells(conn, doc.company, doc.year, cells)
            print(f"{doc.company}/{doc.year}: {len(cells)} cells", flush=True)
        financials = build_financials(read_cells(conn))
        load_financials(conn, financials)
        print(f"financials: {len(financials)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
