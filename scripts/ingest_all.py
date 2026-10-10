"""Ingest PDFs under data/raw/{company}/{year}.pdf: parse, chunk, embed and load into Postgres.

Each step is cached, so re-running only does work for new or changed reports.
Usage:
    uv run python scripts/ingest_all.py                          # every company
    uv run python scripts/ingest_all.py --companies rbc,bce      # only these
"""

import argparse
import sys
import time

from reportlens.core.config import get_settings
from reportlens.core.db import connect
from reportlens.core.logging import setup_logging
from reportlens.ingest.chunking import chunk
from reportlens.ingest.load import is_current, load_document
from reportlens.ingest.parse import parse_cached


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse, chunk, embed and load annual reports.")
    parser.add_argument("--companies", help="comma-separated company folders, e.g. rbc,bce")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    pdfs = sorted(settings.raw_dir.glob("*/*.pdf"))
    if args.companies:
        wanted = {c.strip() for c in args.companies.split(",")}
        pdfs = [pdf for pdf in pdfs if pdf.parent.name in wanted]
    if not pdfs:
        print(f"No PDFs found. Put files at {settings.raw_dir}/{{company}}/{{year}}.pdf")
        return 1

    failures = 0
    with connect() as conn:
        for pdf in pdfs:
            start = time.perf_counter()
            name = pdf.relative_to(settings.raw_dir)
            try:
                doc = parse_cached(pdf)
                if is_current(conn, doc):
                    status = "already loaded"
                else:
                    chunks = chunk(doc)  # strategy from the CHUNKING setting
                    load_document(conn, doc, chunks)
                    status = f"loaded {len(chunks)} chunks"
            except Exception as err:  # keep going so one bad PDF doesn't stop the batch
                failures += 1
                print(f"FAILED {name}: {err}", flush=True)
                continue
            # flush: show progress live, since a long batch may be killed partway through
            print(f"{name}: {status}, {time.perf_counter() - start:.1f}s", flush=True)

    print(f"Done: {len(pdfs) - failures}/{len(pdfs)} ingested")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
