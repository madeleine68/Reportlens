"""Describe the reports' charts with the vision model and load them as search chunks (Phase 3).

Needs the light-mode Docling parses (data/parsed/{company}/{year}.docling.json), the PDFs and the
reports loaded in Postgres. Summaries are cached (data/cache/figures.json), so the script can be
stopped and re-run: each run describes what is still missing, then reloads every report's
figure chunks from the cache. It stops cleanly at the provider's daily token limit.
Usage:
    uv run python scripts/describe_figures.py --dry-run    # count candidates, no LLM calls
    uv run python scripts/describe_figures.py --limit 50   # describe at most 50 new pictures
    uv run python scripts/describe_figures.py              # describe everything still missing
Exit code 0 once every candidate is described, 3 if some are left for a later run.
"""

import argparse
import sys
import time

import openai
from docling_core.types.doc import DoclingDocument
from pypdfium2 import PdfDocument

from reportlens.core.config import get_settings
from reportlens.core.db import connect
from reportlens.core.logging import setup_logging
from reportlens.ingest.chunking import embedding_tokenizer
from reportlens.ingest.figures import (
    ChartSummary,
    FigureCandidate,
    cache_key,
    chart_candidates,
    describe,
    figure_chunks,
    pdf_text_in_box,
    render_png,
)
from reportlens.ingest.graph import FactCache
from reportlens.ingest.load import load_figure_chunks

RATE_LIMIT_WAIT_S = 60


class DailyLimitReached(Exception):
    """The provider's daily token limit is used up; wait a day and re-run."""


def describe_with_retry(candidate: FigureCandidate, png: bytes, cache: FactCache) -> ChartSummary:
    """describe(), waiting out per-minute rate limits; raises DailyLimitReached on the daily one."""
    while True:
        try:
            return describe(candidate, png, cache)
        except openai.RateLimitError as err:
            if "per day" in str(err):
                raise DailyLimitReached(str(err)) from err
            print(f"  per-minute rate limit; waiting {RATE_LIMIT_WAIT_S}s", flush=True)
            time.sleep(RATE_LIMIT_WAIT_S)


def main() -> int:
    parser = argparse.ArgumentParser(description="Describe charts and load them as chunks.")
    parser.add_argument("--dry-run", action="store_true", help="count candidates only")
    parser.add_argument("--limit", type=int, help="describe at most this many new pictures")
    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    cache = FactCache(settings.figure_cache_path)
    count_tokens = lambda text: len(embedding_tokenizer()(text))  # noqa: E731
    budget = args.limit
    stopped = None
    left = 0  # candidates not described in this run
    with connect() as conn:
        reports = conn.execute("SELECT company, year FROM documents ORDER BY 1, 2").fetchall()
        for company, year in reports:
            docling_path = settings.parsed_dir / company / f"{year}.docling.json"
            pdf = PdfDocument(settings.raw_dir / company / f"{year}.pdf")
            doc = DoclingDocument.load_from_json(docling_path)
            candidates = chart_candidates(doc, company, year, pdf_text_in_box(pdf))
            if args.dry_run:
                print(f"{company}/{year}: {len(candidates)} candidates")
                continue

            described: list[tuple[FigureCandidate, ChartSummary]] = []
            for candidate in candidates:
                png = render_png(pdf, candidate.page, candidate.box)
                is_cached = cache.get(cache_key(candidate, png)) is not None
                if stopped or (not is_cached and budget is not None and budget <= 0):
                    left += 1
                    continue  # leave it for the next run
                try:
                    described.append((candidate, describe_with_retry(candidate, png, cache)))
                except DailyLimitReached as err:
                    stopped = f"daily limit reached: {err}"
                    left += 1
                    continue
                if not is_cached and budget is not None:
                    budget -= 1

            chunks = figure_chunks(described, count_tokens)
            load_figure_chunks(conn, f"{company}-{year}", chunks)
            print(
                f"{company}/{year}: {len(candidates)} candidates, {len(described)} described, "
                f"{len(chunks)} charts loaded",
                flush=True,
            )
    if stopped:
        print(f"\nStopped early ({stopped[:200]}).")
    if left:
        print(f"{left} candidates left; re-run later to describe them.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
