"""Ingest every PDF under data/raw/{company}/{year}.pdf.

Phase 1 so far: parse and chunk (embed and load are added in later tasks).
Usage: uv run python scripts/ingest_all.py
"""

import sys
import time

from reportlens.core.config import get_settings
from reportlens.core.logging import setup_logging
from reportlens.ingest.chunking import chunk_document
from reportlens.ingest.parse import parse_cached


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)

    pdfs = sorted(settings.raw_dir.glob("*/*.pdf"))
    if not pdfs:
        print(f"No PDFs found. Put files at {settings.raw_dir}/{{company}}/{{year}}.pdf")
        return 1

    failures = 0
    for pdf in pdfs:
        start = time.perf_counter()
        try:
            doc = parse_cached(pdf)
            chunks = chunk_document(doc)
        except Exception as err:  # keep going so one bad PDF doesn't stop the batch
            failures += 1
            print(f"FAILED {pdf.relative_to(settings.raw_dir)}: {err}", flush=True)
            continue
        print(
            f"{doc.company}/{doc.year}: {doc.num_pages} pages, {len(doc.blocks)} blocks, "
            f"{len(chunks)} chunks, "
            f"{time.perf_counter() - start:.1f}s",
            flush=True,  # show progress live; a long batch may be killed partway through
        )

    print(f"Done: {len(pdfs) - failures}/{len(pdfs)} ingested")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
