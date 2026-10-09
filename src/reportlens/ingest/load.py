"""Store chunks and their embeddings in Postgres, skipping reports that are already up to date."""

import time
from collections.abc import Callable

import numpy as np
import psycopg

from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.ingest.embed import embed_passages
from reportlens.ingest.models import Chunk, ParsedDocument

log = get_logger(__name__)

Embedder = Callable[[list[str]], np.ndarray]


def document_id(doc: ParsedDocument) -> str:
    """Stable id for a report, e.g. 'rbc-2023'."""
    return f"{doc.company}-{doc.year}"


def ingest_config(doc: ParsedDocument) -> str:
    """Fingerprint of every setting that shapes a report's chunks and vectors."""
    s = get_settings()
    return (
        f"converter={doc.converter_version};extract={doc.extract_version};"
        f"chunk={s.chunk_tokens}/{s.chunk_overlap};model={s.embedding_model}"
    )


def is_current(conn: psycopg.Connection, doc: ParsedDocument) -> bool:
    """True if this exact PDF was already loaded with the current settings."""
    row = conn.execute(
        "SELECT sha256, ingest_config FROM documents WHERE document_id = %s",
        (document_id(doc),),
    ).fetchone()
    return row == (doc.sha256, ingest_config(doc))


def load_document(
    conn: psycopg.Connection,
    doc: ParsedDocument,
    chunks: list[Chunk],
    embed: Embedder = embed_passages,
) -> None:
    """Embed chunks and replace the report's rows in one transaction (all or nothing)."""
    start = time.perf_counter()
    vectors = embed([chunk.text for chunk in chunks]) if chunks else []
    doc_id = document_id(doc)

    with conn.transaction():
        conn.execute("DELETE FROM documents WHERE document_id = %s", (doc_id,))  # cascades
        conn.execute(
            "INSERT INTO documents (document_id, company, year, source_path, sha256, ingest_config)"
            " VALUES (%s, %s, %s, %s, %s, %s)",
            (doc_id, doc.company, doc.year, doc.source_path, doc.sha256, ingest_config(doc)),
        )
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (chunk_id, document_id, company, year, section, pages, text,"
                " num_tokens, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        c.chunk_id,
                        doc_id,
                        c.company,
                        c.year,
                        c.section,
                        c.pages,
                        c.text,
                        c.num_tokens,
                        vector,
                    )
                    for c, vector in zip(chunks, vectors, strict=True)
                ],
            )
    log.info(
        "load.document id=%s chunks=%d latency=%.1fs",
        doc_id,
        len(chunks),
        time.perf_counter() - start,
    )
