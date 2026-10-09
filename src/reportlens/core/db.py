"""Postgres connections with pgvector support. The schema is applied idempotently on connect."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from reportlens.core.config import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect() -> psycopg.Connection:
    """Open a connection, make sure the schema exists, and enable vector types.

    Autocommit: each statement commits on its own; group writes with `with conn.transaction():`.
    """
    conn = psycopg.connect(get_settings().database_url, autocommit=True)
    conn.execute(SCHEMA_PATH.read_text())  # also creates the vector extension
    register_vector(conn)  # needs the extension, so it comes after the schema
    return conn
