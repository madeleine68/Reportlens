"""Postgres connections with pgvector support. The schema is applied idempotently on connect."""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from reportlens.core.config import get_settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
# The only tables LLM-written SQL may read (Phase 3). sources/sql_safety.py enforces the same list.
READER_TABLES = ("financials", "table_cells")


def connect() -> psycopg.Connection:
    """Open a connection, make sure the schema exists, and enable vector types.

    Autocommit: each statement commits on its own; group writes with `with conn.transaction():`.
    """
    conn = psycopg.connect(get_settings().database_url, autocommit=True)
    conn.execute(SCHEMA_PATH.read_text())  # also creates the vector extension
    register_vector(conn)  # needs the extension, so it comes after the schema
    ensure_reader_role(conn)
    return conn


def ensure_reader_role(conn: psycopg.Connection) -> None:
    """Create the SQL tool's login role if missing, and let it read the approved tables only.

    The role is read-only by default and has a statement timeout, so even SQL that slipped past
    sources/sql_safety.py could neither write nor run for long.
    """
    s = get_settings()
    role = sql.Identifier(s.sql_reader_user)
    password = sql.Literal(s.sql_reader_password.get_secret_value())
    exists = conn.execute(
        "SELECT 1 FROM pg_roles WHERE rolname = %s", (s.sql_reader_user,)
    ).fetchone()
    if not exists:
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(role, password))
    conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(role))
    conn.execute(
        sql.SQL("ALTER ROLE {} SET statement_timeout = {}").format(
            role, sql.Literal(s.sql_timeout_ms)
        )
    )
    conn.execute(
        sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier(s.postgres_db), role)
    )
    conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role))
    tables = sql.SQL(", ").join(sql.Identifier(t) for t in READER_TABLES)
    conn.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(tables, role))


def connect_reader() -> psycopg.Connection:
    """Read-only connection for LLM-written SQL: the reader role, a read-only session and a
    statement timeout. Not autocommit, so every query runs in a read-only transaction.
    """
    s = get_settings()
    conn = psycopg.connect(
        s.reader_database_url, options=f"-c statement_timeout={s.sql_timeout_ms}"
    )
    conn.read_only = True
    return conn
