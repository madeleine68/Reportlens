"""Check LLM-written SQL before it runs: one read-only query, approved tables only, a row limit.

This is the first of three layers. The query also runs as a role that can only SELECT the
approved tables, inside a read-only transaction with a statement timeout (core/db.py).
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from reportlens.core.db import READER_TABLES

# Functions that reach outside the query: server settings, files, sleeps, other databases.
_BLOCKED_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "set_config", "current_setting", "query_to")
_WRITES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
)


class UnsafeSQL(ValueError):
    """The SQL was rejected before running; the message says why (it is fed back to the LLM)."""


def _function_name(node: exp.Func) -> str:
    return (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def check_sql(query: str, row_limit: int) -> str:
    """The query, re-rendered with a LIMIT of at most `row_limit`; raises UnsafeSQL if rejected."""
    try:
        statements = [s for s in sqlglot.parse(query, read="postgres") if s is not None]
    except ParseError as err:
        first = err.errors[0] if err.errors else {}
        where = f" (line {first.get('line')}, column {first.get('col')})" if first else ""
        detail = first.get("description", "invalid SQL")
        raise UnsafeSQL(f"could not parse the SQL: {detail}{where}") from err
    if len(statements) != 1:
        raise UnsafeSQL(f"expected exactly one statement, got {len(statements)}")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise UnsafeSQL(f"only SELECT queries are allowed, got {tree.key.upper()}")

    for node in tree.walk():
        if isinstance(node, _WRITES):
            raise UnsafeSQL(f"{node.key.upper()} is not allowed")
        if isinstance(node, exp.Into):
            raise UnsafeSQL("SELECT INTO is not allowed")
        if isinstance(node, exp.Lock):
            raise UnsafeSQL("locking clauses (FOR UPDATE / FOR SHARE) are not allowed")
        if isinstance(node, exp.Func) and _function_name(node).startswith(
            _BLOCKED_FUNCTION_PREFIXES
        ):
            raise UnsafeSQL(f"function {_function_name(node)}() is not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if table.db or table.catalog:
            raise UnsafeSQL(f"schema-qualified table {table.sql()} is not allowed")
        if name not in READER_TABLES and name not in cte_names:
            allowed = ", ".join(READER_TABLES)
            raise UnsafeSQL(f"table {name!r} is not allowed; use only: {allowed}")

    return _with_limit(tree, row_limit).sql(dialect="postgres")


def _with_limit(tree: exp.Query, row_limit: int) -> exp.Query:
    """Keep the query's own LIMIT if it is a number no larger than row_limit, else set row_limit."""
    limit = tree.args.get("limit")
    current = limit.expression if isinstance(limit, exp.Limit) else None
    if isinstance(current, exp.Literal) and current.is_int and int(current.this) <= row_limit:
        return tree
    return tree.limit(row_limit, copy=True)
