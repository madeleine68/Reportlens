"""Copy the local data to the hosted databases (Phase 5): Postgres → Neon, the graph → AuraDB.

Nothing is re-parsed or re-embedded, and no LLM is called:
1. Postgres: pg_dump of the local database (run inside the Postgres container, so the client
   matches the server) piped into psql against DATABASE_URL. --clean makes it safe to re-run.
2. The SQL tool's read-only role is created on the target by the usual connect() (with
   SQL_READER_PASSWORD from .env.deploy).
3. Neo4j: scripts/load_graph.py with NEO4J_URI/USER/PASSWORD pointed at AuraDB; it rebuilds the
   graph from the local Postgres and the cached extractions.
Targets come from .env.deploy (git-ignored); connection strings are never printed.
Usage:
    uv run python scripts/deploy_data.py            # both
    uv run python scripts/deploy_data.py --only postgres|graph
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def target_env(values: dict[str, str], names: list[str]) -> dict[str, str]:
    """The current environment plus these deploy values (for a child process)."""
    if missing := [n for n in names if not values.get(n)]:
        raise SystemExit(f"Set these in .env.deploy first: {', '.join(missing)}")
    return {**os.environ, **{n: values[n] for n in names}}


def copy_postgres(values: dict[str, str]) -> None:
    """Dump the local database into the hosted one, then create the read-only role there."""
    env = target_env(values, ["DATABASE_URL", "SQL_READER_PASSWORD"])
    local = dotenv_values(ROOT / ".env")
    user, db = local.get("POSTGRES_USER", "reportlens"), local.get("POSTGRES_DB", "reportlens")
    dump = subprocess.Popen(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "pg_dump",
            "-U",
            user,
            "-d",
            db,
            "--clean",
            "--if-exists",
            "--no-owner",
            "--no-privileges",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
    )
    # The URL reaches psql through the container's environment, not its command line.
    restore = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "-e",
            "TARGET_URL",
            "postgres",
            "sh",
            "-c",
            'psql "$TARGET_URL" -q -v ON_ERROR_STOP=1 > /dev/null',
        ],
        cwd=ROOT,
        stdin=dump.stdout,
        env={**env, "TARGET_URL": env["DATABASE_URL"]},
    )
    dump.stdout.close()  # type: ignore[union-attr]
    if dump.wait() or restore.returncode:
        raise SystemExit("Postgres copy failed (see the messages above).")
    check = (
        "from reportlens.core.db import connect\n"
        "with connect() as c:\n"
        "    n = c.execute('SELECT count(*) FROM chunks').fetchone()[0]\n"
        "    f = c.execute('SELECT count(*) FROM financials').fetchone()[0]\n"
        "print(f'Postgres: {n} chunks, {f} financials rows; read-only role ready')"
    )
    subprocess.run(["uv", "run", "python", "-c", check], cwd=ROOT, env=env, check=True)


def copy_graph(values: dict[str, str]) -> None:
    """Rebuild the graph on the hosted Neo4j from the local data (no LLM calls: cached)."""
    names = ["NEO4J_URI", "NEO4J_PASSWORD"] + (["NEO4J_USER"] if values.get("NEO4J_USER") else [])
    env = target_env(values, names)
    subprocess.run(["uv", "run", "python", "scripts/load_graph.py"], cwd=ROOT, env=env, check=True)
    check = (
        "from reportlens.core.graph_db import connect_graph, read_query\n"
        "with connect_graph() as d:\n"
        "    n = read_query(d, 'MATCH (n) RETURN count(n) AS n')[0]['n']\n"
        "    r = read_query(d, 'MATCH ()-[r]->() RETURN count(r) AS r')[0]['r']\n"
        "print(f'Neo4j: {n} nodes, {r} relationships')"
    )
    subprocess.run(["uv", "run", "python", "-c", check], cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy local data to the hosted databases.")
    parser.add_argument("--only", choices=["postgres", "graph"])
    args = parser.parse_args()
    values = {k: v for k, v in dotenv_values(ROOT / ".env.deploy").items() if v}
    if args.only != "graph":
        copy_postgres(values)
    if args.only != "postgres":
        copy_graph(values)
    return 0


if __name__ == "__main__":
    sys.exit(main())
