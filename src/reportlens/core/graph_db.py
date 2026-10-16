"""Neo4j connection for the Phase 3 knowledge graph (companies, segments, subsidiaries, executives).

Writes come only from scripts/load_graph.py. Anything that answers questions reads through
read_query, which runs in a read transaction: the server refuses writes in it.
"""

from typing import Any

from neo4j import Driver, GraphDatabase, RoutingControl

from reportlens.core.config import get_settings


def connect_graph() -> Driver:
    """Open a driver and check the server is reachable."""
    s = get_settings()
    driver = GraphDatabase.driver(
        s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password.get_secret_value())
    )
    driver.verify_connectivity()
    return driver


def read_query(driver: Driver, cypher: str, **params: Any) -> list[dict[str, Any]]:
    """Run a query in a read transaction and return its records as dicts."""
    records, _, _ = driver.execute_query(cypher, params, routing_=RoutingControl.READ)
    return [r.data() for r in records]
