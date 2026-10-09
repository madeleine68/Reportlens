"""The retrieval entry point used by answer generation; settings pick which techniques run."""

import psycopg

from reportlens.core.config import get_settings
from reportlens.retrieval.filters import extract_filters, known_values
from reportlens.retrieval.models import RetrievedChunk
from reportlens.retrieval.vector import vector_search


def search(conn: psycopg.Connection, question: str) -> list[RetrievedChunk]:
    """Retrieve chunks for a question, with metadata filters when RETRIEVAL_USE_FILTERS is on."""
    filters = None
    if get_settings().retrieval_use_filters:
        filters = extract_filters(question, known_values(conn))
    return vector_search(conn, question, filters=filters)
