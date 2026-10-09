"""The retrieval entry point used by answer generation; settings pick which techniques run."""

import psycopg

from reportlens.core.config import get_settings
from reportlens.retrieval.filters import extract_filters, known_values
from reportlens.retrieval.keyword import keyword_search
from reportlens.retrieval.models import RetrievedChunk
from reportlens.retrieval.vector import vector_search


def search(conn: psycopg.Connection, question: str) -> list[RetrievedChunk]:
    """Retrieve chunks for a question.

    RETRIEVAL_MODE picks the retriever (vector or keyword); RETRIEVAL_USE_FILTERS adds
    LLM-extracted company/year filters to either.
    """
    settings = get_settings()
    filters = None
    if settings.retrieval_use_filters:
        filters = extract_filters(question, known_values(conn))
    if settings.retrieval_mode == "keyword":
        return keyword_search(conn, question, filters=filters)
    return vector_search(conn, question, filters=filters)
