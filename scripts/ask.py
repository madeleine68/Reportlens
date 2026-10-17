"""Ask a question from the command line and print the answer with its sources.

Usage:
    uv run python scripts/ask.py "What was RBC's net income in 2023?"
    uv run python scripts/ask.py --sql "What was RBC's net income in 2023?"   # Phase 3 SQL tool
    uv run python scripts/ask.py --graph "Who was Scotiabank's CFO in 2024?"  # Phase 3 graph tool
    uv run python scripts/ask.py --routed "Who replaced Nadine Ahn as RBC's CFO?"  # Phase 4 router
"""

import argparse

from neo4j.exceptions import ServiceUnavailable

from reportlens.agents.routed import ask, default_answerers
from reportlens.agents.router import route_cache
from reportlens.core.config import get_settings
from reportlens.core.db import connect, connect_reader
from reportlens.core.graph_db import connect_graph
from reportlens.core.logging import setup_logging
from reportlens.generation.answer import answer_question, format_pages
from reportlens.retrieval.filters import known_values
from reportlens.sources.graph_tool import answer_with_graph
from reportlens.sources.sql_tool import RowsAnswer, answer_with_sql


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question from the annual reports.")
    parser.add_argument("question")
    parser.add_argument("--show-chunks", action="store_true", help="also print retrieved chunks")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sql", action="store_true", help="answer from the financial tables")
    source.add_argument("--graph", action="store_true", help="answer from the knowledge graph")
    source.add_argument("--routed", action="store_true", help="let the router pick the source")
    args = parser.parse_args()
    setup_logging(get_settings().log_level)
    if args.sql:
        ask_sql(args.question)
        return
    if args.graph:
        ask_graph(args.question)
        return
    if args.routed:
        ask_routed(args.question)
        return

    with connect() as conn:
        answer = answer_question(conn, args.question)

    print(f"\n{answer.text}\n")
    if answer.citations:
        print("Sources:")
        for c in answer.citations:
            print(f"  [{c.number}] {c.company.upper()} {c.year}, {format_pages(c.pages)}")
    print(
        f"\n({answer.model}, {answer.input_tokens} in / {answer.output_tokens} out tokens, "
        f"{answer.latency_s:.1f}s)"
    )
    if args.show_chunks:
        for i, chunk in enumerate(answer.chunks, start=1):
            print(f"\n--- [{i}] {chunk.chunk_id} score={chunk.score:.3f} ---\n{chunk.text}")


def ask_sql(question: str) -> None:
    """Answer with the SQL tool and print the query and rows behind the answer."""
    connect().close()  # makes sure the schema and the read-only role exist
    with connect_reader() as reader:
        answer = answer_with_sql(reader, question)
    print_rows_answer(answer, f"SQL: {answer.sql or '(none)'}")


def ask_graph(question: str) -> None:
    """Answer with the graph tool and print the Cypher and rows behind the answer."""
    with connect_graph() as driver:
        answer = answer_with_graph(driver, question)
    print_rows_answer(answer, f"Cypher:\n{answer.cypher or '(none)'}")


def ask_routed(question: str) -> None:
    """Route the question, answer it, and print the route and the sources tried."""
    with connect() as conn, connect_reader() as reader:
        try:
            driver = connect_graph()
        except ServiceUnavailable:
            driver = None  # the graph route then falls back to text search
        try:
            answers = default_answerers(conn, reader, driver)
            routed = ask(question, known_values(conn), answers, route_cache())
        finally:
            if driver is not None:
                driver.close()

    answer = routed.answer
    print(f"\nRoute: {routed.route.source} ({routed.route.reason})")
    print(f"Tried: {' -> '.join(routed.path)}")
    print(f"\n{answer.text}\n")
    for c in answer.citations:
        print(f"  [{c.number}] {c.company.upper()} {c.year}, {format_pages(c.pages)}")
    print(f"\n({answer.input_tokens} in / {answer.output_tokens} out main-model tokens)")


def print_rows_answer(answer: RowsAnswer, query: str) -> None:
    """Print a SQL or graph answer, its citations, the query and the rows it used."""
    print(f"\n{answer.text}\n")
    for c in answer.citations:
        print(f"  [{c.number}] {c.company.upper()} {c.year} report, page {c.page}")
    print(f"\n{query}")
    if answer.error:
        print(f"Error: {answer.error}")
    for i, row in enumerate(answer.rows, start=1):
        print(f"  [{i}] " + ", ".join(f"{c}={v}" for c, v in zip(answer.columns, row, strict=True)))


if __name__ == "__main__":
    main()
