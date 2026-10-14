"""Ask a question from the command line and print the answer with its sources.

Usage:
    uv run python scripts/ask.py "What was RBC's net income in 2023?"
    uv run python scripts/ask.py --sql "What was RBC's net income in 2023?"   # Phase 3 SQL tool
"""

import argparse

from reportlens.core.config import get_settings
from reportlens.core.db import connect, connect_reader
from reportlens.core.logging import setup_logging
from reportlens.generation.answer import answer_question, format_pages
from reportlens.sources.sql_tool import answer_with_sql


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question from the annual reports.")
    parser.add_argument("question")
    parser.add_argument("--show-chunks", action="store_true", help="also print retrieved chunks")
    parser.add_argument("--sql", action="store_true", help="answer from the financial tables")
    args = parser.parse_args()
    setup_logging(get_settings().log_level)
    if args.sql:
        ask_sql(args.question)
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

    print(f"\n{answer.text}\n")
    for c in answer.citations:
        print(f"  [{c.number}] {c.company.upper()} {c.year} report, page {c.page}")
    print(f"\nSQL: {answer.sql or '(none)'}")
    if answer.error:
        print(f"Error: {answer.error}")
    for i, row in enumerate(answer.rows, start=1):
        print(f"  [{i}] " + ", ".join(f"{c}={v}" for c, v in zip(answer.columns, row, strict=True)))


if __name__ == "__main__":
    main()
