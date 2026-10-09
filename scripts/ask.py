"""Ask a question from the command line and print the answer with its sources.

Usage: uv run python scripts/ask.py "What was RBC's net income in 2023?"
"""

import argparse

from reportlens.core.config import get_settings
from reportlens.core.db import connect
from reportlens.core.logging import setup_logging
from reportlens.generation.answer import answer_question, format_pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question from the annual reports.")
    parser.add_argument("question")
    parser.add_argument("--show-chunks", action="store_true", help="also print retrieved chunks")
    args = parser.parse_args()
    setup_logging(get_settings().log_level)

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


if __name__ == "__main__":
    main()
