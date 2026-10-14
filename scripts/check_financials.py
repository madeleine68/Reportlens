"""Check the financials table against the numbers in the eval set's expected answers.

For each numeric, trend or comparison question, every number in the expected answer is looked up
among the financials rows (total scope) for the question's companies and report years. No LLM.
Usage:
    uv run python scripts/check_financials.py
"""

import re
import sys
from pathlib import Path

from reportlens.core.db import connect
from reportlens.evals.questions import load_questions

_NUMBER = re.compile(r"\$?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(billion|million|%|bps)?")
_TYPES = {"numeric", "trend", "comparison"}
QUESTIONS = Path(__file__).resolve().parents[1] / "evals" / "questions.jsonl"


def expected_numbers(answer: str) -> list[tuple[float, float]]:
    """(value, tolerance) per number in an expected answer, in each unit it could be stored in.

    "$5.1 billion" could be 5,100 (millions) give or take 50; "10.29 million" could be a count.
    """
    numbers = []
    for digits, scale in _NUMBER.findall(answer):
        value = float(digits.replace(",", ""))
        if scale == "bps" or 1900 <= value <= 2100 and "," not in digits:
            continue  # changes in bps and bare years aren't stored figures
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        half_step = 0.5 * 10**-decimals
        numbers.append((value, half_step))
        for factor in {"billion": (1000, 1e9), "million": (1e6,)}.get(scale, ()):
            numbers.append((value * factor, half_step * factor))
    return numbers


def close(stored: float, expected: tuple[float, float]) -> bool:
    """Equal to the precision the answer is written in."""
    value, tolerance = expected
    return abs(stored - value) <= tolerance + 1e-9


def main() -> int:
    questions = [q for q in load_questions(QUESTIONS) if q.type in _TYPES]
    found = 0
    with connect() as conn:
        for q in questions:
            pairs = [(s.company, s.year) for s in q.sources]
            rows = conn.execute(
                "SELECT company, year, report_year, metric, basis, value FROM financials"
                " WHERE scope = 'total' AND (company, report_year) IN"
                f" ({', '.join(['(%s, %s)'] * len(pairs))})",
                [x for pair in pairs for x in pair],
            ).fetchall()
            numbers = expected_numbers(q.expected_answer)
            hits = sorted(
                {
                    f"{m}/{b} {c} {y}={v:g}"
                    for c, y, _, m, b, v in rows
                    for n in numbers
                    if close(v, n)
                }
            )
            found += bool(hits)
            status = "ok  " if hits else "MISS"
            print(f"{status} {q.id} {q.question[:60]!r} expected {q.expected_answer[:45]!r}")
            for hit in hits[:3]:
                print(f"       {hit}")
    print(f"\n{found}/{len(questions)} questions have a matching financials row")
    return 0


if __name__ == "__main__":
    sys.exit(main())
