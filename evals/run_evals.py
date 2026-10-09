"""Run the eval set and record the results.

Usage:
    uv run python evals/run_evals.py --name "baseline"
    uv run python evals/run_evals.py --name smoke --questions evals/questions.smoke.jsonl

Per-question details go to evals/runs/{date}-{name}.jsonl. A summary row is appended to
evals/results.md only when running the real eval set (evals/questions.jsonl).
"""

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

from reportlens.core.config import get_settings
from reportlens.core.db import connect
from reportlens.core.logging import setup_logging
from reportlens.evals.questions import load_questions
from reportlens.evals.runner import QuestionResult, evaluate_question, results_row, summarize

EVALS_DIR = Path(__file__).resolve().parent
REAL_QUESTIONS = EVALS_DIR / "questions.jsonl"
RESULTS_MD = EVALS_DIR / "results.md"
RUNS_DIR = EVALS_DIR / "runs"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ReportLens eval set.")
    parser.add_argument("--name", required=True, help="experiment name, e.g. 'baseline'")
    parser.add_argument("--questions", type=Path, default=REAL_QUESTIONS)
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--notes", default="", help="extra text for the results.md Notes column")
    args = parser.parse_args()
    setup_logging(get_settings().log_level)

    if not args.questions.exists():
        print(f"No questions file at {args.questions}. See CLAUDE.md 'Eval set format'.")
        return 1
    questions = load_questions(args.questions)[: args.limit]

    results: list[QuestionResult] = []
    with connect() as conn:
        for i, q in enumerate(questions, start=1):
            try:
                r = evaluate_question(conn, q)
            except Exception as err:  # one failure (e.g. rate limit) shouldn't sink the run
                r = QuestionResult(
                    id=q.id,
                    type=q.type,
                    question=q.question,
                    expected_answer=q.expected_answer,
                    error=f"{type(err).__name__}: {err}",
                )
            results.append(r)
            status = r.error or (
                f"correct={r.correctness} recall={r.recall} faithful={r.faithfulness}"
            )
            print(f"[{i}/{len(questions)}] {q.id}: {status}", flush=True)

    today = dt.date.today().isoformat()
    RUNS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", args.name.lower()).strip("-")
    run_path = RUNS_DIR / f"{today}-{slug}.jsonl"
    run_path.write_text("".join(r.model_dump_json() + "\n" for r in results))

    summary = summarize(results)
    row = results_row(today, args.name, summary, args.notes)
    print(f"\n{summary.model_dump_json(indent=2)}\n\n{row}\n\nDetails: {run_path}")

    if args.questions.resolve() == REAL_QUESTIONS:
        with RESULTS_MD.open("a") as f:
            f.write(row + "\n")
        print(f"Appended to {RESULTS_MD}")
    else:
        print("Not the real eval set, so results.md was not changed.")
    return 1 if summary.errors else 0


if __name__ == "__main__":
    sys.exit(main())
