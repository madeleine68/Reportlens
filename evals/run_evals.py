"""Run the eval set and record the results.

Usage:
    uv run python evals/run_evals.py --name "baseline"
    uv run python evals/run_evals.py --name smoke --questions evals/questions.smoke.jsonl
    uv run python evals/run_evals.py --name "sql tool" --source sql --types numeric,trend,comparison
    uv run python evals/run_evals.py --name "graph tool" --source graph --types relationship
    uv run python evals/run_evals.py --name "router" --source routed   # Phase 4 router
    uv run python evals/run_evals.py --name "corrective" --source corrective   # Phase 4 CRAG

Per-question details go to evals/runs/{date}-{name}.jsonl. A summary row is appended to
evals/results.md only when running the real eval set (evals/questions.jsonl) and every question
succeeded.

Free-tier quotas: when the provider's daily token limit is hit, the run stops and exits with 3.
Re-running with --resume continues the latest run of the same name (on any day): questions that
succeeded are kept, the rest are asked again.
"""

import argparse
import contextlib
import datetime as dt
import json
import re
import sys
from pathlib import Path

from reportlens.agents.corrective import answer_corrective, crag_cache, default_tools
from reportlens.agents.routed import ask, default_answerers
from reportlens.agents.router import route_cache
from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.db import connect, connect_reader
from reportlens.core.graph_db import connect_graph
from reportlens.core.logging import setup_logging
from reportlens.evals.questions import load_questions
from reportlens.evals.resume import latest_run, load_finished
from reportlens.evals.runner import (
    AnswerFn,
    QuestionResult,
    evaluate_question,
    evaluate_retrieval,
    results_row,
    summarize,
)
from reportlens.generation.answer import Answer, answer_question
from reportlens.retrieval.filters import known_values
from reportlens.sources.graph_tool import answer_with_graph
from reportlens.sources.sql_tool import answer_with_sql, as_answer

EVALS_DIR = Path(__file__).resolve().parent
REAL_QUESTIONS = EVALS_DIR / "questions.jsonl"
RESULTS_MD = EVALS_DIR / "results.md"
RUNS_DIR = EVALS_DIR / "runs"
DAILY_LIMIT_EXIT = 3  # out of quota: re-run later with --resume


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ReportLens eval set.")
    parser.add_argument("--name", required=True, help="experiment name, e.g. 'baseline'")
    parser.add_argument("--questions", type=Path, default=REAL_QUESTIONS)
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--notes", default="", help="extra text for the results.md Notes column")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="score recall and MRR only, with no answer model or judge (cheap)",
    )
    parser.add_argument(
        "--source",
        choices=["text", "sql", "graph", "routed", "corrective"],
        default="text",
        help="answer from text search (default), the financial tables (Phase 3 SQL tool) "
        "the knowledge graph (Phase 3 graph tool), whichever the router picks (Phase 4) or "
        "text search with graded chunks (Phase 4 corrective RAG)",
    )
    parser.add_argument("--types", help="only these question types, e.g. numeric,trend")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the latest run of this name: keep its successes, re-ask the rest",
    )
    args = parser.parse_args()
    setup_logging(get_settings().log_level)
    if args.source != "text" and args.retrieval_only:
        parser.error("--retrieval-only applies to text search only")

    if not args.questions.exists():
        print(f"No questions file at {args.questions}. See CLAUDE.md 'Eval set format'.")
        return 1
    questions = load_questions(args.questions)
    if args.types:
        wanted = {t.strip() for t in args.types.split(",")}
        questions = [q for q in questions if q.type in wanted]
    questions = questions[: args.limit]

    slug = re.sub(r"[^a-z0-9]+", "-", args.name.lower()).strip("-")
    run_path = RUNS_DIR / f"{dt.date.today().isoformat()}-{slug}.jsonl"
    finished: dict[str, QuestionResult] = {}
    finished_routes: dict[str, str] = {}
    if args.resume and (earlier := latest_run(RUNS_DIR, slug)):
        run_path = earlier  # keep the file (and date) the run started with
        finished, finished_routes = load_finished(earlier)
        print(f"Resuming {earlier.name}: {len(finished)}/{len(questions)} already done")

    results: list[QuestionResult] = []
    out_of_quota = False
    uses_graph = args.source in ("graph", "routed")
    graph = connect_graph() if uses_graph else contextlib.nullcontext()
    routes: dict[str, str] = {}  # question -> sources tried (routed) or steps (corrective)
    with connect() as conn, connect_reader() as reader, graph as driver:

        def answer_from_sql(_conn: object, question: str) -> Answer:
            return as_answer(answer_with_sql(reader, question))  # the read-only connection

        def answer_from_graph(_conn: object, question: str) -> Answer:
            return as_answer(answer_with_graph(driver, question))

        answerers = default_answerers(conn, reader, driver) if args.source == "routed" else {}

        def answer_routed_fn(_conn: object, question: str) -> Answer:
            routed = ask(question, known_values(conn), answerers, route_cache())
            routes[question] = " -> ".join(routed.path + [f"check: {c}" for c in routed.checks])
            return routed.answer

        crag_tools = default_tools(conn, crag_cache()) if args.source == "corrective" else None

        def answer_corrective_fn(_conn: object, question: str) -> Answer:
            result = answer_corrective(question, crag_tools)
            routes[question] = " -> ".join(result.path)
            return result.answer

        answer_fn: AnswerFn = {
            "text": answer_question,
            "sql": answer_from_sql,
            "graph": answer_from_graph,
            "routed": answer_routed_fn,
            "corrective": answer_corrective_fn,
        }[args.source]
        for i, q in enumerate(questions, start=1):
            if q.id in finished:
                results.append(finished[q.id])
                continue
            if out_of_quota:
                continue  # left out of the file, so --resume asks it next time
            try:
                r = (
                    evaluate_retrieval(conn, q)
                    if args.retrieval_only
                    else evaluate_question(conn, q, answer_fn)
                )
            except Exception as err:  # one failure (e.g. rate limit) shouldn't sink the run
                if llm.is_daily_limit(err):
                    out_of_quota = True
                    print(f"[{i}/{len(questions)}] {q.id}: daily token limit reached; stopping")
                    continue
                r = QuestionResult(
                    id=q.id,
                    type=q.type,
                    question=q.question,
                    expected_answer=q.expected_answer,
                    error=f"{type(err).__name__}: {err}",
                )
            results.append(r)
            status = r.error or (
                f"recall={r.recall} rr={r.reciprocal_rank}"
                if args.retrieval_only
                else f"correct={r.correctness} recall={r.recall} faithful={r.faithfulness}"
            )
            route = f" route={routes[q.question]}" if q.question in routes else ""
            print(f"[{i}/{len(questions)}] {q.id}: {status}{route}", flush=True)

    today = dt.date.today().isoformat()
    RUNS_DIR.mkdir(exist_ok=True)
    run_path.write_text("".join(r.model_dump_json() + "\n" for r in results))
    if routes or finished_routes:  # sources tried or steps taken per question
        by_id = {r.id: routes.get(r.question) or finished_routes.get(r.id) for r in results}
        run_path.with_suffix(".routes.json").write_text(json.dumps(by_id, indent=1))
    if out_of_quota:
        left = len(questions) - sum(r.error is None for r in results)
        print(f"\nStopped on the daily token limit; {left} questions left. Details: {run_path}")
        print("Re-run the same command with --resume once the quota has recovered.")
        return DAILY_LIMIT_EXIT

    summary = summarize(results)
    auto = [
        "retrieval only" if args.retrieval_only else "",
        f"source={args.source}" if args.source != "text" else "",
        f"types={args.types}" if args.types else "",
    ]
    notes = "; ".join(filter(None, [*auto, args.notes]))
    row = results_row(today, args.name, summary, notes)
    print(f"\n{summary.model_dump_json(indent=2)}\n\n{row}\n\nDetails: {run_path}")

    if args.questions.resolve() != REAL_QUESTIONS:
        print("Not the real eval set, so results.md was not changed.")
    elif summary.errors:
        # Averages over the questions that happened to succeed would be misleading.
        print(
            f"{summary.errors} questions errored, so results.md was not changed. "
            "Re-run with --resume."
        )
    else:
        with RESULTS_MD.open("a") as f:
            f.write(row + "\n")
        print(f"Appended to {RESULTS_MD}")
    return 1 if summary.errors else 0


if __name__ == "__main__":
    sys.exit(main())
