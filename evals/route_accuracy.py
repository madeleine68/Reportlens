"""Check the router's picks against the source that scored best per question (Phase 4).

Makes one fast-model call per question (cached in data/cache/routes.json) and no answer calls.
Usage:
    uv run python evals/route_accuracy.py
"""

import sys
from collections import Counter
from pathlib import Path

from reportlens.agents.router import route_cache, route_question
from reportlens.core.config import get_settings
from reportlens.core.db import connect
from reportlens.core.logging import setup_logging
from reportlens.evals.questions import load_questions
from reportlens.evals.routing import check_routes, load_scores
from reportlens.retrieval.filters import known_values

EVALS_DIR = Path(__file__).resolve().parent
RUNS = EVALS_DIR / "runs"
# The best recorded run of each source (see results.md); later files override earlier ones.
REFERENCE_RUNS = {
    "vector": [
        RUNS / "2026-10-10-hybrid-filters-rerank-naive-chunks-n-42.jsonl",
        RUNS / "2026-10-16-text-search-best-stack-n-9-relationship.jsonl",
    ],
    "sql": [RUNS / "2026-10-15-sql-tool-comparison-rule-n-33.jsonl"],
    "graph": [RUNS / "2026-10-16-graph-tool-n-9-relationship.jsonl"],
}


def main() -> int:
    setup_logging(get_settings().log_level)
    questions = load_questions(EVALS_DIR / "questions.jsonl")
    cache = route_cache()
    with connect() as conn:
        known = known_values(conn)
    routed = {q.id: route_question(q.question, known, cache).source for q in questions}

    print("Routes for all", len(questions), "questions:", dict(Counter(routed.values())))
    by_type: dict[str, Counter[str]] = {}
    for q in questions:
        by_type.setdefault(q.type, Counter())[routed[q.id]] += 1
    for qtype, counts in sorted(by_type.items()):
        print(f"  {qtype:<14} {dict(counts)}")

    types = {q.id: q.type for q in questions}
    checks = check_routes(routed, types, load_scores(REFERENCE_RUNS))
    right = sum(c.correct for c in checks)
    print(f"\nRoute accuracy: {right}/{len(checks)} = {right / len(checks):.2f}")
    print("(questions scored by 2+ sources where at least one was right; ties count either way)")
    for c in checks:
        if not c.correct:
            print(f"  {c.id} ({c.type}): routed {c.routed}, best {c.best}, scores {c.scores}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
