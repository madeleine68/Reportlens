"""Route accuracy (Phase 4): does the router pick the source that scored best on each question?

The reference comes from eval runs already recorded per source: for each question, the sources
with the highest correctness are the right routes (ties: any of them). Only questions scored by
at least two sources count, since one score can't say which source is better. Routing to 'web'
counts as 'vector', because web questions are answered by text search until a web tool exists.
"""

import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel


class RouteCheck(BaseModel):
    """One question: the router's pick, the best sources and their scores."""

    id: str
    type: str
    routed: str
    best: list[str]
    scores: dict[str, float]

    @property
    def correct(self) -> bool:
        routed = "vector" if self.routed == "web" else self.routed
        return routed in self.best


def load_scores(runs: dict[str, list[Path]]) -> dict[str, dict[str, float]]:
    """{question id: {source: correctness}} from run files; later files override earlier ones."""
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for source, paths in runs.items():
        for path in paths:
            for line in path.read_text().splitlines():
                result = json.loads(line)
                if result.get("correctness") is not None:
                    scores[result["id"]][source] = result["correctness"]
    return dict(scores)


def best_sources(scores: dict[str, float]) -> list[str]:
    """The sources with the top score, or none if every source scored 0."""
    top = max(scores.values())
    return sorted(s for s, score in scores.items() if score == top) if top > 0 else []


def check_routes(
    routed: dict[str, str], types: dict[str, str], scores: dict[str, dict[str, float]]
) -> list[RouteCheck]:
    """A RouteCheck for each question scored by at least two sources, one of which got it right."""
    checks = []
    for qid, source in routed.items():
        by_source = scores.get(qid, {})
        if len(by_source) < 2 or not (best := best_sources(by_source)):
            continue
        checks.append(
            RouteCheck(id=qid, type=types[qid], routed=source, best=best, scores=by_source)
        )
    return checks
