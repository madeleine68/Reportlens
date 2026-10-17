"""Resuming an eval run across days, for free-tier quotas a full run doesn't fit in."""

import json
from pathlib import Path

from reportlens.evals.runner import QuestionResult


def latest_run(runs_dir: Path, slug: str) -> Path | None:
    """The newest run file for this experiment name, from any day."""
    runs = sorted(runs_dir.glob(f"????-??-??-{slug}.jsonl"))
    return runs[-1] if runs else None


def load_finished(path: Path) -> tuple[dict[str, QuestionResult], dict[str, str]]:
    """Results that succeeded in an earlier run (by question id), and their routes."""
    results = [QuestionResult.model_validate_json(line) for line in path.read_text().splitlines()]
    routes_path = path.with_suffix(".routes.json")
    routes = json.loads(routes_path.read_text()) if routes_path.exists() else {}
    finished = {r.id: r for r in results if r.error is None}
    return finished, {qid: route for qid, route in routes.items() if qid in finished and route}
