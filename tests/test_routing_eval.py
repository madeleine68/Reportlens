import json
from pathlib import Path

from reportlens.evals.routing import RouteCheck, best_sources, check_routes, load_scores


def _run(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_load_scores_skips_unscored_rows_and_later_runs_override(tmp_path: Path) -> None:
    old = _run(
        tmp_path / "a.jsonl", [{"id": "q1", "correctness": 0.0}, {"id": "q2", "correctness": None}]
    )
    new = _run(tmp_path / "b.jsonl", [{"id": "q1", "correctness": 1.0}])
    sql = _run(tmp_path / "c.jsonl", [{"id": "q1", "correctness": 0.5}])
    assert load_scores({"vector": [old, new], "sql": [sql]}) == {"q1": {"vector": 1.0, "sql": 0.5}}


def test_best_sources_keeps_ties_and_ignores_all_zero() -> None:
    assert best_sources({"vector": 1.0, "sql": 1.0, "graph": 0.5}) == ["sql", "vector"]
    assert best_sources({"vector": 0.0, "sql": 0.0}) == []


def test_check_routes_needs_two_sources_and_a_right_one() -> None:
    scores = {
        "q1": {"vector": 0.5, "sql": 1.0},
        "q2": {"vector": 1.0},  # one source: no comparison
        "q3": {"vector": 0.0, "graph": 0.0},  # nobody right
        "q4": {"vector": 1.0, "graph": 0.0},
    }
    routed = {"q1": "sql", "q2": "vector", "q3": "graph", "q4": "web"}
    types = dict.fromkeys(routed, "numeric")
    checks = check_routes(routed, types, scores)
    assert [(c.id, c.correct) for c in checks] == [
        ("q1", True),
        ("q4", True),
    ]  # web counts as vector


def test_wrong_route_is_reported() -> None:
    check = RouteCheck(id="q", type="t", routed="graph", best=["vector"], scores={})
    assert not check.correct
