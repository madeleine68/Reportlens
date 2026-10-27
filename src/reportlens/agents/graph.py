r"""Agentic RAG (Phase 4): the router and the sources it picks, as one LangGraph state graph.

    START -> route --(sql | graph | web | none)--> ask_routed --found--> [check] -> END
                 \                                     |
                  \--(vector, or source not on)--+     | nothing found, or the source failed
                                                 v     v
                                                 fallback (text search) --> [check] -> END

- route: the fast LLM picks a source (agents/router.py); the pick is sent as a "route" event.
- ask_routed: the picked source answers. SQL, the graph and the web answer only what they hold,
  so finding nothing, or a broken tool, moves the question to text search instead of failing it.
  A rate-limit error is the exception: it stops the question, so running out of quota is never
  mistaken for "nothing found".
- fallback: text search (corrective RAG by default), which always answers, refusing if it must.
- check: Self-RAG, only with SELF_RAG=true (agents/self_rag.py).

The graph only wires these steps together; each step's logic lives in its own module, and the
answer functions come from agents/routed.py, so tests can pass stubs.
"""

import operator
from collections.abc import Callable
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from reportlens.agents.events import emit
from reportlens.agents.router import Route
from reportlens.agents.self_rag import CheckedAnswer
from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.generation.answer import Answer

log = get_logger(__name__)

AnswerFn = Callable[[str], Answer]
RouteFn = Callable[[str], Route]
CheckFn = Callable[[Answer], CheckedAnswer]
FALLBACK = "vector"


class RoutedAnswer(BaseModel):
    """The answer, the router's decision, and the sources tried in order (e.g. sql → vector)."""

    answer: Answer
    route: Route
    path: list[str]
    checks: list[str] = []  # Self-RAG's steps, e.g. ["2 unsupported", "revise", "supported"]
    supported: bool | None = None  # Self-RAG's verdict (None: not checked)


class RoutedState(TypedDict, total=False):
    question: str
    route: Route
    answer: Answer  # set once a source has answered
    attempts: list[Answer]  # answers that found nothing, whose cost still counts
    path: Annotated[list[str], operator.add]
    checks: list[str]
    supported: bool | None


def add_costs(answer: Answer, earlier: list[Answer]) -> Answer:
    """The answer, with the tokens and time of the attempts that came before it added in."""
    return answer.model_copy(
        update={
            "input_tokens": answer.input_tokens + sum(a.input_tokens for a in earlier),
            "output_tokens": answer.output_tokens + sum(a.output_tokens for a in earlier),
            "latency_s": round(answer.latency_s + sum(a.latency_s for a in earlier), 3),
        }
    )


def build_graph(  # noqa: ANN201  # LangGraph's compiled type
    route_fn: RouteFn, answerers: dict[str, AnswerFn], check: CheckFn | None = None
):
    """The routed-answer state graph, compiled and ready to invoke."""
    finish = "check" if check else END

    def route(state: RoutedState) -> RoutedState:
        picked = route_fn(state["question"])
        emit("route", source=picked.source, reason=picked.reason)
        return {"route": picked}

    def after_route(state: RoutedState) -> Literal["ask_routed", "fallback"]:
        source = state["route"].source
        return "ask_routed" if source != FALLBACK and source in answerers else "fallback"

    def ask_routed(state: RoutedState) -> RoutedState:
        source = state["route"].source
        emit("step", stage="routed", text=f"asking {source}")
        try:
            answer = answerers[source](state["question"])
        except llm.RateLimitError:
            raise  # out of quota, not a broken tool: falling back would hide it
        except Exception as err:  # noqa: BLE001  # a broken tool falls back to text search
            log.warning("routed.source_failed source=%s error=%s", source, err)
            return {"path": [f"{source} (error)"]}
        if answer.found:
            return {"answer": answer, "path": [source]}
        return {"attempts": [answer], "path": [source]}

    def after_ask(state: RoutedState) -> str:
        return finish if "answer" in state else "fallback"

    def fallback(state: RoutedState) -> RoutedState:
        emit("step", stage="routed", text=f"asking {FALLBACK}")
        answer = answerers[FALLBACK](state["question"])
        return {"answer": add_costs(answer, state.get("attempts", [])), "path": [FALLBACK]}

    def check_answer(state: RoutedState) -> RoutedState:
        assert check is not None  # the node only exists when a check is given
        checked = check(state["answer"])
        return {"answer": checked.answer, "checks": checked.path, "supported": checked.supported}

    graph = StateGraph(RoutedState)
    graph.add_node("route", route)
    graph.add_node("ask_routed", ask_routed)
    graph.add_node("fallback", fallback)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route)
    graph.add_conditional_edges("ask_routed", after_ask, [finish, "fallback"])
    graph.add_edge("fallback", finish)
    if check:
        graph.add_node("check", check_answer)
        graph.add_edge("check", END)
    return graph.compile()


def run_routed(
    question: str,
    route_fn: RouteFn,
    answerers: dict[str, AnswerFn],
    check: CheckFn | None = None,
) -> RoutedAnswer:
    """Route the question, answer it (falling back to text search), and check it if asked."""
    graph = build_graph(route_fn, answerers, check)
    start: RoutedState = {"question": question, "attempts": [], "path": []}
    state = graph.invoke(start, {"recursion_limit": get_settings().agent_max_steps})
    return RoutedAnswer(
        answer=state["answer"],
        route=state["route"],
        path=state["path"],
        checks=state.get("checks", []),
        supported=state.get("supported"),
    )
