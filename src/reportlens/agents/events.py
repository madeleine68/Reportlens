"""Progress events from the agents (route chosen, each graph step), for the API's stream.

A listener is set for the duration of one question with `listening(callback)`; without one,
`emit` does nothing, so the CLI, the evals and the tests are unaffected. A ContextVar keeps
listeners separate when several questions run at once in different threads.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

Listener = Callable[[str, dict[str, Any]], None]  # (event kind, data)

_listener: ContextVar[Listener | None] = ContextVar("reportlens_listener", default=None)


def emit(kind: str, **data: Any) -> None:
    """Report progress to the current listener, if any."""
    if (listener := _listener.get()) is not None:
        listener(kind, data)


@contextmanager
def listening(listener: Listener) -> Iterator[None]:
    """Send this thread's events to `listener` until the block ends."""
    token = _listener.set(listener)
    try:
        yield
    finally:
        _listener.reset(token)


def run_graph(graph: Any, start: dict[str, Any], config: dict[str, Any], stage: str) -> dict:
    """graph.invoke(start, config), emitting each new entry of the state's `path` as a step.

    LangGraph's "values" stream yields the whole state after every node, so the last one is the
    final state and new path entries are what that node did.
    """
    state, seen = start, len(start.get("path", []))
    for state in graph.stream(start, config, stream_mode="values"):
        for step in state.get("path", [])[seen:]:
            emit("step", stage=stage, text=step)
        seen = len(state.get("path", []))
    return state
