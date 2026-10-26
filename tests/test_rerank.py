import pytest

from reportlens.retrieval import rerank as rerank_mod
from reportlens.retrieval import search as search_mod
from reportlens.retrieval.models import RetrievedChunk
from reportlens.retrieval.rerank import rerank


def _c(chunk_id: str, text: str = "", score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, company="rbc", year=2023, section=None, pages=[1], text=text, score=score
    )


class FakeCrossEncoder:
    """Scores a pair by how many question words appear in the chunk."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
        self.pairs = pairs
        return [float(sum(w in text for w in question.split())) for question, text in pairs]


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeCrossEncoder:
    model = FakeCrossEncoder()
    monkeypatch.setattr(rerank_mod, "_model", lambda: model)
    return model


def test_reorders_by_cross_encoder_score_and_keeps_top_n(fake_model: FakeCrossEncoder) -> None:
    chunks = [
        _c("a", "board meeting", score=0.9),
        _c("b", "net income rose", score=0.1),
        _c("c", "net income", score=0.5),
    ]
    results = rerank("net income rose", chunks, top_n=2)

    assert [r.chunk_id for r in results] == ["b", "c"]  # the retriever's order is ignored
    assert [r.score for r in results] == [3.0, 2.0]  # score is now the reranker's
    assert fake_model.pairs[0] == ("net income rose", "board meeting")  # question paired with text


def test_reranker_reads_the_label_with_the_text(fake_model: FakeCrossEncoder) -> None:
    labelled = _c("a", "net income").model_copy(update={"context": "RBC 2023 annual report"})
    rerank("q", [labelled], top_n=1)
    assert fake_model.pairs == [("q", "RBC 2023 annual report\nnet income")]


def test_empty_candidates_skip_the_model(fake_model: FakeCrossEncoder) -> None:
    assert rerank("q", [], top_n=5) == []
    assert fake_model.pairs == []


@pytest.mark.parametrize("enabled", [True, False])
def test_search_fetches_more_candidates_only_when_reranking(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reportlens.core.config import get_settings

    monkeypatch.setenv("RETRIEVAL_RERANK", str(enabled).lower())
    get_settings.cache_clear()
    requested: list[int] = []

    def fake_vector_search(conn, question, k, filters, kind=None):
        requested.append(k)
        return [_c(f"c{i}") for i in range(k)]

    monkeypatch.setattr(search_mod, "vector_search", fake_vector_search)
    monkeypatch.setattr(search_mod, "rerank", lambda q, chunks, top_n: chunks[::-1][:top_n])

    results = search_mod.search(None, "q")

    assert requested == [40 if enabled else 5]
    assert len(results) == 5
    assert results[0].chunk_id == ("c39" if enabled else "c0")


@pytest.mark.slow
def test_real_cross_encoder_prefers_the_relevant_chunk() -> None:
    """Uses ms-marco-MiniLM-L-6-v2 (downloads on first run)."""
    chunks = [
        _c("noise", "The board met eleven times and reviewed its committee charters."),
        _c("answer", "Net income of $14,866 million was down $941 million or 6% from last year."),
    ]
    assert rerank("What was RBC's net income in 2023?", chunks, top_n=2)[0].chunk_id == "answer"
