import numpy as np
import pytest

from reportlens.ingest import embed


class FakeModel:
    """Records what it was asked to encode; returns unit vectors."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.seen.extend(texts)
        assert kwargs["normalize_embeddings"] is True
        return np.tile(np.eye(1, 384), (len(texts), 1))


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel()
    monkeypatch.setattr(embed, "_model", lambda: model)
    return model


def test_query_gets_instruction_prefix(fake_model: FakeModel) -> None:
    vector = embed.embed_query("What was RBC's net income?")
    assert fake_model.seen == [
        "Represent this sentence for searching relevant passages: What was RBC's net income?"
    ]
    assert vector.shape == (384,)


def test_passages_are_embedded_as_is(fake_model: FakeModel) -> None:
    vectors = embed.embed_passages(["chunk one", "chunk two"])
    assert fake_model.seen == ["chunk one", "chunk two"]
    assert vectors.shape == (2, 384)


def test_query_prefix_is_configurable(
    fake_model: FakeModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reportlens.core.config import get_settings

    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "")
    get_settings.cache_clear()
    embed.embed_query("plain question")
    assert fake_model.seen == ["plain question"]


@pytest.mark.slow
def test_real_model_ranks_relevant_passage_first() -> None:
    """Uses bge-small (downloads on first run)."""
    passages = embed.embed_passages(
        ["Net income rose to $16.2 billion in 2023.", "The board met four times this year."]
    )
    query = embed.embed_query("How much profit did the bank make?")
    scores = passages @ query
    assert passages.shape == (2, 384)
    assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-5)
    assert scores[0] > scores[1]
