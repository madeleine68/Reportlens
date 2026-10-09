"""LLM-as-judge for correctness and faithfulness, on the fast model (grading is cheap work)."""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from reportlens.core import llm
from reportlens.retrieval.models import RetrievedChunk

Structured = Callable[..., BaseModel]


class CorrectnessVerdict(BaseModel):
    reason: str  # asked for first, so the model explains before it decides
    verdict: Literal["correct", "partially_correct", "incorrect"]


class FaithfulnessVerdict(BaseModel):
    reason: str
    total_claims: int
    unsupported_claims: list[str]


CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}

_CORRECTNESS_SYSTEM = """You grade answers to questions about company annual reports.
Compare the ANSWER with the EXPECTED answer.
- correct: states the expected facts; numbers match (ignore formatting such as $16.2B vs \
$16,231 million) and refer to the right company and year. Extra correct detail is fine.
- partially_correct: some expected facts are right but others are missing or wrong.
- incorrect: wrong, missing the key fact, about the wrong company/year, or a refusal."""

_FAITHFULNESS_SYSTEM = """You check whether an answer is supported by its sources.
Split the ANSWER into factual claims. A claim is supported only if the SOURCES state it \
(simple arithmetic on stated numbers counts as supported). List every unsupported claim. \
Ignore citation markers like [1]."""


def judge_correctness(
    question: str, expected: str, answer: str, structured: Structured = llm.structured
) -> CorrectnessVerdict:
    """Grade an answer against the expected answer."""
    user = f"QUESTION: {question}\n\nEXPECTED: {expected}\n\nANSWER: {answer}"
    messages = [
        {"role": "system", "content": _CORRECTNESS_SYSTEM},
        {"role": "user", "content": user},
    ]
    return structured(messages, CorrectnessVerdict, tier="fast")


def judge_faithfulness(
    answer: str, chunks: list[RetrievedChunk], structured: Structured = llm.structured
) -> FaithfulnessVerdict:
    """List the answer's claims that its retrieved sources don't support."""
    sources = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))
    user = f"SOURCES:\n{sources}\n\nANSWER: {answer}"
    messages = [
        {"role": "system", "content": _FAITHFULNESS_SYSTEM},
        {"role": "user", "content": user},
    ]
    return structured(messages, FaithfulnessVerdict, tier="fast")


def faithfulness_score(verdict: FaithfulnessVerdict) -> float:
    """Share of claims that are supported (1.0 when the answer makes no claims)."""
    if verdict.total_claims <= 0:
        return 1.0
    unsupported = min(len(verdict.unsupported_claims), verdict.total_claims)
    return 1 - unsupported / verdict.total_claims
