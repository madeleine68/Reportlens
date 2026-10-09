"""The eval question format (evals/questions.jsonl, one JSON object per line) and its loader."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

QuestionType = Literal["lookup", "numeric", "comparison", "trend", "relationship", "out_of_corpus"]


class ExpectedSource(BaseModel):
    """A page where the answer can be found."""

    company: str
    year: int
    page: int


class EvalQuestion(BaseModel):
    id: str
    question: str
    type: QuestionType
    expected_answer: str
    sources: list[ExpectedSource] = []  # empty for out_of_corpus questions
    notes: str = ""


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read a JSONL file; blank lines and lines starting with // are skipped."""
    questions = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            questions.append(EvalQuestion.model_validate_json(line))
        except ValueError as err:
            raise ValueError(f"{path}:{line_no}: {err}") from err
    return questions
