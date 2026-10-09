"""The eval question format (evals/questions.jsonl, one JSON object per line) and its loader."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator

QuestionType = Literal["lookup", "numeric", "comparison", "trend", "relationship", "out_of_corpus"]


class ExpectedSource(BaseModel):
    """One piece of evidence the answer needs, and every PDF page where it appears.

    Retrieving any one of `pages` counts as finding this source. Pages are PDF page numbers
    (1 = first page of the file), not the numbers printed on the page.
    """

    company: str
    year: int
    pages: list[int]

    @model_validator(mode="before")
    @classmethod
    def _accept_single_page(cls, data: object) -> object:
        """Allow the shorthand {"page": 12} for {"pages": [12]}."""
        if isinstance(data, dict) and "page" in data and "pages" not in data:
            data = {**data, "pages": [data["page"]]}
            del data["page"]
        return data


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
