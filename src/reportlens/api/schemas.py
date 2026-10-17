"""Request and response models for the API: one answer shape for every source."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from reportlens.generation.answer import Answer

Source = Literal["auto", "text", "corrective", "sql", "graph"]
MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    source: Source = "auto"  # auto: the router picks (Phase 4); the rest force one source

    @field_validator("question")
    @classmethod
    def not_blank(cls, question: str) -> str:
        if not question.strip():
            raise ValueError("question is blank")
        return question.strip()


class CitationOut(BaseModel):
    number: int  # the [n] in the answer text
    company: str  # "web" for web results
    year: int
    pages: list[int]
    url: str | None = None  # web results only


class ChunkOut(BaseModel):
    id: str
    company: str
    year: int
    section: str | None
    pages: list[int]
    text: str
    score: float


class RouteOut(BaseModel):
    source: str
    reason: str


class AskResponse(BaseModel):
    """The answer plus how it was found: route, steps, sources, cost."""

    question: str
    answer: str
    found: bool
    source: Source  # what was asked for
    route: RouteOut | None = None  # the router's pick (source=auto only)
    path: list[str] = []  # sources tried, or corrective RAG's steps
    checks: list[str] = []  # Self-RAG's steps (SELF_RAG=true)
    supported: bool | None = None  # Self-RAG's verdict
    query: str | None = None  # the SQL or Cypher behind a sql/graph answer
    citations: list[CitationOut]
    chunks: list[ChunkOut]  # everything the answer model saw, cited or not
    model: str | None
    input_tokens: int
    output_tokens: int
    latency_s: float


def response_fields(answer: Answer) -> dict:
    """The parts of an AskResponse that come straight from an Answer."""
    return {
        "question": answer.question,
        "answer": answer.text,
        "found": answer.found,
        "citations": [
            CitationOut(
                number=c.number,
                company=c.company,
                year=c.year,
                pages=c.pages,
                url=c.chunk_id if c.company == "web" else None,
            )
            for c in answer.citations
        ],
        "chunks": [
            ChunkOut(
                id=c.chunk_id,
                company=c.company,
                year=c.year,
                section=c.section,
                pages=c.pages,
                text=c.text,
                score=c.score,
            )
            for c in answer.chunks
        ],
        "model": answer.model,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "latency_s": answer.latency_s,
    }


class Health(BaseModel):
    postgres: bool
    neo4j: bool
    web_search: bool
    self_rag: bool


class Meta(BaseModel):
    companies: list[str]
    years: list[int]
    sources: list[Source]
