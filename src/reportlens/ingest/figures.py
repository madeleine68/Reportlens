"""Chart summaries (Phase 3): find charts in the reports, have a vision model describe them.

Docling's layout model marks pictures even in light mode, but drops the text inside them, so the
charts were invisible to search. Most charts in these reports are vector graphics whose labels
and values are real text in the PDF, which gives a free first filter:

1. Candidates: pictures (from the cached Docling output) with at least MIN_NUMBERS numbers in
   the PDF text inside their box. Photos and logos have none. No re-parse, no LLM.
2. Each candidate is cropped from the PDF and sent, with its printed text, to the vision model,
   which returns a ChartSummary. Values read off a bar's height are marked printed=False, since
   they are estimates (one test read 98,000 for a bar at about 110,000).
3. Each chart becomes one chunk (kind 'figure') that search can find and answers can cite.

Summaries are cached by model, prompt and image, so re-runs cost nothing.
"""

import io
import re
from collections.abc import Callable

from docling_core.types.doc import CoordOrigin, DocItemLabel, DoclingDocument, PictureItem, TextItem
from pydantic import BaseModel, Field
from pypdfium2 import PdfDocument

from reportlens.core import llm
from reportlens.core.config import get_settings
from reportlens.core.logging import get_logger
from reportlens.generation import prompts
from reportlens.ingest.graph import COMPANY_NAMES, FactCache
from reportlens.ingest.models import Chunk

log = get_logger(__name__)

MIN_NUMBERS = 4  # numbers printed inside a picture before it counts as a chart candidate
MAX_IMAGE_SIDE = 1024  # pixels; bigger images cost more vision tokens without reading better
MAX_RENDER_SCALE = 2.0  # 144 dpi: small chart labels stay legible
PRINTED_TEXT_CHARS = 1500  # printed text sent with the image (and kept in the chunk)

_NUMBER = re.compile(r"\d[\d,.]*")
_HEADINGS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}

Box = tuple[float, float, float, float]  # left, top, right, bottom in PDF points, top-left origin
TextInBox = Callable[[int, Box], str]  # (page number, box) -> the PDF text inside it
Structured = Callable[..., BaseModel]


class FigureCandidate(BaseModel):
    """A picture that may be a chart, with where it is and the text printed inside it."""

    company: str
    year: int
    page: int
    box: Box
    printed_text: str
    caption: str
    heading: str  # nearest heading above it: often the chart's title and unit, printed outside


class ChartValue(BaseModel):
    """One data point of a chart."""

    label: str
    period: str | None = None
    value: float
    unit: str | None = None
    printed: bool  # False: estimated from the chart's shape, not printed on it


class ChartSummary(BaseModel):
    """What the vision model returns for one picture."""

    is_chart: bool
    title: str = ""
    summary: str = ""
    values: list[ChartValue] = Field(default_factory=list)


def count_numbers(text: str) -> int:
    """How many numbers (e.g. '2023', '15,000', '13.2') a piece of text holds."""
    return len(_NUMBER.findall(text))


def picture_box(item: PictureItem, page_height: float) -> Box:
    """The picture's box in PDF points with a top-left origin."""
    bbox = item.prov[0].bbox
    if bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
        bbox = bbox.to_top_left_origin(page_height)
    return (bbox.l, bbox.t, bbox.r, bbox.b)


def chart_candidates(
    doc: DoclingDocument, company: str, year: int, text_in_box: TextInBox
) -> list[FigureCandidate]:
    """Pictures with at least MIN_NUMBERS numbers printed inside them, in reading order."""
    candidates = []
    heading = ""
    for item, _level in doc.iterate_items():
        if isinstance(item, TextItem) and item.label in _HEADINGS:
            heading = item.text.strip()
        if not isinstance(item, PictureItem) or not item.prov:
            continue
        page = item.prov[0].page_no
        box = picture_box(item, doc.pages[page].size.height)
        text = " ".join(text_in_box(page, box).split())
        if count_numbers(text) >= MIN_NUMBERS:
            candidates.append(
                FigureCandidate(
                    company=company,
                    year=year,
                    page=page,
                    box=box,
                    printed_text=text[:PRINTED_TEXT_CHARS],
                    caption=item.caption_text(doc),
                    heading=heading,
                )
            )
    return candidates


def pdf_text_in_box(pdf: PdfDocument) -> TextInBox:
    """A TextInBox that reads the PDF's own text layer."""

    def text_in_box(page_no: int, box: Box) -> str:
        page = pdf[page_no - 1]
        height = page.get_height()
        left, top, right, bottom = box
        return page.get_textpage().get_text_bounded(
            left=left, bottom=height - bottom, right=right, top=height - top
        )

    return text_in_box


def render_png(pdf: PdfDocument, page_no: int, box: Box) -> bytes:
    """The box cropped from the page as a PNG, at most MAX_IMAGE_SIDE pixels on its long side."""
    page = pdf[page_no - 1]
    left, top, right, bottom = box
    scale = min(MAX_RENDER_SCALE, MAX_IMAGE_SIDE / max(right - left, bottom - top, 1.0))
    # pdfium's crop is the margin to cut from each side: left, bottom, right, top.
    crop = (left, page.get_height() - bottom, page.get_width() - right, top)
    image = page.render(scale=scale, crop=crop).to_pil()
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def figure_messages(candidate: FigureCandidate, png: bytes) -> list[llm.Message]:
    """System prompt plus the image and its printed text."""
    system = prompts.FIGURE_SYSTEM.format(
        company_name=COMPANY_NAMES.get(candidate.company, candidate.company),
        year=candidate.year,
        page=candidate.page,
    )
    user = prompts.FIGURE_USER.format(
        heading=candidate.heading or "(none)",
        printed_text=candidate.printed_text or "(none)",
        caption=candidate.caption or "(none)",
    )
    return [{"role": "system", "content": system}, llm.image_message(user, png)]


def cache_key(candidate: FigureCandidate, png: bytes) -> str:
    """Cache key of a picture's summary: the vision model plus the exact messages (image too)."""
    return FactCache.key(figure_messages(candidate, png), model=get_settings().llm_model_vision)


def describe(
    candidate: FigureCandidate,
    png: bytes,
    cache: FactCache,
    structured: Structured = llm.structured,
) -> ChartSummary:
    """The vision model's summary of one picture, reusing a cached one for the same image."""
    messages = figure_messages(candidate, png)
    key = cache_key(candidate, png)
    if (cached := cache.get(key)) is not None:
        return ChartSummary.model_validate(cached)
    summary = structured(messages, ChartSummary, tier="vision", max_retries=2)
    cache.put(key, summary.model_dump())
    log.info(
        "figures.described %s/%d p%d chart=%s values=%d",
        candidate.company,
        candidate.year,
        candidate.page,
        summary.is_chart,
        len(summary.values),
    )
    return summary


def format_value(value: ChartValue) -> str:
    """'Deposits, 2023: 235,000 millions of dollars (estimated from the chart)'."""
    label = value.label
    if value.period and value.period != value.label:
        label = f"{label}, {value.period}"
    number = f"{value.value:,.0f}" if value.value.is_integer() else f"{value.value:,}"
    unit = f" {value.unit}" if value.unit else ""
    note = "" if value.printed else " (estimated from the chart)"
    return f"{label}: {number}{unit}{note}"


def figure_text(candidate: FigureCandidate, summary: ChartSummary) -> str:
    """A chart's chunk text: title, summary, values (estimates marked) and its printed text."""
    lines = [f"Chart, page {candidate.page}: {summary.title}", summary.summary]
    if summary.values:
        lines.append("Values: " + "; ".join(format_value(v) for v in summary.values) + ".")
    if candidate.caption:
        lines.append(f"Caption: {candidate.caption}")
    lines.append(f"Printed in the chart: {candidate.printed_text}")
    return "\n".join(line for line in lines if line)


def figure_chunks(
    described: list[tuple[FigureCandidate, ChartSummary]], count_tokens: Callable[[str], int]
) -> list[Chunk]:
    """One chunk per chart; pictures the model says aren't charts are dropped."""
    chunks = []
    for candidate, summary in described:
        if not summary.is_chart:
            continue
        text = figure_text(candidate, summary)
        chunks.append(
            Chunk(
                chunk_id=f"{candidate.company}-{candidate.year}-fig-{len(chunks) + 1:03d}",
                company=candidate.company,
                year=candidate.year,
                text=text,
                pages=[candidate.page],
                section=summary.title or None,
                num_tokens=count_tokens(text),
            )
        )
    return chunks
