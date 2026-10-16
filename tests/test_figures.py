from pathlib import Path

import pytest
from docling_core.types.doc import (
    BoundingBox,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
)
from pydantic import BaseModel

from reportlens.core import llm
from reportlens.ingest import figures
from reportlens.ingest.figures import (
    ChartSummary,
    ChartValue,
    FigureCandidate,
    chart_candidates,
    count_numbers,
    describe,
    figure_chunks,
    figure_text,
    format_value,
)
from reportlens.ingest.graph import FactCache

PAGE_HEIGHT = 800.0


def _prov(page: int, l: float, t: float, r: float, b: float) -> ProvenanceItem:  # noqa: E741
    """Provenance with a bottom-left-origin box, as Docling stores it; l/t/r/b in top-left terms."""
    box = BoundingBox(
        l=l, t=PAGE_HEIGHT - t, r=r, b=PAGE_HEIGHT - b, coord_origin=CoordOrigin.BOTTOMLEFT
    )
    return ProvenanceItem(page_no=page, bbox=box, charspan=(0, 0))


def _doc() -> DoclingDocument:
    """Page 1: a heading, a chart (numbers inside) and a photo; page 2: another chart."""
    doc = DoclingDocument(name="test")
    for page in (1, 2):
        doc.add_page(page_no=page, size=Size(width=600, height=PAGE_HEIGHT))
    doc.add_heading("OPERATING REVENUES ($ millions)", prov=_prov(1, 50, 40, 300, 60))
    doc.add_picture(prov=_prov(1, 50, 100, 300, 300))  # chart
    doc.add_picture(prov=_prov(1, 320, 100, 550, 300))  # photo
    doc.add_text(DocItemLabel.TEXT, "Body text", prov=_prov(2, 50, 40, 300, 60))
    doc.add_picture(prov=_prov(2, 50, 100, 300, 300))
    return doc


PRINTED = {
    (1, 50.0): "2021 +2.5% $23,449  2020 $22,883",
    (1, 320.0): "Our people",
    (2, 50.0): "2019 29 2020 19 2021 13",
}


def _text_in_box(page: int, box: figures.Box) -> str:
    return PRINTED[(page, box[0])]


def _candidate(**overrides: object) -> FigureCandidate:
    values = {
        "company": "bce",
        "year": 2021,
        "page": 7,
        "box": (50.0, 100.0, 300.0, 300.0),
        "printed_text": "2021 $23,449 2020 $22,883",
        "caption": "",
        "heading": "OPERATING REVENUES ($ millions)",
    }
    return FigureCandidate(**{**values, **overrides})


SUMMARY = ChartSummary(
    is_chart=True,
    title="Operating revenues",
    summary="Revenues rose 2.5% in 2021.",
    values=[
        ChartValue(label="2021", period="2021", value=23449, unit="$ millions", printed=True),
        ChartValue(label="Deposits", period="2022", value=232000.0, unit=None, printed=False),
    ],
)


def test_count_numbers() -> None:
    assert count_numbers("2021 +2.5% $23,449 and (10.5%)") == 3 + 1
    assert count_numbers("Our people") == 0


def test_candidates_are_pictures_with_numbers_printed_inside_with_their_heading() -> None:
    found = chart_candidates(_doc(), "bce", 2021, _text_in_box)
    assert [(c.page, c.box) for c in found] == [
        (1, (50.0, 100.0, 300.0, 300.0)),  # converted to a top-left origin
        (2, (50.0, 100.0, 300.0, 300.0)),
    ]
    assert found[0].printed_text == "2021 +2.5% $23,449 2020 $22,883"  # whitespace collapsed
    assert found[0].heading == "OPERATING REVENUES ($ millions)"
    assert found[1].heading == "OPERATING REVENUES ($ millions)"  # body text doesn't reset it


def test_format_value_marks_estimates_and_skips_a_period_equal_to_the_label() -> None:
    printed, estimated = SUMMARY.values
    assert format_value(printed) == "2021: 23,449 $ millions"
    assert format_value(estimated) == "Deposits, 2022: 232,000 (estimated from the chart)"
    assert format_value(ChartValue(label="CET1", value=13.2, unit="%", printed=True)) == (
        "CET1: 13.2 %"
    )


def test_figure_text_holds_title_summary_values_and_printed_text() -> None:
    text = figure_text(_candidate(caption="Figure 3"), SUMMARY)
    assert text.splitlines() == [
        "Chart, page 7: Operating revenues",
        "Revenues rose 2.5% in 2021.",
        "Values: 2021: 23,449 $ millions; Deposits, 2022: 232,000 (estimated from the chart).",
        "Caption: Figure 3",
        "Printed in the chart: 2021 $23,449 2020 $22,883",
    ]


def test_figure_chunks_drop_non_charts_and_number_the_rest() -> None:
    photo = ChartSummary(is_chart=False)
    chunks = figure_chunks(
        [(_candidate(page=3), photo), (_candidate(page=7), SUMMARY), (_candidate(page=9), SUMMARY)],
        count_tokens=len,
    )
    assert [(c.chunk_id, c.pages, c.section) for c in chunks] == [
        ("bce-2021-fig-001", [7], "Operating revenues"),
        ("bce-2021-fig-002", [9], "Operating revenues"),
    ]
    assert chunks[0].num_tokens == len(chunks[0].text)


def test_describe_sends_the_image_to_the_vision_tier_and_caches_the_result(
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[llm.Message], str]] = []

    def fake_structured(
        messages: list[llm.Message], schema: type[BaseModel], tier: str, **_: object
    ) -> BaseModel:
        calls.append((messages, tier))
        return SUMMARY

    cache = FactCache(tmp_path / "figures.json")
    assert describe(_candidate(), b"png-bytes", cache, fake_structured) == SUMMARY
    assert describe(_candidate(), b"png-bytes", cache, fake_structured) == SUMMARY
    assert len(calls) == 1  # the second came from the cache

    messages, tier = calls[0]
    assert tier == "vision"
    text_part, image_part = messages[1]["content"]
    assert "OPERATING REVENUES ($ millions)" in text_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    describe(_candidate(), b"other-image", cache, fake_structured)
    assert len(calls) == 2  # a different image is a different cache entry


@pytest.mark.parametrize("enabled", [False, True])
def test_retrieval_reads_figure_chunks_only_when_enabled(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from reportlens.core.config import get_settings
    from reportlens.retrieval.filters import filter_conditions

    monkeypatch.setenv("RETRIEVAL_FIGURES", str(enabled).lower())
    get_settings.cache_clear()
    conditions, _ = filter_conditions(None)
    assert ("kind = 'text'" in conditions) is not enabled
