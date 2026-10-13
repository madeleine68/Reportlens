from pathlib import Path
from types import SimpleNamespace

import pytest
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableData,
)

from reportlens.core.config import get_settings
from reportlens.ingest import parse


def _prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))


def _sample_doc() -> DoclingDocument:
    doc = DoclingDocument(name="sample")
    for page in (1, 2):
        doc.add_page(page_no=page, size=Size(width=612, height=792))
    doc.add_title("Annual Report 2023", prov=_prov(1))
    doc.add_text(
        DocItemLabel.PAGE_HEADER, "RBC | 2023", prov=_prov(1), content_layer=ContentLayer.FURNITURE
    )
    doc.add_text(DocItemLabel.TEXT, "Intro paragraph.", prov=_prov(1))
    doc.add_heading("Net income", prov=_prov(2))
    doc.add_text(DocItemLabel.TEXT, "Net income was $14.9 billion.", prov=_prov(2))
    doc.add_text(DocItemLabel.TEXT, "   ", prov=_prov(2))
    doc.add_list_item("Strong capital ratios", prov=_prov(2), parent=doc.add_list_group())
    doc.add_text(
        DocItemLabel.PAGE_FOOTER, "Page 2", prov=_prov(2), content_layer=ContentLayer.FURNITURE
    )
    return doc


def test_blocks_keep_page_numbers_and_sections() -> None:
    blocks = parse.blocks_from_docling(_sample_doc())
    assert [(b.text, b.page, b.section, b.block_type) for b in blocks] == [
        ("Annual Report 2023", 1, "Annual Report 2023", "heading"),
        ("Intro paragraph.", 1, "Annual Report 2023", "text"),
        ("Net income", 2, "Net income", "heading"),
        ("Net income was $14.9 billion.", 2, "Net income", "text"),
        ("Strong capital ratios", 2, "Net income", "list_item"),
    ]


def test_blocks_drop_headers_footers_and_blank_text() -> None:
    texts = [b.text for b in parse.blocks_from_docling(_sample_doc())]
    assert "RBC | 2023" not in texts
    assert "Page 2" not in texts
    assert all(t.strip() for t in texts)


def test_company_year_from_path() -> None:
    assert parse.company_year_from_path(Path("data/raw/rbc/2023.pdf")) == ("rbc", 2023)


@pytest.mark.parametrize("bad", ["data/raw/rbc/annual-2023.pdf", "data/raw/rbc/23.pdf"])
def test_company_year_rejects_bad_names(bad: str) -> None:
    with pytest.raises(ValueError):
        parse.company_year_from_path(Path(bad))


def test_text_inside_table_is_folded_into_the_table_block() -> None:
    doc = DoclingDocument(name="tables")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    table_box = BoundingBox(l=0, t=0, r=300, b=200)
    heading_box = BoundingBox(l=10, t=250, r=200, b=270)  # outside the table area
    doc.add_heading(
        "Strategic priorities", prov=ProvenanceItem(page_no=1, bbox=heading_box, charspan=(0, 1))
    )
    doc.add_table(
        data=TableData(num_rows=1, num_cols=1),
        prov=ProvenanceItem(page_no=1, bbox=table_box, charspan=(0, 0)),
    )
    for i, line in enumerate(["Priority", "Grow in Canada", "2023 progress"]):
        box = BoundingBox(l=10, t=10 + 20 * i, r=200, b=25 + 20 * i)
        doc.add_text(
            DocItemLabel.TEXT, line, prov=ProvenanceItem(page_no=1, bbox=box, charspan=(0, 1))
        )
    outside = BoundingBox(l=10, t=400, r=200, b=420)
    doc.add_text(
        DocItemLabel.TEXT,
        "Prose below the table.",
        prov=ProvenanceItem(page_no=1, bbox=outside, charspan=(0, 1)),
    )

    blocks = parse.blocks_from_docling(doc)

    assert [(b.block_type, b.text) for b in blocks] == [
        ("heading", "Strategic priorities"),
        ("table", "Priority\nGrow in Canada\n2023 progress"),
        ("text", "Prose below the table."),
    ]
    assert all(b.section == "Strategic priorities" for b in blocks)


@pytest.fixture
def raw_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[Path]]:
    """A fake PDF under a temp data dir, with the slow Docling step stubbed and call-counted."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    pdf = tmp_path / "raw" / "rbc" / "2023.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-fake v1")

    calls: list[Path] = []

    def fake_convert(path: Path) -> DoclingDocument:
        calls.append(path)
        return _sample_doc()

    monkeypatch.setattr(parse, "convert_pdf", fake_convert)
    return pdf, calls


def test_parse_cached_writes_then_reuses(raw_pdf: tuple[Path, list[Path]]) -> None:
    pdf, calls = raw_pdf
    first = parse.parse_cached(pdf)
    second = parse.parse_cached(pdf)
    assert len(calls) == 1
    assert first == second
    folder = pdf.parents[2] / "parsed" / "rbc"
    assert (folder / "2023.json").exists()
    assert (folder / "2023.docling.json").exists()


def test_parse_cached_reconverts_changed_pdf(raw_pdf: tuple[Path, list[Path]]) -> None:
    pdf, calls = raw_pdf
    parse.parse_cached(pdf)
    pdf.write_bytes(b"%PDF-fake v2")
    parse.parse_cached(pdf)
    assert len(calls) == 2


def test_parse_cached_reconverts_on_converter_version_bump(
    raw_pdf: tuple[Path, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf, calls = raw_pdf
    parse.parse_cached(pdf)
    monkeypatch.setattr(parse, "CONVERTER_VERSION", "2-test")
    parse.parse_cached(pdf)
    assert len(calls) == 2


def test_parse_cached_reextracts_without_reconverting_on_extract_version_bump(
    raw_pdf: tuple[Path, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf, calls = raw_pdf
    parse.parse_cached(pdf)
    monkeypatch.setattr(parse, "EXTRACT_VERSION", "99-test")
    result = parse.parse_cached(pdf)
    assert len(calls) == 1
    assert result.extract_version == "99-test"
    assert len(result.blocks) == 5


@pytest.mark.slow
def test_parse_pdf_end_to_end(tmp_path: Path) -> None:
    """Runs real Docling on a generated two-page PDF (downloads models on first run)."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf = tmp_path / "raw" / "td" / "2022.pdf"
    pdf.parent.mkdir(parents=True)
    c = canvas.Canvas(str(pdf), pagesize=letter)
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, "Total revenue increased by eight percent compared to last year.")
    c.showPage()
    c.drawString(72, 700, "The board declared a quarterly dividend of one dollar per share.")
    c.save()

    result = parse.to_parsed_document(parse.convert_pdf(pdf), pdf, parse.file_sha256(pdf))

    assert (result.company, result.year, result.num_pages) == ("td", 2022, 2)
    pages = {b.page for b in result.blocks if "revenue" in b.text} | {
        b.page for b in result.blocks if "dividend" in b.text
    }
    assert pages == {1, 2}


def test_compact_markdown_strips_column_padding() -> None:
    padded = (
        "| ($ millions)   | 2023     |\n"
        "|----------------|----------|\n"
        "| Net income     | $ 7,528  |"
    )
    assert parse.compact_markdown(padded) == (
        "| ($ millions) | 2023 |\n| --- | --- |\n| Net income | $ 7,528 |"
    )


def test_table_parses_use_their_own_cache_files_and_version(
    raw_pdf: tuple[Path, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf, calls = raw_pdf
    light = parse.parse_cached(pdf)
    monkeypatch.setenv("PARSE_TABLES", "true")
    get_settings.cache_clear()
    tables = parse.parse_cached(pdf)

    assert len(calls) == 2  # the light cache isn't reused for a table parse...
    assert (light.converter_version, tables.converter_version) == ("1-light", "2-tables")
    folder = pdf.parents[2] / "parsed" / "rbc"
    assert (folder / "2023.tables.json").exists()
    assert (folder / "2023.json").exists()  # ...and isn't overwritten either


def test_page_slices_cover_every_page_once() -> None:
    assert parse.page_slices(45, 20) == [(1, 20), (21, 40), (41, 45)]
    assert parse.page_slices(20, 20) == [(1, 20)]


def _fake_run(outcomes: list[str]):
    """Stub for subprocess.run: each call crashes, hangs or writes a one-page slice."""
    calls: list[list[str]] = []

    def run(cmd: list[str], timeout: float, capture_output: bool):
        calls.append(cmd)
        outcome = outcomes[len(calls) - 1]
        if outcome == "hang":
            raise parse.subprocess.TimeoutExpired(cmd, timeout)
        if outcome == "ok":
            _sample_doc().save_as_json(Path(cmd[-1]))
        return SimpleNamespace(returncode=0 if outcome == "ok" else 139)

    return run, calls


def test_slice_is_retried_after_a_crash_and_a_hang(monkeypatch: pytest.MonkeyPatch) -> None:
    run, calls = _fake_run(["crash", "hang", "ok"])
    monkeypatch.setattr(parse.subprocess, "run", run)

    doc = parse._convert_slice_in_subprocess(Path("r.pdf"), 21, 40)

    assert len(calls) == 3
    assert calls[0][-4:-1] == ["r.pdf", "21", "40"]
    assert doc.num_pages() == 2


def test_slice_gives_up_after_all_attempts_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    run, calls = _fake_run(["crash"] * parse.SLICE_ATTEMPTS)
    monkeypatch.setattr(parse.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="pages 1-20"):
        parse._convert_slice_in_subprocess(Path("r.pdf"), 1, 20)
    assert len(calls) == parse.SLICE_ATTEMPTS
