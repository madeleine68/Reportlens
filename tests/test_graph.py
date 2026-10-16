from pathlib import Path

from pydantic import BaseModel

from reportlens.ingest.graph import (
    Executive,
    ExecutiveFact,
    ExecutiveList,
    FactCache,
    SubsidiaryFact,
    SubsidiaryList,
    executive_passages,
    executives_for,
    page_naming,
    subsidiaries_for,
    subsidiary_table,
    table_batches,
)
from reportlens.ingest.graph_load import display_names
from reportlens.ingest.graph_names import (
    clean_name,
    clean_title,
    is_board_only_title,
    name_key,
    person_key,
)
from reportlens.ingest.models import ParsedBlock, ParsedDocument

RBC_TABLE = """| Principal subsidiaries (1) | Principal office address (2) | Carrying value |
| --- | --- | --- |
| RBC US Group Holdings LLC (2) RBC USA Holdco Corporation City National Bank | Toronto | 34,826 |
| RBC Europe Limited | London, England | 3,170 |"""


def _doc(blocks: list[ParsedBlock], company: str = "rbc") -> ParsedDocument:
    return ParsedDocument(
        company=company,
        year=2024,
        source_path="x.pdf",
        sha256="0",
        converter_version="t",
        extract_version="t",
        num_pages=300,
        blocks=blocks,
    )


def _block(text: str, page: int = 1, section: str | None = None, kind: str = "text") -> ParsedBlock:
    return ParsedBlock(text=text, page=page, section=section, block_type=kind)


class FakeStructured:
    """Stands in for llm.structured: returns fixed results and counts calls."""

    def __init__(self, result: BaseModel) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, messages: list, schema: type, **_: object) -> BaseModel:
        self.calls += 1
        return self.result


def test_clean_name_drops_footnotes_and_reads_the_share_printed_in_the_name() -> None:
    assert clean_name("RBC US Group Holdings LLC (2)") == ("RBC US Group Holdings LLC", None)
    assert clean_name("Scotiabank Chile S.A. (99.79%)") == ("Scotiabank Chile S.A.", 99.79)
    assert clean_name("Scotiabank& Trust (Cayman) Ltd.") == (
        "Scotiabank & Trust (Cayman) Ltd.",
        None,
    )
    assert clean_name("Scotia Capital (USA) Inc.")[0] == "Scotia Capital (USA) Inc."


def test_name_keys_match_spellings_of_the_same_company_or_person() -> None:
    assert name_key("The Royal Trust Company") == name_key("Royal Trust Company")
    assert name_key("Scotiabank & Trust (Cayman) Ltd.") == name_key(
        "Scotiabank and Trust (Cayman) Ltd"
    )
    assert person_key("David I. McKay") == person_key("David McKay")
    assert person_key("Devorah lithwick") == person_key("Devorah Lithwick")


def test_clean_title_drops_the_company_it_is_at() -> None:
    title = "Executive Vice President and Chief Financial Officer BCE Inc. and Bell Canada"
    assert clean_title(title) == "Executive Vice President and Chief Financial Officer"
    assert clean_title("President and Chief Executive Officer of Scotiabank") == (
        "President and Chief Executive Officer"
    )
    assert clean_title("Group Head, Canadian Banking") == "Group Head, Canadian Banking"


def test_board_only_titles() -> None:
    assert is_board_only_title("Director")
    assert is_board_only_title("Chairman of the Board")
    assert not is_board_only_title("Group Head and Chief Financial Officer")


def test_subsidiary_table_skips_other_tables_that_mention_subsidiaries() -> None:
    doc = _doc(
        [
            _block("| Bank foreign subsidiaries | 41,383 |", page=101, kind="table"),
            _block(
                "| Chile | 790 |", 222, "(b) Non-controlling interests in subsidiaries", "table"
            ),
            _block(RBC_TABLE, page=247, section="Note 32 Principal subsidiaries", kind="table"),
        ]
    )
    table = subsidiary_table(doc)
    assert table is not None and table.page == 247


def test_table_batches_repeat_the_header() -> None:
    batches = table_batches(RBC_TABLE, rows=1)
    assert len(batches) == 2
    assert all(b.startswith("| Principal subsidiaries") for b in batches)
    assert batches[1].endswith("| RBC Europe Limited | London, England | 3,170 |")


def test_executive_passages_keep_short_titled_blocks_with_their_heading() -> None:
    doc = _doc(
        [
            _block("(signed) Katherine Gibson Chief Financial Officer", page=142),
            _block("(signed) Katherine Gibson Chief Financial Officer", page=150),  # duplicate
            _block("Executive Vice President and Chief Financial Officer", 169, "Curtis Millen"),
            _block("The Chief Risk Officer oversees risk. " * 20, page=80),  # prose, too long
            _block("Net income rose 11%.", page=5),
        ]
    )
    assert executive_passages(doc) == [
        (142, "(signed) Katherine Gibson Chief Financial Officer"),
        (169, "Curtis Millen: Executive Vice President and Chief Financial Officer"),
    ]


def test_subsidiaries_for_cleans_dedupes_and_cites_the_table_page(tmp_path: Path) -> None:
    doc = _doc([_block(RBC_TABLE, 247, "Note 32 Principal subsidiaries", "table")])
    found = SubsidiaryList(
        subsidiaries=[
            SubsidiaryFact(name="City National Bank", location="Los Angeles"),
            SubsidiaryFact(name="City National Bank (2)"),  # same company again
            SubsidiaryFact(name="Scotiabank Chile S.A. (99.79%)"),
            SubsidiaryFact(name="Scotia"),  # fragment of a name the PDF split
            SubsidiaryFact(name="Royal Bank of Canada"),  # the parent itself
        ]
    )
    subsidiaries = subsidiaries_for(doc, FactCache(tmp_path / "c.json"), FakeStructured(found))
    assert [(s.name, s.ownership_pct, s.page) for s in subsidiaries] == [
        ("City National Bank", None, 247),
        ("Scotiabank Chile S.A.", 99.79, 247),
    ]


def test_executives_for_drops_directors_and_people_not_in_the_passages(tmp_path: Path) -> None:
    doc = _doc(
        [
            _block("(signed) David I. McKay President and Chief Executive Officer", page=142),
            _block("David I. McKay Frank Vettese Director", page=148),
        ]
    )
    found = ExecutiveList(
        executives=[
            ExecutiveFact(name="David I. McKay", title="President and CEO", role="ceo", page=142),
            ExecutiveFact(name="Frank Vettese", title="Director", role="other", page=148),
            ExecutiveFact(name="Jane Madeup", title="Chief Financial Officer", role="cfo", page=1),
        ]
    )
    executives = executives_for(doc, FactCache(tmp_path / "c.json"), FakeStructured(found))
    assert [(e.name, e.role, e.page) for e in executives] == [("David I. McKay", "ceo", 142)]


def test_page_naming_falls_back_to_a_passage_that_names_the_person() -> None:
    passages = [(112, "(signed) Curtis Millen EVP and CFO"), (169, "Curtis Millen: CFO")]
    fact = ExecutiveFact(name="Curtis Millen", title="CFO", role="cfo", page=21)
    assert page_naming(fact, passages) == 112
    assert page_naming(fact.model_copy(update={"page": 169}), passages) == 169
    assert page_naming(fact.model_copy(update={"name": "Someone Else"}), passages) is None


def test_cached_extraction_makes_no_second_llm_call(tmp_path: Path) -> None:
    doc = _doc([_block(RBC_TABLE, 247, "Principal subsidiaries", "table")])
    fake = FakeStructured(SubsidiaryList(subsidiaries=[SubsidiaryFact(name="RBC Europe Limited")]))
    path = tmp_path / "c.json"
    first = subsidiaries_for(doc, FactCache(path), fake)
    second = subsidiaries_for(doc, FactCache(path), fake)  # a fresh cache reads the file
    assert fake.calls == 1
    assert first == second


def test_display_names_prefer_the_most_common_spelling() -> None:
    executives = [
        Executive(
            company="rbc",
            report_year=year,
            name=name,
            key=person_key(name),
            title="President and Chief Executive Officer",
            role="ceo",
            page=1,
        )
        for year, name in [
            (2021, "David McKay"),
            (2022, "David I. McKay"),
            (2023, "David I. McKay"),
        ]
    ]
    assert display_names(executives) == {"david mckay": "David I. McKay"}
