"""Cleaning and matching the names of subsidiaries and executives (Phase 3 knowledge graph).

The same company or person is printed differently across reports ("Scotiabank& Trust",
"David I. McKay" vs "David McKay"); match keys make each one a single graph node.
"""

import re

from reportlens.ingest.tables import clean_label

# A title often ends with the company it is at ("... Officer, BCE Inc. and Bell Canada").
_TITLE_COMPANY_SUFFIX = re.compile(
    r"[,\s]*(?:of\s+)?(?:BCE Inc\.(?: and Bell Canada)?|Bell Canada|Scotiabank|"
    r"Royal Bank of Canada|RBC)\s*$"
)
_BOARD_ONLY_TITLE = re.compile(r"^(?:director|chair(?:man)?(?: of the board)?)$", re.I)
_PCT_IN_NAME = re.compile(r"\s*\((\d{1,3}(?:\.\d+)?)\s*%\)")
_FOOTNOTE_IN_NAME = re.compile(r"\s*\((?:\d{1,2}|[a-z])\)")


def clean_name(name: str) -> tuple[str, float | None]:
    """(name without footnotes or an ownership share, the share if it was printed in the name)."""
    pct = _PCT_IN_NAME.search(name)
    name = _PCT_IN_NAME.sub("", name)
    name = _FOOTNOTE_IN_NAME.sub("", clean_label(name))
    name = re.sub(r"\s*&\s*", " & ", name)
    return " ".join(name.split()).strip(" ,;"), float(pct.group(1)) if pct else None


def clean_title(title: str) -> str:
    """The title on one line, without the company name it sometimes ends with."""
    return _TITLE_COMPANY_SUFFIX.sub("", " ".join(title.split())).strip(" ,")


def is_board_only_title(title: str) -> bool:
    """True for 'Director' or 'Chair': a board member signing the statements, not an officer."""
    return bool(_BOARD_ONLY_TITLE.match(title))


def name_key(name: str) -> str:
    """Match key for a company name: case, punctuation, '&' and a leading 'The' don't matter."""
    key = name.casefold().replace("&", " and ")
    key = re.sub(r"[.,'’]", "", key)
    key = re.sub(r"^the\s+", "", " ".join(key.split()))
    return key


def person_key(name: str) -> str:
    """Match key for a person: as name_key, ignoring middle initials."""
    return " ".join(t for t in name_key(name).split() if len(t) > 1)
