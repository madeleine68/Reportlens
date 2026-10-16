"""Check LLM-written Cypher before it runs: one read-only query on our labels, with a row limit.

This is the first of two layers. Neo4j Community has no read-only users, so the second is the
server itself: the query runs in a read transaction, which refuses writes, with a timeout
(sources/graph_tool.py). This check also stops the reads a read transaction allows but we don't
want: procedures (CALL), LOAD CSV (reads files and URLs), and switching database (USE).

There is no maintained Cypher parser for Python, so the check works on the query's text after
string literals and comments are blanked out, so a name such as 'Set Inc.' can't trip it.
"""

import re

LABELS = ("Company", "Segment", "Subsidiary", "Executive")
RELATIONSHIPS = ("HAS_SEGMENT", "OWNS", "HOLDS_ROLE")

_BLOCKED_KEYWORDS = (
    # writes and schema changes (the read transaction refuses these too)
    "CREATE", "MERGE", "SET", "DELETE", "DETACH", "REMOVE", "DROP", "FOREACH", "ALTER", "RENAME",
    # reads we don't allow: procedures, files and URLs, other databases, server admin
    "CALL", "LOAD", "USE", "SHOW", "GRANT", "DENY", "REVOKE", "START", "STOP", "TERMINATE",
)  # fmt: skip
_BLOCKED_NAMESPACES = ("apoc", "dbms", "db", "gds", "genai")  # functions outside plain Cypher

# Strings and comments in one pattern, scanned left to right, so '//' inside a string stays text.
_STRING_OR_COMMENT = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|//[^\n]*|/\*.*?\*/", re.S)
_KEYWORD = re.compile(rf"(?<![.\w$`])({'|'.join(_BLOCKED_KEYWORDS)})\b", re.I)
_NAMESPACE = re.compile(rf"(?<![.\w$`])({'|'.join(_BLOCKED_NAMESPACES)})\s*\.", re.I)
_MAP_KEY = re.compile(r"([{,]\s*)`?\w+`?\s*:")  # {id: 'rbc'} is a map key, not a label
_LABELS = re.compile(r":\s*(`?\w+`?(?:\s*[|&]\s*:?\s*`?\w+`?)*)")  # :Company, :OWNS|HAS_SEGMENT
_TRAILING_LIMIT = re.compile(r"\bLIMIT\s+(\S+)\s*$", re.I)


class UnsafeCypher(ValueError):
    """The Cypher was rejected before running; the message says why (it is fed back to the LLM)."""


def code_only(query: str) -> str:
    """The query with comments and string contents blanked out, keeping every character's position.

    Keeping positions lets a match in the cleaned text be applied to the original query.
    """

    def blank(match: re.Match[str]) -> str:
        text = match.group()
        if text[0] in "'\"":
            return text[0] + " " * (len(text) - 2) + text[0]  # keep the quotes: still a value
        return " " * len(text)

    return _STRING_OR_COMMENT.sub(blank, query)


def check_cypher(query: str, row_limit: int) -> str:
    """The query with a LIMIT of at most `row_limit`; raises UnsafeCypher if it is rejected."""
    query = query.strip().rstrip(";").strip()
    code = code_only(query)
    if not code.strip():
        raise UnsafeCypher("the query is empty")
    if ";" in code:
        raise UnsafeCypher("expected exactly one statement")
    if not re.match(r"\s*(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN)\b", code, re.I):
        raise UnsafeCypher(
            "the query must start with MATCH, OPTIONAL MATCH, WITH, UNWIND or RETURN"
        )
    if match := _KEYWORD.search(code):
        raise UnsafeCypher(f"{match.group(1).upper()} is not allowed; write a read-only query")
    if match := _NAMESPACE.search(code):
        raise UnsafeCypher(f"{match.group(1)}.* functions are not allowed")

    allowed = {*LABELS, *RELATIONSHIPS}
    for chain in _LABELS.findall(_MAP_KEY.sub(r"\1", code)):
        for name in re.split(r"[|&:]", chain):
            name = name.strip().strip("`")
            if name and name not in allowed:
                raise UnsafeCypher(
                    f"label or relationship {name!r} does not exist; use only: "
                    f"{', '.join(LABELS)} and {', '.join(RELATIONSHIPS)}"
                )
    return _with_limit(query, code, row_limit)


def _with_limit(query: str, code: str, row_limit: int) -> str:
    """Keep a trailing LIMIT that is a number no larger than row_limit, else set row_limit."""
    match = _TRAILING_LIMIT.search(code)
    if match is None:
        return f"{query}\nLIMIT {row_limit}"
    value = match.group(1)
    if value.isdigit() and int(value) <= row_limit:
        return query
    return f"{query[: match.start()]}LIMIT {row_limit}"
