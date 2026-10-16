import pytest

from reportlens.sources.cypher_safety import UnsafeCypher, check_cypher, code_only

OWNS = "MATCH (c:Company)-[r:OWNS]->(s:Subsidiary) RETURN s.name"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (OWNS, f"{OWNS}\nLIMIT 50"),
        (f"{OWNS} LIMIT 10", f"{OWNS} LIMIT 10"),
        (f"{OWNS} LIMIT 1000", f"{OWNS} LIMIT 50"),
        (f"{OWNS} LIMIT $n", f"{OWNS} LIMIT 50"),
        (f"{OWNS};", f"{OWNS}\nLIMIT 50"),
        (f"{OWNS} // all of them", f"{OWNS} // all of them\nLIMIT 50"),
        (f"{OWNS} LIMIT 5 // top five", f"{OWNS} LIMIT 5 // top five"),
    ],
)
def test_safe_queries_pass_with_a_row_limit(query: str, expected: str) -> None:
    assert check_cypher(query, row_limit=50) == expected


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (c:Company {id: 'rbc'})-[r:OWNS|HAS_SEGMENT]->(x) RETURN x.name",
        "MATCH (e:Executive)-[r:HOLDS_ROLE {report_year: 2024, role: 'cfo'}]->(c) RETURN e.name",
        "MATCH (c:Company) WHERE toLower(c.name) CONTAINS 'set' RETURN c.id",
        "MATCH (s:Subsidiary) WHERE s.name STARTS WITH 'Scotia' RETURN s.name",
        "MATCH (c:Company)-[r:OWNS]->(s) RETURN c {.id, total: count(s)}",
        "MATCH (s:Subsidiary) WHERE s.name = 'Delete: Create; Set Ltd. //' RETURN s",
        "MATCH (c:Company) WHERE EXISTS { MATCH (c)-[:OWNS]->(:Subsidiary) } RETURN c.id",
        "WITH ['rbc', 'bce'] AS ids UNWIND ids AS id MATCH (c:Company {id: id}) RETURN c.name",
        "OPTIONAL MATCH (c:Company) RETURN [x IN collect(c.id) | toUpper(x)]",
    ],
)
def test_read_only_queries_on_our_labels_pass(query: str) -> None:
    check_cypher(query, row_limit=50)


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("", "empty"),
        ("// only a comment", "empty"),
        (f"{OWNS}; MATCH (n) DETACH DELETE n", "exactly one statement"),
        ("CREATE (c:Company {id: 'x'})", "must start with"),
        ("SHOW USERS", "must start with"),
        ("MATCH (c:Company) SET c.name = 'x' RETURN c", "SET is not allowed"),
        ("MATCH (c:Company) DETACH DELETE c", "DETACH is not allowed"),
        ("MATCH (c:Company) MERGE (c)-[:OWNS]->(:Subsidiary {name: 'x'})", "MERGE"),
        ("MATCH (c:Company) REMOVE c.name RETURN c", "REMOVE"),
        ("MATCH (c:Company) FOREACH (x IN [1] | CREATE (:Company))", "FOREACH"),
        ("MATCH (c:Company) CALL db.labels() YIELD label RETURN label", "CALL"),
        ("MATCH (c:Company) CALL { MATCH (n) RETURN n } RETURN c", "CALL"),
        ("WITH 1 AS x LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row", "LOAD"),
        ("MATCH (c:Company) RETURN apoc.text.join([c.id], ',')", "apoc.* functions"),
        ("MATCH (c:Company) RETURN dbms.components()", "dbms.* functions"),
        ("MATCH (u:User) RETURN u", "'User' does not exist"),
        ("MATCH (c:Company)-[:EMPLOYS]->(p) RETURN p", "'EMPLOYS' does not exist"),
        ("MATCH (c:Company)-[:OWNS|ACQUIRED]->(s) RETURN s", "'ACQUIRED' does not exist"),
        ("MATCH (n) WHERE n:Person RETURN n", "'Person' does not exist"),
    ],
)
def test_writes_procedures_and_unknown_labels_are_rejected(query: str, reason: str) -> None:
    with pytest.raises(UnsafeCypher, match=reason):
        check_cypher(query, row_limit=50)


def test_code_only_blanks_strings_and_comments_but_keeps_positions() -> None:
    query = "MATCH (s) WHERE s.name = 'http://x; SET' // CALL\nRETURN s"
    code = code_only(query)
    assert len(code) == len(query)
    assert "SET" not in code and "CALL" not in code and ";" not in code
    assert code.index("RETURN") == query.index("RETURN")
