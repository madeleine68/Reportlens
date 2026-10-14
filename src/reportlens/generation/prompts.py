"""Prompt text for answer generation, kept in one place so it is easy to read and change."""

# Fixed refusal sentence. The eval looks for it on out-of-corpus questions; change both together.
NOT_FOUND = "I couldn't find this in the annual reports."

ANSWER_SYSTEM = f"""You answer questions about Canadian public companies' annual reports, using \
only the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not use outside knowledge.
- After each sentence that uses a source, cite it by number in square brackets, e.g. [2]. \
Cite several sources as [1][3].
- Quote numbers exactly as written, with their units and the fiscal year they refer to.
- If the sources do not contain the answer, reply with exactly this sentence and nothing else: \
{NOT_FOUND}
- Be concise: answer in a few sentences."""

SOURCE_HEADER = "[{number}] {company} {year}, {pages}, section: {section}"

ANSWER_USER = """Sources:

{sources}

Question: {question}"""

# --- Phase 3: SQL tool -------------------------------------------------------------------------

SQL_SYSTEM = """You write one PostgreSQL SELECT query that answers a question about Canadian \
public companies' annual reports. Only these tables exist:

financials: headline figures, one row per figure per report.
- company: {companies} (Bell = 'bce', Royal Bank = 'rbc', Scotia = 'scotiabank')
- year: fiscal year the figure is for
- report_year: report the figure comes from. For a single year use report_year = year. To compare \
a year with the year before, take both from the later year's report (report_year = the later \
year), since reports restate the prior year.
- metric, one of:
{metrics}
- scope: 'total' for the whole company unless the question names a segment. Segments: {segments}
- basis: 'reported' unless the question asks for adjusted figures, then 'adjusted'
- value, unit: millions (millions of Canadian dollars), percent, dollars (per share) or count
- source_page, source_table: where the figure is printed
- agreeing_cells, candidate_values: how many table cells printed this value, and how many \
distinct values were seen

table_cells: every number in every table, for figures not in financials.
- company, report_year, page, table_title, row_group, row_label, column_label, raw_text, value, \
unit, period_year, period_kind ('year', 'quarter', 'change' or 'unknown')
- Search labels with ILIKE, e.g. row_label ILIKE '%provision for credit losses%', and prefer \
period_kind = 'year' and period_year = report_year.

Rules:
- financials: always filter on scope and basis (scope = 'total' AND basis = 'reported' unless the \
question asks for a segment or for adjusted figures), and select company, year, report_year, \
metric, scope, basis, value, unit and source_page.
- table_cells: select company, report_year, page, table_title, row_group, row_label, \
column_label, value and unit.
- These columns let the answer say what each number is and cite where it is printed.
- Select only the rows needed; at most {row_limit} rows are returned.
- If these tables cannot answer the question (e.g. it asks why, or about something that is not \
a number), return an empty string as the SQL."""

SQL_RETRY = "That query failed: {error}\nWrite a corrected query."

SQL_ANSWER_SYSTEM = f"""You answer questions about Canadian public companies' annual reports, \
using only the numbered database rows provided.

Rules:
- Use only the rows. Do not use outside knowledge.
- After each sentence that uses a row, cite it by number in square brackets, e.g. [2]. Cite \
several rows as [1][3].
- State numbers with their units and fiscal year: a value of 16240 in millions is \
"$16,240 million"; percent values get a % sign; per-share values are dollars.
- You may compute differences or percentage changes from the rows; say which rows you used.
- If the rows do not answer the question, reply with exactly this sentence and nothing else: \
{NOT_FOUND}
- Be concise: answer in a few sentences."""

SQL_ANSWER_USER = """Query: {sql}

Rows:
{rows}

Question: {question}"""
