-- ReportLens schema. Applied on every connect, so every statement must be idempotent.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per loaded report. ingest_config fingerprints everything that shapes its chunks;
-- if the PDF or any setting changes, the report's chunks are rebuilt.
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,           -- e.g. 'rbc-2023'
    company       TEXT NOT NULL,
    year          INTEGER NOT NULL,
    source_path   TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    ingest_config TEXT NOT NULL,
    loaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- company/year/section are copied onto each chunk so metadata filters are plain WHERE clauses.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,             -- e.g. 'rbc-2023-00042'
    document_id TEXT NOT NULL REFERENCES documents ON DELETE CASCADE,
    company     TEXT NOT NULL,
    year        INTEGER NOT NULL,
    section     TEXT,
    pages       INTEGER[] NOT NULL,
    text        TEXT NOT NULL,
    num_tokens  INTEGER NOT NULL,
    embedding   vector(384) NOT NULL          -- bge-small-en-v1.5 dimension
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_company_year ON chunks (company, year);

-- Phase 2 keyword search: Postgres full-text index over chunk text (stemmed, stopwords removed).
-- A generated column fills itself for existing and new rows, so no re-ingest is needed.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Phase 2 contextual chunking: the label embedded with each chunk ('' for naive chunks).
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context TEXT NOT NULL DEFAULT '';

-- Phase 3 charts: 'text' for chunks of the report's text, 'figure' for a chart's summary
-- (scripts/describe_figures.py). Retrieval reads figures only when RETRIEVAL_FIGURES=true.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS kind TEXT NOT NULL DEFAULT 'text';

-- Phase 3: every numeric cell of every parsed table, with its row and column labels.
-- Not tied to `documents` (which tracks chunk loads): scripts/load_tables.py replaces a report's
-- cells on every run. The SQL tool reads this table, so it holds no text beyond the labels.
CREATE TABLE IF NOT EXISTS table_cells (
    company      TEXT NOT NULL,
    report_year  INTEGER NOT NULL,
    table_id     TEXT NOT NULL,               -- e.g. 'rbc-2023-t0042'
    table_title  TEXT NOT NULL,
    page         INTEGER NOT NULL,
    row_index    INTEGER NOT NULL,
    row_label    TEXT NOT NULL,
    row_group    TEXT NOT NULL,               -- latest label-only row above, e.g. 'Adjusted Results'
    column_index INTEGER NOT NULL,
    column_label TEXT NOT NULL,
    raw_text     TEXT NOT NULL,
    value        DOUBLE PRECISION NOT NULL,
    unit         TEXT,                        -- millions | billions | thousands | percent | bps
    period_year  INTEGER,
    period_kind  TEXT NOT NULL,               -- year | quarter | change | unknown
    PRIMARY KEY (table_id, row_index, column_index)
);

CREATE INDEX IF NOT EXISTS table_cells_company_year ON table_cells (company, report_year);

-- Phase 3: a few headline metrics picked from table_cells by ingest/financials.py.
-- One row per figure per report, since a later report may restate the prior year.
CREATE TABLE IF NOT EXISTS financials (
    company          TEXT NOT NULL,
    year             INTEGER NOT NULL,        -- fiscal year the figure is for
    report_year      INTEGER NOT NULL,        -- report it was read from (year or year + 1)
    metric           TEXT NOT NULL,           -- e.g. 'net_income', 'cet1_ratio'
    scope            TEXT NOT NULL,           -- 'total' or a segment name
    basis            TEXT NOT NULL,           -- 'reported' or 'adjusted'
    value            DOUBLE PRECISION NOT NULL,
    unit             TEXT NOT NULL,           -- millions | percent | dollars | count
    source_page      INTEGER NOT NULL,
    source_table     TEXT NOT NULL,
    agreeing_cells   INTEGER NOT NULL,        -- cells that printed this value
    candidate_values INTEGER NOT NULL,        -- distinct values seen (> 1: tables disagreed)
    PRIMARY KEY (company, year, report_year, metric, scope, basis)
);
