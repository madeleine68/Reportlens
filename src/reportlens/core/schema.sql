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
