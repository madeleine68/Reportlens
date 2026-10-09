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
