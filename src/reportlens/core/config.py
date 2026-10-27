"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote, urlsplit, urlunsplit

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All runtime configuration. Field names map to upper-case env vars."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Postgres
    postgres_user: str = "reportlens"
    postgres_password: SecretStr = SecretStr("reportlens")
    postgres_db: str = "reportlens"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    # Hosted databases (deployment): a full connection string replaces the parts above, e.g.
    # Neon's postgresql://user:pass@host/db?sslmode=require. The SQL tool's read-only URL is the
    # same with the reader's name and password swapped in.
    postgres_url: SecretStr | None = Field(default=None, validation_alias="DATABASE_URL")

    # LLM (any OpenAI-compatible endpoint)
    llm_base_url: str
    llm_api_key: SecretStr
    llm_model_main: str
    llm_model_fast: str
    llm_model_vision: str = "qwen/qwen3.8-27b"  # Phase 3 chart summaries; must accept images
    # Retries on rate limits / transient errors (the SDK waits between tries). Free tiers need more.
    llm_max_retries: int = 6
    # Main-model prices in USD per million tokens, for cost/query in evals. 0 = free tier.
    llm_price_input_per_mtok: float = 0.0
    llm_price_output_per_mtok: float = 0.0

    # Parsing: run Docling's table-structure model (TableFormer) so tables keep their rows and
    # columns. Slow on CPU (~4.6 s per table); table parses get their own cache files.
    parse_tables: bool = False

    # Embeddings and chunking. bge-small reads at most 512 tokens, so chunks stay well under
    # that to leave room for the Phase 2 context prefix.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge models expect this instruction before queries (not passages); "" for models that don't.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    chunk_tokens: int = 450
    chunk_overlap: int = 50
    # "naive" (Phase 1 fixed windows), "contextual" (Phase 2 section-aware, with a label) or
    # "tables" (Phase 3: naive windows for text, one chunk per table with its header row kept).
    # Keep each strategy in its own database (POSTGRES_DB) to compare them without re-embedding.
    chunking: Literal["naive", "contextual", "tables"] = "naive"
    chunk_min_tokens: int = 100  # contextual: a heading starts a new chunk only after this many
    context_reserve_tokens: int = 40  # contextual: tokens kept free for the label

    # Retrieval
    retrieval_top_k: int = 5
    # Phase 2: LLM-extracted company/year filters (off = Phase 1 baseline behaviour).
    retrieval_use_filters: bool = False
    # Which retriever answers use: "vector" (Phase 1), "keyword" (Postgres full-text), or
    # "hybrid" (both, fused with Reciprocal Rank Fusion).
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    hybrid_candidates: int = 20  # results taken from each retriever before fusing
    rrf_k: int = 60  # RRF damping constant; 60 is the value from the original paper
    # Rerank: fetch this many candidates, rescore them with a cross-encoder, keep retrieval_top_k.
    retrieval_rerank: bool = False
    # Phase 3: also search the chart summaries (chunks with kind = 'figure').
    retrieval_figures: bool = False
    # Phase 3: instead, reserve this many of the top k for the best chart summaries, searched
    # on their own so charts never push text out of the other slots (0 = off).
    retrieval_figure_slots: int = 0
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 40

    # Phase 3 SQL tool: LLM-written SQL runs as this read-only role, which core/db.py creates
    # with SELECT on the approved tables only. Every query gets a timeout and a row limit.
    sql_reader_user: str = "reportlens_reader"
    sql_reader_password: SecretStr = SecretStr("reportlens_reader")
    sql_timeout_ms: int = 5000
    sql_row_limit: int = 50

    # Phase 3 graph: Neo4j holds companies, segments, subsidiaries and executives.
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("reportlens")
    neo4j_host: str = "localhost"
    neo4j_bolt_port: int = 7687
    # Hosted Neo4j (deployment), e.g. AuraDB's neo4j+s://xxxx.databases.neo4j.io
    neo4j_url: str | None = Field(default=None, validation_alias="NEO4J_URI")
    # LLM-written Cypher runs in a read transaction with this timeout and row limit.
    cypher_timeout_s: float = 5.0
    cypher_row_limit: int = 50

    # Phase 4 web search (Tavily): off by default, so out-of-corpus questions are refused and the
    # evals stay comparable. With it on, the router's 'web' route searches the web.
    web_search: bool = False
    tavily_api_key: SecretStr | None = None
    web_max_results: int = 5
    web_timeout_s: float = 15.0

    # Phase 5 public demo: with DEMO_PASSWORD set, /ask needs it (header X-Demo-Password).
    # Limits protect the free-tier LLM quota; 0 = no limit (local use).
    demo_password: SecretStr | None = None
    rate_limit_per_hour: int = 0  # questions per visitor per rolling hour
    daily_question_cap: int = 0  # questions per day for everyone together

    # Phase 4 corrective RAG: the fast model grades retrieved chunks; if they are not enough,
    # it rewrites the query (at most this many times), then falls back to the web or refuses.
    crag_max_rewrites: int = 1
    # LangGraph's step limit for one question (the longest normal run takes 6 steps).
    agent_max_steps: int = 12
    # The router's text-search route (and its fallback) runs corrective RAG; false: plain search.
    routed_corrective: bool = True
    # Phase 4 multi-source answers: questions routed to SQL or the graph are answered from their
    # rows plus the top text chunks, in one main-LLM call (agents/combined.py). Off until measured.
    multi_source: bool = False
    # Phase 4 Self-RAG: the fast model checks each routed answer against its sources; if claims
    # are unsupported, the main model revises it (at most this many times). Off until measured.
    self_rag: bool = False
    self_rag_max_revisions: int = 1

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"

    log_level: str = "INFO"

    @property
    def raw_dir(self) -> Path:
        """Directory holding raw PDFs: data/raw/{company}/{year}.pdf."""
        return self.data_dir / "raw"

    @property
    def filter_cache_path(self) -> Path:
        """Cache of LLM-extracted metadata filters, keyed by model, prompt and question."""
        return self.data_dir / "cache" / "filters.json"

    @property
    def graph_cache_path(self) -> Path:
        """Cache of LLM-extracted graph facts, keyed by model, prompt and source text."""
        return self.data_dir / "cache" / "graph_facts.json"

    @property
    def route_cache_path(self) -> Path:
        """Cache of router decisions, keyed by model, prompt and question."""
        return self.data_dir / "cache" / "routes.json"

    @property
    def crag_cache_path(self) -> Path:
        """Cache of corrective RAG grades and rewrites, keyed by model, prompt and input."""
        return self.data_dir / "cache" / "corrective.json"

    @property
    def self_rag_cache_path(self) -> Path:
        """Cache of Self-RAG support checks, keyed by model, prompt and answer."""
        return self.data_dir / "cache" / "self_rag.json"

    @property
    def figure_cache_path(self) -> Path:
        """Cache of vision-model chart summaries, keyed by model, prompt and image."""
        return self.data_dir / "cache" / "figures.json"

    @property
    def neo4j_uri(self) -> str:
        """Address of the Neo4j server: NEO4J_URI if set, else the local Bolt port."""
        return self.neo4j_url or f"bolt://{self.neo4j_host}:{self.neo4j_bolt_port}"

    @property
    def parsed_dir(self) -> Path:
        """Directory holding cached parser output."""
        return self.data_dir / "parsed"

    @property
    def database_url(self) -> str:
        """Postgres connection string for psycopg: DATABASE_URL if set, else built from parts."""
        if self.postgres_url is not None:
            return self.postgres_url.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def reader_database_url(self) -> str:
        """Connection string for the read-only role the SQL tool uses: the main one, with the
        reader's name and password in place of the owner's."""
        url = urlsplit(self.database_url)
        user = quote(self.sql_reader_user, safe="")
        password = quote(self.sql_reader_password.get_secret_value(), safe="")
        host = url.hostname or ""
        netloc = f"{user}:{password}@{host}" + (f":{url.port}" if url.port else "")
        return urlunsplit(url._replace(netloc=netloc))


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance (read once, then cached)."""
    return Settings()
