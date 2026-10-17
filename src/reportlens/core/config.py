"""Application settings loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
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
    # LLM-written Cypher runs in a read transaction with this timeout and row limit.
    cypher_timeout_s: float = 5.0
    cypher_row_limit: int = 50

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
    def figure_cache_path(self) -> Path:
        """Cache of vision-model chart summaries, keyed by model, prompt and image."""
        return self.data_dir / "cache" / "figures.json"

    @property
    def neo4j_uri(self) -> str:
        """Bolt address of the Neo4j server."""
        return f"bolt://{self.neo4j_host}:{self.neo4j_bolt_port}"

    @property
    def parsed_dir(self) -> Path:
        """Directory holding cached parser output."""
        return self.data_dir / "parsed"

    @property
    def database_url(self) -> str:
        """Postgres connection string for psycopg."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def reader_database_url(self) -> str:
        """Connection string for the read-only role the SQL tool uses."""
        return (
            f"postgresql://{self.sql_reader_user}:{self.sql_reader_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance (read once, then cached)."""
    return Settings()
