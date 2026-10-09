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
    # Retries on rate limits / transient errors (the SDK waits between tries). Free tiers need more.
    llm_max_retries: int = 6
    # Main-model prices in USD per million tokens, for cost/query in evals. 0 = free tier.
    llm_price_input_per_mtok: float = 0.0
    llm_price_output_per_mtok: float = 0.0

    # Embeddings and chunking. bge-small reads at most 512 tokens, so chunks stay well under
    # that to leave room for the Phase 2 context prefix.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # bge models expect this instruction before queries (not passages); "" for models that don't.
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    chunk_tokens: int = 450
    chunk_overlap: int = 50

    # Retrieval
    retrieval_top_k: int = 5
    # Phase 2: LLM-extracted company/year filters (off = Phase 1 baseline behaviour).
    retrieval_use_filters: bool = False
    # Which retriever answers use: "vector" (Phase 1), "keyword" (Postgres full-text), or
    # "hybrid" (both, fused with Reciprocal Rank Fusion).
    retrieval_mode: Literal["vector", "keyword", "hybrid"] = "vector"
    hybrid_candidates: int = 20  # results taken from each retriever before fusing
    rrf_k: int = 60  # RRF damping constant; 60 is the value from the original paper

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"

    log_level: str = "INFO"

    @property
    def raw_dir(self) -> Path:
        """Directory holding raw PDFs: data/raw/{company}/{year}.pdf."""
        return self.data_dir / "raw"

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


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide Settings instance (read once, then cached)."""
    return Settings()
