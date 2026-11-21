"""Typed application configuration.

All settings are read from environment variables (and an optional ``.env``
file for local development). Nothing sensitive has a production default.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["sentence_transformers", "openai", "fastembed"]
LLMProvider = Literal["ollama", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "DocuMind"
    environment: Literal["development", "test", "staging", "production"] = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = Field(default=60, ge=5, le=60 * 24 * 30)
    jwt_algorithm: str = "HS256"

    # ------------------------------------------------------------- database
    database_url: str = "postgresql+psycopg://documind:documind@localhost:5432/documind"
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=10, ge=0, le=100)
    db_pool_recycle: int = Field(default=1800, ge=60)
    db_connect_timeout: int = Field(default=20, ge=1, le=120)
    db_echo: bool = False

    # -------------------------------------------------------------- uploads
    upload_dir: Path = Path("./uploads")
    max_upload_size_mb: int = Field(default=20, ge=1, le=200)
    allowed_extensions: str = ".pdf,.txt"

    # ------------------------------------------------------------- chunking
    chunk_size: int = Field(default=900, ge=100, le=8000)
    chunk_overlap: int = Field(default=150, ge=0, le=2000)

    # ----------------------------------------------------------- embeddings
    embedding_provider: EmbeddingProvider = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, ge=8, le=8192)
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_device: str = "cpu"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"

    # ------------------------------------------------------------------ llm
    llm_provider: LLMProvider = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_timeout_seconds: float = Field(default=120.0, ge=5)
    openai_llm_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    # ---------------------------------------------------------------- rag
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    rag_max_context_chars: int = Field(default=6000, ge=500, le=50000)
    max_question_length: int = Field(default=2000, ge=10, le=10000)

    # ----------------------------------------------------------- security
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    rate_limit_default: str = "120/minute"
    rate_limit_auth: str = "10/minute"
    rate_limit_chat: str = "30/minute"
    rate_limit_upload: str = "20/minute"

    # ------------------------------------------------------------ validators
    @field_validator("secret_key")
    @classmethod
    def _secret_key_strength(cls, value: str) -> str:
        if value == "change-me-in-production" and False:  # pragma: no cover
            raise ValueError("unsafe secret")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk(cls, value: int, info: Any) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @model_validator(mode="after")
    def _validate_production(self) -> "Settings":
        if self.environment == "production":
            if self.secret_key == "change-me-in-production":
                raise ValueError("SECRET_KEY must be set in production")
            if not self.cors_origins.strip():
                self.cors_origins = ""
        return self

    # ------------------------------------------------------------ properties
    @property
    def sqlalchemy_url(self) -> str:
        """Normalise legacy ``postgres://`` URLs to the psycopg 3 driver."""
        url = self.database_url.strip()
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    @property
    def allowed_extension_set(self) -> set[str]:
        return {
            ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
            for ext in self.allowed_extensions.split(",")
            if ext.strip()
        }

    @property
    def cors_origin_list(self) -> list[str]:
        raw = self.cors_origins.strip()
        if not raw or raw == "[]":
            return []
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
