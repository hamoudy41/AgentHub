from functools import lru_cache
from typing import Literal, Optional

from pydantic import AnyHttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "AgentHub"
    environment: Literal["local", "dev", "prod"] = "local"
    api_v1_prefix: str = "/api/v1"
    default_tenant_id: str = "default"
    tenant_header_name: str = "X-Tenant-ID"
    database_url: str = "sqlite+aiosqlite:///./app.db"

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        """Convert Render's postgres:// to postgresql+asyncpg:// for SQLAlchemy async."""
        if not v or "sqlite" in v:
            return v
        s = str(v).strip()
        if s.startswith("postgres://"):
            return "postgresql+asyncpg://" + s[len("postgres://") :]
        if s.startswith("postgresql://") and "+" not in s.split("://")[0]:
            return s.replace("postgresql://", "postgresql+asyncpg://", 1)
        return s

    redis_url: Optional[str] = None
    rate_limit_per_minute: int = 120
    llm_provider: Literal["ollama", "openai_compatible", ""] = ""
    llm_base_url: Optional[AnyHttpUrl] = None
    llm_api_key: Optional[str] = None
    llm_model: str = "llama3.2"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    log_level: str = "INFO"
    enable_prometheus: bool = True
    api_key: Optional[str] = None
    # Comma-separated origins for CORS. Empty or "*" = allow all. Prod: set to frontend URLs.
    cors_allowed_origins: str = "*"
    # Purge ai_call_audit records older than this many days. 0 = disabled.
    ai_audit_retention_days: int = 0
    embedding_model: str = "mock"
    embedding_dimension: int = 384
    search_provider: Literal["duckduckgo", "tavily"] = "duckduckgo"
    search_region: str = (
        "us-en"  # DuckDuckGo region for English results (us-en, uk-en, wt-wt, etc.)
    )
    tavily_api_key: Optional[str] = None

    @model_validator(mode="after")
    def require_api_key_in_prod(self: "Settings") -> "Settings":
        if self.environment == "prod" and (not self.api_key or not self.api_key.strip()):
            raise ValueError(
                "API_KEY is required when ENVIRONMENT=prod. Set API_KEY in your environment."
            )
        return self

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
