"""Tests for config/Settings behavior."""

from __future__ import annotations


import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_database_url_normalizes_postgres_scheme():
    """Render-style postgres:// URLs are converted to postgresql+asyncpg://."""
    s = Settings(database_url="postgres://user:pass@host/db")
    assert s.database_url == "postgresql+asyncpg://user:pass@host/db"


def test_database_url_normalizes_postgresql_without_driver():
    """postgresql:// without +driver becomes postgresql+asyncpg://."""
    s = Settings(database_url="postgresql://user:pass@host/db")
    assert s.database_url == "postgresql+asyncpg://user:pass@host/db"


def test_database_url_unchanged_for_sqlite():
    """SQLite URLs are left as-is."""
    s = Settings(database_url="sqlite+aiosqlite:///./app.db")
    assert "sqlite" in s.database_url
    assert "asyncpg" not in s.database_url


def test_database_url_unchanged_when_asyncpg_already_present():
    """Already-correct postgresql+asyncpg:// URLs are unchanged."""
    url = "postgresql+asyncpg://user:pass@host/db"
    s = Settings(database_url=url)
    assert s.database_url == url


def test_api_key_required_when_environment_prod():
    """Settings raises when ENVIRONMENT=prod and API_KEY is missing."""
    with pytest.raises((ValueError, ValidationError)) as exc_info:
        Settings(environment="prod", api_key=None)
    assert "API_KEY" in str(exc_info.value).upper()


def test_api_key_required_when_environment_prod_and_empty_string():
    """Settings raises when ENVIRONMENT=prod and API_KEY is empty string."""
    with pytest.raises((ValueError, ValidationError)) as exc_info:
        Settings(environment="prod", api_key="")
    assert "API_KEY" in str(exc_info.value).upper()


def test_api_key_not_required_when_environment_local():
    """API_KEY is optional when environment is local."""
    s = Settings(environment="local", api_key=None)
    assert s.api_key is None


def test_api_key_ok_when_environment_prod():
    """Settings accepts prod when API_KEY is set."""
    s = Settings(environment="prod", api_key="secret-key")
    assert s.environment == "prod"
    assert s.api_key == "secret-key"
