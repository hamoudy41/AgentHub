from __future__ import annotations

import os

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
# LLM must be configured for AI endpoints; use mock URL (tests mock httpx or llm_client)
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["LLM_BASE_URL"] = "http://localhost:11434"

from app.api import create_app
from app.db import get_engine
from app.models import AiCallAudit, Base, Document


@pytest.fixture(autouse=True)
async def _clean_db():
    """Ensure tables exist and clear documents + audit before each test for isolation."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(delete(AiCallAudit))
        await conn.execute(delete(Document))
    yield


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def tenant_headers():
    return {"X-Tenant-ID": "tenant-1"}


@pytest.fixture
def app_with_api_key():
    """App with API_KEY and ENVIRONMENT=prod for auth tests."""
    from app.core.config import get_settings

    orig_env = os.environ.copy()
    os.environ["API_KEY"] = "test-secret-key"
    os.environ["ENVIRONMENT"] = "prod"
    get_settings.cache_clear()
    try:
        yield create_app()
    finally:
        os.environ.clear()
        os.environ.update(orig_env)
        get_settings.cache_clear()
        # Restore conftest defaults
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://localhost:11434"


@pytest.fixture
async def client_with_api_key(app_with_api_key):
    """Client for app with API key auth enabled."""
    transport = ASGITransport(app=app_with_api_key)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
