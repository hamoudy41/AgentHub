"""Tests for agent chat API endpoints - TDD."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_agent_chat_returns_200_with_answer(client, tenant_headers):
    """POST /ai/agents/chat returns 200 with answer and tools_used."""
    with patch("app.api.run_agent", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {
            "answer": "The answer is 4.",
            "tools_used": ["calculator_tool"],
        }
        r = await client.post(
            "/api/v1/ai/agents/chat",
            headers=tenant_headers,
            json={"message": "What is 2+2?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["answer"] == "The answer is 4."
        assert "calculator_tool" in data["tools_used"]


@pytest.mark.asyncio
async def test_agent_chat_validates_payload(client, tenant_headers):
    """POST /ai/agents/chat returns 422 for empty message."""
    r = await client.post(
        "/api/v1/ai/agents/chat",
        headers=tenant_headers,
        json={"message": ""},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_agent_chat_validates_message_max_length(client, tenant_headers):
    """POST /ai/agents/chat returns 422 for message exceeding 4000 chars."""
    r = await client.post(
        "/api/v1/ai/agents/chat",
        headers=tenant_headers,
        json={"message": "x" * 4001},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_agent_chat_stream_returns_sse(client, tenant_headers):
    """POST /ai/agents/chat/stream returns SSE streaming."""
    with patch("app.api.run_agent_stream") as mock_stream:

        async def fake_stream(*args, **kwargs):
            yield "The "
            yield "answer."

        mock_stream.return_value = fake_stream()

        async with client.stream(
            "POST",
            "/api/v1/ai/agents/chat/stream",
            headers=tenant_headers,
            json={"message": "Hi"},
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            body = (await r.aread()).decode()
            assert "data:" in body
            assert "The answer." in body or "done" in body.lower()


@pytest.mark.asyncio
async def test_agent_chat_returns_fallback_when_llm_not_configured(client, tenant_headers):
    """POST /ai/agents/chat returns 200 with fallback when LLM not configured."""
    with patch("app.core.config.get_settings") as mock_settings:
        mock_settings.return_value.llm_base_url = None
        mock_settings.return_value.llm_provider = ""

        r = await client.post(
            "/api/v1/ai/agents/chat",
            headers=tenant_headers,
            json={"message": "What is 2+2?"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "LLM not configured" in data["answer"]
        assert data.get("error") == "llm_not_configured"


@pytest.mark.asyncio
async def test_agent_chat_stream_handles_errors(client, tenant_headers):
    """POST /ai/agents/chat/stream yields error event on exception."""
    with patch("app.api.run_agent_stream") as mock_stream:

        async def fail_stream(*args, **kwargs):
            raise RuntimeError("Agent failed")
            yield  # make async generator

        mock_stream.return_value = fail_stream()

        async with client.stream(
            "POST",
            "/api/v1/ai/agents/chat/stream",
            headers=tenant_headers,
            json={"message": "Hi"},
        ) as r:
            assert r.status_code == 200
            body = (await r.aread()).decode()
            assert "error" in body.lower() or "Agent failed" in body
