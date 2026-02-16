"""Tests for streaming AI endpoints (TDD)."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_ask_stream_returns_sse_content_type(client, tenant_headers):
    """Streaming ask returns 200 with text/event-stream content type."""
    with patch("app.api.run_ask_flow_stream") as mock_stream:

        async def fake_stream(*args, **kwargs):
            yield "Hello"
            yield " world"

        mock_stream.return_value = fake_stream()
        async with client.stream(
            "POST",
            "/api/v1/ai/ask/stream",
            headers=tenant_headers,
            json={"question": "What is the total?", "context": "The total is 50 EUR."},
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_ask_stream_yields_token_events(client, tenant_headers):
    """Streaming ask yields SSE events with token data."""
    with patch("app.api.run_ask_flow_stream") as mock_stream:

        async def fake_stream(*args, **kwargs):
            yield "50"
            yield " EUR"

        mock_stream.return_value = fake_stream()
        chunks = []
        async with client.stream(
            "POST",
            "/api/v1/ai/ask/stream",
            headers=tenant_headers,
            json={"question": "Total?", "context": "Total is 50 EUR."},
        ) as r:
            assert r.status_code == 200
            async for chunk in r.aiter_text():
                chunks.append(chunk)
        body = "".join(chunks)
        assert '"token":"50"' in body or '"token": "50"' in body
        assert '"token":" EUR"' in body or '"token": " EUR"' in body
        assert '"done":true' in body or '"done": true' in body


@pytest.mark.asyncio
async def test_ask_stream_returns_400_when_llm_not_configured(client, tenant_headers):
    """Streaming ask returns 400 when LLM is not configured."""
    with patch("app.services_ai_flows.llm_client.is_configured", return_value=False):
        r = await client.post(
            "/api/v1/ai/ask/stream",
            headers=tenant_headers,
            json={"question": "Hi", "context": "Context here."},
        )
    assert r.status_code == 400
    assert "LLM not configured" in r.json()["detail"]


@pytest.mark.asyncio
async def test_ask_stream_validates_payload(client, tenant_headers):
    """Streaming ask returns 422 for invalid payload (missing fields)."""
    r = await client.post(
        "/api/v1/ai/ask/stream",
        headers=tenant_headers,
        json={},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_ask_stream_respects_tenant(client, tenant_headers):
    """Streaming ask passes tenant_id to the flow."""
    with patch("app.api.run_ask_flow_stream") as mock_stream:

        async def fake_stream(*args, **kwargs):
            yield "ok"

        mock_stream.return_value = fake_stream()
        async with client.stream(
            "POST",
            "/api/v1/ai/ask/stream",
            headers=tenant_headers,
            json={"question": "Q", "context": "C"},
        ) as r:
            assert r.status_code == 200
        mock_stream.assert_called_once()
        assert mock_stream.call_args.kwargs["tenant_id"] == "tenant-1"
