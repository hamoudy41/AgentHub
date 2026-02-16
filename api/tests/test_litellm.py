"""Tests for LiteLLM gateway (TDD)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_litellm_complete_returns_text():
    """LiteLLM gateway complete() returns LLMResult with raw_text."""
    with patch("app.services_litellm_gateway.acompletion") as mock_complete:
        mock_response = AsyncMock()
        mock_response.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": "Hello world"})()})()
        ]
        mock_response.model = "llama3.2"
        mock_response.usage = type("Usage", (), {"total_tokens": 5})()
        mock_complete.return_value = mock_response

        from app.services_litellm_gateway import litellm_gateway

        result = await litellm_gateway.complete(
            prompt="Say hello",
            model="ollama/llama3.2",
        )
        assert result.raw_text == "Hello world"
        assert result.model == "llama3.2"


@pytest.mark.asyncio
async def test_litellm_complete_uses_provided_model():
    """LiteLLM gateway passes model to acompletion."""
    with patch("app.services_litellm_gateway.acompletion") as mock_complete:
        mock_response = AsyncMock()
        mock_response.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": "Hi"})()})()
        ]
        mock_response.model = "gpt-4"
        mock_response.usage = type("Usage", (), {"total_tokens": 2})()
        mock_complete.return_value = mock_response

        from app.services_litellm_gateway import litellm_gateway

        await litellm_gateway.complete(
            prompt="Hi",
            model="openai/gpt-4",
        )
        mock_complete.assert_called_once()
        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["model"] == "openai/gpt-4"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]


@pytest.mark.asyncio
async def test_litellm_complete_with_system_prompt():
    """LiteLLM gateway includes system_prompt in messages when provided."""
    with patch("app.services_litellm_gateway.acompletion") as mock_complete:
        mock_response = AsyncMock()
        mock_response.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": "OK"})()})()
        ]
        mock_response.model = "ollama/llama3.2"
        mock_response.usage = type("Usage", (), {"total_tokens": 1})()
        mock_complete.return_value = mock_response

        from app.services_litellm_gateway import litellm_gateway

        await litellm_gateway.complete(
            prompt="Classify this",
            model="ollama/llama3.2",
            system_prompt="You are a classifier.",
        )
        call_kwargs = mock_complete.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a classifier."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Classify this"


@pytest.mark.asyncio
async def test_litellm_complete_raises_on_error():
    """LiteLLM gateway raises LiteLLMGatewayError on completion failure."""
    with patch("app.services_litellm_gateway.acompletion") as mock_complete:
        mock_complete.side_effect = Exception("API rate limit")

        from app.services_litellm_gateway import LiteLLMGatewayError, litellm_gateway

        with pytest.raises(LiteLLMGatewayError) as exc_info:
            await litellm_gateway.complete(
                prompt="Hi",
                model="openai/gpt-4",
            )
        assert "API rate limit" in str(exc_info.value)
