from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services_llm import LLMClient, LLMNotConfiguredError


def _mock_response(status_code=200, json_body=None):
    r = MagicMock()
    r.status_code = status_code
    r.text = ""
    r.json.return_value = json_body or {}
    return r


@pytest.mark.asyncio
async def test_is_configured_false_when_no_url():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = None
        m.return_value.llm_provider = "ollama"
        client = LLMClient()
        assert client.is_configured() is False


@pytest.mark.asyncio
async def test_is_configured_false_when_no_provider():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = "http://localhost"
        m.return_value.llm_provider = ""
        client = LLMClient()
        assert client.is_configured() is False


@pytest.mark.asyncio
async def test_complete_raises_when_not_configured():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = None
        m.return_value.llm_provider = ""
        client = LLMClient()
        with pytest.raises(LLMNotConfiguredError) as exc_info:
            await client.complete("hello", tenant_id="t1")
        assert "LLM not configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_complete_ollama_success():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = "http://localhost:11434"
        m.return_value.llm_provider = "ollama"
        m.return_value.llm_model = "llama3.2"
        m.return_value.llm_timeout_seconds = 30
        client = LLMClient()
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(
                return_value=_mock_response(
                    200, {"response": "Classified as invoice.", "model": "llama3.2"}
                )
            )
            mock_client.return_value.__aenter__.return_value.post = mock_post
            out = await client.complete("Classify: invoice", tenant_id="t1")
            assert out.raw_text == "Classified as invoice."
            assert out.model == "llama3.2"


@pytest.mark.asyncio
async def test_complete_ollama_with_system_prompt():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = "http://localhost:11434"
        m.return_value.llm_provider = "ollama"
        m.return_value.llm_model = "llama3.2"
        m.return_value.llm_timeout_seconds = 30
        client = LLMClient()
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(
                return_value=_mock_response(200, {"response": "ok", "model": "llama3.2"})
            )
            mock_client.return_value.__aenter__.return_value.post = mock_post
            await client.complete("hi", system_prompt="You are helpful.", tenant_id="t1")
            call_kw = mock_post.call_args.kwargs
            assert call_kw["json"].get("system") == "You are helpful."


@pytest.mark.asyncio
async def test_complete_ollama_non_200_raises():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = "http://localhost:11434"
        m.return_value.llm_provider = "ollama"
        m.return_value.llm_model = "llama3.2"
        m.return_value.llm_timeout_seconds = 30
        client = LLMClient()
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=_mock_response(500, None))
            mock_post.return_value.text = "Server error"
            mock_client.return_value.__aenter__.return_value.post = mock_post
            with pytest.raises(Exception):
                await client.complete("hi", tenant_id="t1")


@pytest.mark.asyncio
async def test_complete_ollama_empty_response_raises():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = "http://localhost:11434"
        m.return_value.llm_provider = "ollama"
        m.return_value.llm_model = "llama3.2"
        m.return_value.llm_timeout_seconds = 30
        client = LLMClient()
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(
                return_value=_mock_response(200, {"response": "  ", "model": "x"})
            )
            mock_client.return_value.__aenter__.return_value.post = mock_post
            with pytest.raises(Exception):
                await client.complete("hi", tenant_id="t1")


@pytest.mark.asyncio
async def test_complete_openai_success():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = "http://localhost:8000"
        m.return_value.llm_provider = "openai_compatible"
        m.return_value.llm_model = "llama"
        m.return_value.llm_api_key = None
        m.return_value.llm_timeout_seconds = 30
        client = LLMClient()
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(
                return_value=_mock_response(
                    200,
                    {
                        "choices": [{"message": {"content": "The answer is 42."}}],
                        "model": "llama",
                    },
                )
            )
            mock_client.return_value.__aenter__.return_value.post = mock_post
            out = await client.complete("What is the answer?", tenant_id="t1")
            assert out.raw_text == "The answer is 42."
            assert out.model == "llama"


@pytest.mark.asyncio
async def test_complete_openai_with_api_key():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = "http://localhost:8000"
        m.return_value.llm_provider = "openai_compatible"
        m.return_value.llm_model = "llama"
        m.return_value.llm_api_key = "sk-secret"
        m.return_value.llm_timeout_seconds = 30
        client = LLMClient()
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(
                return_value=_mock_response(
                    200,
                    {
                        "choices": [{"message": {"content": "Hi"}}],
                        "model": "llama",
                    },
                )
            )
            mock_client.return_value.__aenter__.return_value.post = mock_post
            await client.complete("hi", tenant_id="t1")
            call_kw = mock_post.call_args.kwargs
            assert "Authorization" in call_kw["headers"]
            assert "Bearer sk-secret" in call_kw["headers"]["Authorization"]


@pytest.mark.asyncio
async def test_complete_openai_no_choices_raises():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = "http://localhost:8000"
        m.return_value.llm_provider = "openai_compatible"
        m.return_value.llm_model = "llama"
        m.return_value.llm_api_key = None
        m.return_value.llm_timeout_seconds = 30
        client = LLMClient()
        with patch("httpx.AsyncClient") as mock_client:
            mock_post = AsyncMock(return_value=_mock_response(200, {"choices": []}))
            mock_client.return_value.__aenter__.return_value.post = mock_post
            with pytest.raises(Exception):
                await client.complete("hi", tenant_id="t1")


@pytest.mark.asyncio
async def test_generate_notary_summary_raises_when_not_configured():
    with patch("app.services_llm.get_settings") as m:
        m.return_value.llm_base_url = None
        m.return_value.llm_provider = ""
        client = LLMClient()
        with pytest.raises(LLMNotConfiguredError) as exc_info:
            await client.generate_notary_summary("Summarize this.", tenant_id="t1")
        assert "LLM not configured" in str(exc_info.value)
