from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.providers.llm import LLMProviderError
from app.providers.llm_impl.ollama import OllamaProvider
from app.providers.llm_impl.openai import OpenAICompatibleProvider, _yield_stream_chunks


def _settings(**overrides: Any) -> Settings:
    base = {
        "llm_provider": "ollama",
        "llm_base_url": "http://localhost:11434",
        "llm_model": "llama3.2",
        "llm_timeout_seconds": 1,
        "llm_max_retries": 1,
    }
    base.update(overrides)
    return Settings(**base)


class _StreamResponse:
    def __init__(self, status_code: int, lines: list[str] | None = None, body: bytes = b"err"):
        self.status_code = status_code
        self._lines = lines or []
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aread(self) -> bytes:
        await asyncio.sleep(0)
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _Client:
    def __init__(self, response=None, error: Exception | None = None):
        self._response = response
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, *args, **kwargs):
        if self._error:
            raise self._error
        return self._response


@pytest.mark.asyncio
async def test_ollama_stream_complete_success_and_errors(monkeypatch):
    provider = OllamaProvider(_settings())

    monkeypatch.setattr(
        "app.providers.llm_impl.ollama.httpx.AsyncClient",
        lambda timeout=1: _Client(
            _StreamResponse(200, lines=['{"response":"A"}', "invalid-json", '{"text":"B"}', ""])
        ),
    )
    chunks = [c async for c in provider.stream_complete("prompt")]
    assert "".join(chunks) == "AB"

    monkeypatch.setattr(
        "app.providers.llm_impl.ollama.httpx.AsyncClient",
        lambda timeout=1: _Client(_StreamResponse(500, body=b"oops")),
    )
    with pytest.raises(LLMProviderError):
        [c async for c in provider.stream_complete("prompt")]

    monkeypatch.setattr(
        "app.providers.llm_impl.ollama.httpx.AsyncClient",
        lambda timeout=1: _Client(error=httpx.RequestError("down")),
    )
    with pytest.raises(LLMProviderError):
        [c async for c in provider.stream_complete("prompt")]


@pytest.mark.asyncio
async def test_openai_stream_complete_and_yield_stream_chunks(monkeypatch):
    provider = OpenAICompatibleProvider(_settings(llm_provider="openai"))

    monkeypatch.setattr(
        "app.providers.llm_impl.openai._yield_stream_chunks",
        lambda *args, **kwargs: _fake_chunks(["A", "B"]),
    )
    chunks = [c async for c in provider.stream_complete("prompt")]
    assert "".join(chunks) == "AB"

    class _RaisingAsyncIter:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(0)
            raise httpx.RequestError("down")

    monkeypatch.setattr(
        "app.providers.llm_impl.openai._yield_stream_chunks",
        lambda *args, **kwargs: _RaisingAsyncIter(),
    )
    with pytest.raises(LLMProviderError):
        [c async for c in provider.stream_complete("prompt")]

    client = _Client(_StreamResponse(500, body=b"error"))
    with pytest.raises(LLMProviderError):
        [c async for c in _yield_stream_chunks(client, url="u", payload={}, headers={})]


async def _fake_chunks(chunks: list[str]):
    for chunk in chunks:
        yield chunk
