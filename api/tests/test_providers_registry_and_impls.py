from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.providers.llm import LLMNotConfiguredError, LLMProviderError, LLMTimeoutError
from app.providers.llm_impl.ollama import OllamaProvider
from app.providers.llm_impl.openai import (
    OpenAICompatibleProvider,
    _build_headers,
    _build_messages,
    _parse_stream_line,
)
from app.providers.registry import ProviderRegistry
from app.providers.search import SearchError
from app.providers.search_impl.duckduckgo import _extract_results
from app.providers.search_impl.tavily import TavilySearchProvider


def _settings(**overrides: Any) -> Settings:
    base = {
        "llm_provider": "ollama",
        "llm_base_url": "http://localhost:11434",
        "llm_model": "llama3.2",
        "llm_timeout_seconds": 1,
        "llm_max_retries": 1,
        "embedding_provider": "mock",
        "search_provider": "mock",
    }
    base.update(overrides)
    return Settings(**base)


def test_openai_build_messages_and_headers():
    assert _build_messages("hello") == [{"role": "user", "content": "hello"}]
    assert _build_messages("hello", "sys") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]

    headers = _build_headers(_settings(llm_api_key="secret"))
    assert headers["Authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    ("line", "is_done", "chunk"),
    [
        ("", False, None),
        ("no-prefix", False, None),
        ("data: [DONE]", True, None),
        ('data: {"choices":[{"delta":{"content":"hi"}}]}', False, "hi"),
        ('data: {"choices":[{"delta":{}}]}', False, None),
        ("data: {invalid", False, None),
    ],
)
def test_openai_parse_stream_line_cases(line: str, is_done: bool, chunk: str | None):
    assert _parse_stream_line(line) == (is_done, chunk)


def test_duckduckgo_extract_results_handles_nested_topics():
    data = {
        "RelatedTopics": [
            {"Text": "Title 1 - Snippet 1", "FirstURL": "https://a.example"},
            {
                "Topics": [
                    {"Text": "Title 2 - Snippet 2", "FirstURL": "https://b.example"},
                    {"Text": "skip missing url"},
                ]
            },
        ]
    }

    results = _extract_results(data, limit=5)
    assert len(results) == 2
    assert results[0].url == "https://a.example"
    assert results[1].url == "https://b.example"


def test_provider_registry_llm_provider_selection_and_cache(monkeypatch):
    class _FakeOllama:
        def __init__(self, settings):
            self.settings = settings

    class _FakeOpenAI:
        def __init__(self, settings):
            self.settings = settings

    monkeypatch.setattr("app.providers.registry.OllamaProvider", _FakeOllama)
    monkeypatch.setattr("app.providers.registry.OpenAICompatibleProvider", _FakeOpenAI)

    reg = ProviderRegistry(_settings(llm_provider="ollama"))
    p1 = reg.get_llm_provider()
    p2 = reg.get_llm_provider()
    p3 = reg.get_llm_provider(force_new=True)

    assert isinstance(p1, _FakeOllama)
    assert p1 is p2
    assert p3 is not p1

    reg2 = ProviderRegistry(_settings(llm_provider="openai_compatible"))
    assert isinstance(reg2.get_llm_provider(), _FakeOpenAI)


def test_provider_registry_llm_unknown_raises():
    reg = ProviderRegistry(_settings())
    reg._settings.llm_provider = "unknown"
    with pytest.raises(LLMNotConfiguredError):
        reg.get_llm_provider()


def test_provider_registry_embedding_and_search_selection(monkeypatch):
    class _FakeSentence:
        def __init__(self, model_name: str, settings):
            self.model_name = model_name

    class _FakeOpenAIEmbedding:
        def __init__(self, settings, model: str):
            self.model = model

    class _FakeDDG:
        pass

    class _FakeTavily:
        def __init__(self, api_key: str, settings):
            self.api_key = api_key

    monkeypatch.setattr(
        "app.providers.registry.SentenceTransformersEmbeddingProvider", _FakeSentence
    )
    monkeypatch.setattr("app.providers.registry.OpenAIEmbeddingProvider", _FakeOpenAIEmbedding)
    monkeypatch.setattr("app.providers.registry.DuckDuckGoSearchProvider", _FakeDDG)
    monkeypatch.setattr("app.providers.registry.TavilySearchProvider", _FakeTavily)

    reg_sentence = ProviderRegistry(
        _settings(embedding_provider="sentence-transformers", embedding_model="mini")
    )
    emb_sentence = reg_sentence.get_embedding_provider()
    assert isinstance(emb_sentence, _FakeSentence)
    assert emb_sentence.model_name == "mini"

    reg_openai_emb = ProviderRegistry(_settings(embedding_provider="openai", embedding_model="emb"))
    emb_openai = reg_openai_emb.get_embedding_provider()
    assert isinstance(emb_openai, _FakeOpenAIEmbedding)
    assert emb_openai.model == "emb"

    reg_ddg = ProviderRegistry(_settings(search_provider="duckduckgo"))
    assert isinstance(reg_ddg.get_search_provider(), _FakeDDG)

    reg_tavily = ProviderRegistry(_settings(search_provider="tavily", tavily_api_key="k"))
    tavily = reg_tavily.get_search_provider()
    assert isinstance(tavily, _FakeTavily)
    assert tavily.api_key == "k"


def test_provider_registry_search_tavily_missing_key_and_unknown():
    reg_missing = ProviderRegistry(_settings(search_provider="tavily", tavily_api_key=None))
    with pytest.raises(SearchError):
        reg_missing.get_search_provider()

    reg_unknown = ProviderRegistry(_settings())
    reg_unknown._settings.search_provider = "x"
    with pytest.raises(SearchError):
        reg_unknown.get_search_provider()


def test_provider_registry_reset_caches(monkeypatch):
    class _FakeOllama:
        def __init__(self, settings):
            self.settings = settings

    monkeypatch.setattr("app.providers.registry.OllamaProvider", _FakeOllama)
    reg = ProviderRegistry(_settings())
    p1 = reg.get_llm_provider()
    reg.reset_caches()
    p2 = reg.get_llm_provider()
    assert p1 is not p2


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.mark.asyncio
async def test_ollama_complete_success(monkeypatch):
    provider = OllamaProvider(_settings())

    async def _fake_post(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(200, payload={"response": "hello", "model": "llama-test"})

    monkeypatch.setattr("app.providers.llm_impl.ollama._post_with_retries", _fake_post)

    out = await provider.complete("prompt")
    assert out.raw_text == "hello"
    assert out.model == "llama-test"


@pytest.mark.asyncio
async def test_ollama_complete_timeout_and_request_error(monkeypatch):
    provider = OllamaProvider(_settings())

    async def _timeout(**kwargs):
        raise httpx.TimeoutException("slow")

    monkeypatch.setattr("app.providers.llm_impl.ollama._post_with_retries", _timeout)
    with pytest.raises(LLMTimeoutError):
        await provider.complete("prompt")

    async def _request_err(**kwargs):
        raise httpx.RequestError("down")

    monkeypatch.setattr("app.providers.llm_impl.ollama._post_with_retries", _request_err)
    with pytest.raises(LLMProviderError):
        await provider.complete("prompt")


@pytest.mark.asyncio
async def test_ollama_complete_invalid_responses(monkeypatch):
    provider = OllamaProvider(_settings())

    async def _bad_status(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(500, text="oops")

    monkeypatch.setattr("app.providers.llm_impl.ollama._post_with_retries", _bad_status)
    with pytest.raises(LLMProviderError):
        await provider.complete("prompt")

    class _BadJsonResp(_FakeResponse):
        def json(self):
            raise ValueError("bad")

    async def _bad_json(**kwargs):
        await asyncio.sleep(0)
        return _BadJsonResp(200)

    monkeypatch.setattr("app.providers.llm_impl.ollama._post_with_retries", _bad_json)
    with pytest.raises(LLMProviderError):
        await provider.complete("prompt")

    async def _empty(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(200, payload={"response": "  "})

    monkeypatch.setattr("app.providers.llm_impl.ollama._post_with_retries", _empty)
    with pytest.raises(LLMProviderError):
        await provider.complete("prompt")


@pytest.mark.asyncio
async def test_openai_complete_success_and_failures(monkeypatch):
    provider = OpenAICompatibleProvider(_settings(llm_provider="openai"))

    async def _success(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(
            200,
            payload={
                "model": "gpt-x",
                "choices": [{"message": {"content": "hi"}}],
            },
        )

    monkeypatch.setattr("app.providers.llm_impl.openai._post_with_retries", _success)
    out = await provider.complete("prompt")
    assert out.raw_text == "hi"
    assert out.model == "gpt-x"

    async def _timeout(**kwargs):
        raise httpx.TimeoutException("slow")

    monkeypatch.setattr("app.providers.llm_impl.openai._post_with_retries", _timeout)
    with pytest.raises(LLMTimeoutError):
        await provider.complete("prompt")

    async def _request_err(**kwargs):
        raise httpx.RequestError("down")

    monkeypatch.setattr("app.providers.llm_impl.openai._post_with_retries", _request_err)
    with pytest.raises(LLMProviderError):
        await provider.complete("prompt")

    async def _bad_status(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(429, text="rate")

    monkeypatch.setattr("app.providers.llm_impl.openai._post_with_retries", _bad_status)
    with pytest.raises(LLMProviderError):
        await provider.complete("prompt")

    class _BadJsonResp(_FakeResponse):
        def json(self):
            raise ValueError("bad")

    async def _bad_json(**kwargs):
        await asyncio.sleep(0)
        return _BadJsonResp(200)

    monkeypatch.setattr("app.providers.llm_impl.openai._post_with_retries", _bad_json)
    with pytest.raises(LLMProviderError):
        await provider.complete("prompt")

    async def _no_choices(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(200, payload={"choices": []})

    monkeypatch.setattr("app.providers.llm_impl.openai._post_with_retries", _no_choices)
    with pytest.raises(LLMProviderError):
        await provider.complete("prompt")

    async def _empty_content(**kwargs):
        await asyncio.sleep(0)
        return _FakeResponse(200, payload={"choices": [{"message": {"content": " "}}]})

    monkeypatch.setattr("app.providers.llm_impl.openai._post_with_retries", _empty_content)
    with pytest.raises(LLMProviderError):
        await provider.complete("prompt")


@pytest.mark.asyncio
async def test_tavily_provider_validation_and_get_name():
    with pytest.raises(SearchError):
        TavilySearchProvider(api_key="")

    provider = TavilySearchProvider(api_key="key")
    assert provider.get_name() == "tavily"
