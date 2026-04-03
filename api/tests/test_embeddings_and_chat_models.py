from __future__ import annotations

import sys
import types
from typing import Any

import httpx
import pytest

from app.agents import chat_models
from app.core.config import Settings
from app.providers.embedding import EmbeddingError
from app.providers.embedding_impl.mock import MockEmbeddingProvider
from app.providers.embedding_impl.openai import OpenAIEmbeddingProvider
from app.providers.embedding_impl.sentence_transformers import SentenceTransformersEmbeddingProvider


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


def test_filter_init_kwargs_handles_strict_and_kwargs_classes():
    class _Strict:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    class _Flexible:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    strict = chat_models._filter_init_kwargs(_Strict, {"a": 1, "b": 2, "x": 3})
    flexible = chat_models._filter_init_kwargs(_Flexible, {"a": 1, "b": 2, "x": 3})

    assert strict == {"a": 1, "b": 2}
    assert flexible == {"a": 1, "b": 2, "x": 3}


def test_filter_init_kwargs_signature_failure_returns_original(monkeypatch):
    class _Any:
        def __init__(self, a):
            self.a = a

    def _raise_signature_error(*_args, **_kwargs):
        raise ValueError("bad")

    monkeypatch.setattr("app.agents.chat_models.inspect.signature", _raise_signature_error)
    out = chat_models._filter_init_kwargs(_Any, {"a": 1, "extra": 2})
    assert out == {"a": 1, "extra": 2}


def test_create_chat_model_returns_none_when_missing_settings():
    s1 = _settings(llm_base_url=None)
    s2 = _settings(llm_provider="")
    assert chat_models.create_chat_model(s1) is None
    assert chat_models.create_chat_model(s2) is None


def test_cached_chat_model_ollama_and_openai(monkeypatch):
    chat_models._cached_chat_model.cache_clear()

    class _FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(
        sys.modules, "langchain_ollama", types.SimpleNamespace(ChatOllama=_FakeChatOllama)
    )
    monkeypatch.setitem(
        sys.modules, "langchain_openai", types.SimpleNamespace(ChatOpenAI=_FakeChatOpenAI)
    )

    ollama = chat_models._cached_chat_model("ollama", "http://host/", "m", None, 10.0, 2)
    assert isinstance(ollama, _FakeChatOllama)
    assert ollama.kwargs["base_url"] == "http://host"

    openai = chat_models._cached_chat_model("openai", "http://host/", "m", "k", 10.0, 2)
    assert isinstance(openai, _FakeChatOpenAI)
    assert openai.kwargs["base_url"] == "http://host/v1"
    assert openai.kwargs["api_key"] == "k"

    s = _settings(llm_provider="openai", llm_api_key="key")
    created = chat_models.create_chat_model(s)
    assert isinstance(created, _FakeChatOpenAI)


@pytest.mark.asyncio
async def test_mock_embedding_provider_shape_and_name():
    provider = MockEmbeddingProvider(dimension=10)
    emb = await provider.embed("hello")
    assert len(emb) == 10
    assert provider.get_dimension() == 10
    assert provider.get_name() == "mock"


@pytest.mark.asyncio
async def test_openai_embedding_provider_paths(monkeypatch):
    with pytest.raises(EmbeddingError):
        OpenAIEmbeddingProvider(_settings(llm_api_key=None))

    provider = OpenAIEmbeddingProvider(_settings(llm_api_key="key"), model="emb-model")

    class _Client:
        def __init__(self, response=None, error: Exception | None = None):
            self._response = response
            self._error = error

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict[str, Any], headers: dict[str, str]):
            import asyncio

            await asyncio.sleep(0)
            if self._error:
                raise self._error
            return self._response

    class _Response:
        def __init__(self, status_code: int, payload: dict[str, Any] | None = None, text: str = ""):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self):
            return self._payload

    monkeypatch.setattr(
        "app.providers.embedding_impl.openai.httpx.AsyncClient",
        lambda timeout=30.0: _Client(_Response(200, payload={"data": [{"embedding": [0.1, 0.2]}]})),
    )
    emb = await provider.embed("hello")
    assert emb == [0.1, 0.2]
    assert provider.get_dimension() == 2
    assert provider.get_name() == "emb-model"

    monkeypatch.setattr(
        "app.providers.embedding_impl.openai.httpx.AsyncClient",
        lambda timeout=30.0: _Client(error=httpx.RequestError("down")),
    )
    with pytest.raises(EmbeddingError):
        await provider.embed("hello")

    monkeypatch.setattr(
        "app.providers.embedding_impl.openai.httpx.AsyncClient",
        lambda timeout=30.0: _Client(_Response(500, text="oops")),
    )
    with pytest.raises(EmbeddingError):
        await provider.embed("hello")

    class _BadJsonResponse(_Response):
        def json(self):
            raise ValueError("bad")

    monkeypatch.setattr(
        "app.providers.embedding_impl.openai.httpx.AsyncClient",
        lambda timeout=30.0: _Client(_BadJsonResponse(200)),
    )
    with pytest.raises(EmbeddingError):
        await provider.embed("hello")

    monkeypatch.setattr(
        "app.providers.embedding_impl.openai.httpx.AsyncClient",
        lambda timeout=30.0: _Client(_Response(200, payload={"data": []})),
    )
    with pytest.raises(EmbeddingError):
        await provider.embed("hello")

    monkeypatch.setattr(
        "app.providers.embedding_impl.openai.httpx.AsyncClient",
        lambda timeout=30.0: _Client(_Response(200, payload={"data": [{"embedding": "bad"}]})),
    )
    with pytest.raises(EmbeddingError):
        await provider.embed("hello")


def test_openai_embedding_dimension_unknown_raises():
    provider = OpenAIEmbeddingProvider(_settings(llm_api_key="key"))
    with pytest.raises(EmbeddingError):
        provider.get_dimension()


@pytest.mark.asyncio
async def test_sentence_transformers_provider_success_and_error(monkeypatch):
    class _FakeEncoded:
        def tolist(self):
            return [0.5, 0.7]

    class _FakeModel:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, text: str, convert_to_tensor: bool = False):
            return _FakeEncoded()

    fake_module = types.SimpleNamespace(SentenceTransformer=_FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    provider = SentenceTransformersEmbeddingProvider("mini")
    emb = await provider.embed("hello")
    assert emb == [0.5, 0.7]
    assert provider.get_dimension() == 2
    assert provider.get_name() == "mini"

    class _BoomModel(_FakeModel):
        def encode(self, text: str, convert_to_tensor: bool = False):
            raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_BoomModel),
    )
    boom_provider = SentenceTransformersEmbeddingProvider("mini")
    with pytest.raises(EmbeddingError):
        await boom_provider.embed("hello")


def test_sentence_transformers_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(EmbeddingError):
        SentenceTransformersEmbeddingProvider("mini")
