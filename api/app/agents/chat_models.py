"""LangChain chat-model creation for the agent stack.

This module isolates provider-specific wiring (Ollama vs OpenAI-compatible) and keeps
`react_agent.py` focused on orchestration.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

from app.core.config import Settings


def _filter_init_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs not accepted by a class' `__init__` (helps across dependency versions)."""
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return kwargs

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = set(sig.parameters) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


@lru_cache(maxsize=8)
def _cached_chat_model(
    provider: str,
    base_url: str,
    model: str,
    api_key: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> Any:
    base = str(base_url).rstrip("/")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = {
            "base_url": base,
            "model": model,
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 512,
            # Some versions use `timeout`, others `request_timeout`.
            "timeout": timeout_seconds,
            "request_timeout": timeout_seconds,
        }
        return ChatOllama(**_filter_init_kwargs(ChatOllama, kwargs))

    # openai_compatible
    from langchain_openai import ChatOpenAI

    kwargs = {
        "base_url": f"{base}/v1",
        "api_key": api_key or "not-needed",
        "model": model,
        "temperature": 0,
        "timeout": timeout_seconds,
        "max_retries": max_retries,
    }
    return ChatOpenAI(**_filter_init_kwargs(ChatOpenAI, kwargs))


def create_chat_model(settings: Settings) -> Any:
    """Create a configured LangChain chat model based on current settings."""
    if not settings.llm_base_url or not settings.llm_provider:
        return None

    timeout_seconds = float(settings.llm_timeout_seconds)
    max_retries = int(settings.llm_max_retries)

    return _cached_chat_model(
        str(settings.llm_provider),
        str(settings.llm_base_url),
        str(settings.llm_model),
        settings.llm_api_key,
        timeout_seconds,
        max_retries,
    )
