from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.metrics import LLM_ERRORS, LLM_LATENCY
from app.security import sanitize_for_logging

from .errors import LLMError, LLMProviderError, LLMTimeoutError
from .types import LLMResult


logger = get_logger(__name__)


async def _post_with_retries(
    *,
    url: str,
    json_payload: dict[str, Any],
    headers: dict[str, str] | None,
    timeout_seconds: float,
    max_retries: int,
):
    async for attempt in AsyncRetrying(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(max(1, int(max_retries))),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        reraise=True,
    ):
        with attempt:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                return await client.post(url, json=json_payload, headers=headers)


def _build_openai_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _build_openai_headers(settings: Settings) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    return headers


class OllamaProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        timeout: float | None = None,
    ) -> LLMResult:
        base_url = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 512},
        }
        if system_prompt:
            payload["system"] = system_prompt

        timeout_seconds = timeout if timeout is not None else self._settings.llm_timeout_seconds
        started = time.perf_counter()

        try:
            response = await _post_with_retries(
                url=url,
                json_payload=payload,
                headers=None,
                timeout_seconds=timeout_seconds,
                max_retries=self._settings.llm_max_retries,
            )
        except httpx.TimeoutException as exc:
            logger.error(
                "llm.ollama_timeout",
                error=str(exc),
                timeout=timeout_seconds,
                prompt_preview=sanitize_for_logging(prompt, 100),
            )
            LLM_ERRORS.labels(provider="ollama", error_type="timeout").inc()
            raise LLMTimeoutError(f"Ollama request timed out after {timeout_seconds}s") from exc
        except httpx.RequestError as exc:
            logger.error(
                "llm.ollama_request_error",
                error=str(exc),
                error_type=type(exc).__name__,
                prompt_preview=sanitize_for_logging(prompt, 100),
            )
            LLM_ERRORS.labels(provider="ollama", error_type="request_error").inc()
            raise LLMProviderError("Ollama request failed", provider="ollama") from exc

        if response.status_code != 200:
            error_body = sanitize_for_logging(response.text, 500)
            logger.error(
                "llm.ollama_error_response",
                status_code=response.status_code,
                response_preview=error_body,
            )
            raise LLMProviderError(
                f"Ollama returned {response.status_code}: {error_body}",
                status_code=response.status_code,
                provider="ollama",
            )

        data = response.json()
        text = data.get("response") or data.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            logger.error("llm.ollama_empty_response", data_keys=list(data.keys()))
            raise LLMProviderError("Ollama returned empty response", provider="ollama")

        latency_ms = (time.perf_counter() - started) * 1000
        LLM_LATENCY.labels(provider="ollama", flow="generic").observe(latency_ms / 1000.0)
        logger.info(
            "llm.ollama_success",
            latency_ms=round(latency_ms, 2),
            response_length=len(text),
            model=data.get("model", self._settings.llm_model),
        )
        return LLMResult(
            raw_text=text.strip(),
            model=data.get("model", self._settings.llm_model),
            latency_ms=round(latency_ms, 2),
        )

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        base_url = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "prompt": prompt,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            try:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise LLMError(
                            f"Ollama returned {response.status_code}: {body.decode()[:500]}"
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk = data.get("response") or data.get("text") or ""
                        if isinstance(chunk, str) and chunk:
                            yield chunk
            except httpx.RequestError as exc:
                logger.warning("llm.ollama_stream_error", error=str(exc))
                raise LLMError("Ollama stream request failed") from exc


class OpenAICompatibleProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        timeout: float | None = None,
    ) -> LLMResult:
        base_url = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": _build_openai_messages(prompt, system_prompt),
            "max_tokens": 2048,
        }
        timeout_seconds = timeout if timeout is not None else self._settings.llm_timeout_seconds
        started = time.perf_counter()

        try:
            response = await _post_with_retries(
                url=url,
                json_payload=payload,
                headers=_build_openai_headers(self._settings),
                timeout_seconds=timeout_seconds,
                max_retries=self._settings.llm_max_retries,
            )
        except httpx.TimeoutException as exc:
            logger.error(
                "llm.openai_timeout",
                error=str(exc),
                timeout=timeout_seconds,
                prompt_preview=sanitize_for_logging(prompt, 100),
            )
            LLM_ERRORS.labels(provider="openai", error_type="timeout").inc()
            raise LLMTimeoutError(f"OpenAI request timed out after {timeout_seconds}s") from exc
        except httpx.RequestError as exc:
            logger.error(
                "llm.openai_request_error",
                error=str(exc),
                error_type=type(exc).__name__,
                prompt_preview=sanitize_for_logging(prompt, 100),
            )
            LLM_ERRORS.labels(provider="openai", error_type="request_error").inc()
            raise LLMProviderError("OpenAI-compatible request failed", provider="openai") from exc

        if response.status_code != 200:
            error_body = sanitize_for_logging(response.text, 500)
            logger.error(
                "llm.openai_error_response",
                status_code=response.status_code,
                response_preview=error_body,
            )
            raise LLMProviderError(
                f"OpenAI-compatible returned {response.status_code}: {error_body}",
                status_code=response.status_code,
                provider="openai",
            )

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            logger.error("llm.openai_no_choices", data_keys=list(data.keys()))
            raise LLMProviderError("OpenAI-compatible returned no choices", provider="openai")

        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            logger.error("llm.openai_empty_content", message_keys=list(message.keys()))
            raise LLMProviderError("OpenAI-compatible returned empty content", provider="openai")

        latency_ms = (time.perf_counter() - started) * 1000
        LLM_LATENCY.labels(provider="openai", flow="generic").observe(latency_ms / 1000.0)
        logger.info(
            "llm.openai_success",
            latency_ms=round(latency_ms, 2),
            response_length=len(text),
            model=data.get("model", self._settings.llm_model),
        )
        return LLMResult(
            raw_text=text.strip(),
            model=data.get("model", self._settings.llm_model),
            latency_ms=round(latency_ms, 2),
        )

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        base_url = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": _build_openai_messages(prompt, system_prompt),
            "max_tokens": 2048,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            try:
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=_build_openai_headers(self._settings),
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise LLMError(
                            f"OpenAI-compatible returned {response.status_code}: {body.decode()[:500]}"
                        )
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        for choice in data.get("choices", []):
                            delta = choice.get("delta", {})
                            content = delta.get("content")
                            if isinstance(content, str) and content:
                                yield content
            except httpx.RequestError as exc:
                logger.warning("llm.openai_stream_error", error=str(exc))
                raise LLMError("OpenAI-compatible stream request failed") from exc
