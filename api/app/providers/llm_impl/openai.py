"""OpenAI-compatible LLM provider implementation."""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Optional

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.metrics import LLM_ERRORS, LLM_LATENCY
from app.security import sanitize_for_logging

from ..llm import LLMProvider, LLMResult, LLMProviderError, LLMTimeoutError

logger = get_logger(__name__)


async def _post_with_retries(
    *,
    url: str,
    json_payload: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
    timeout_seconds: float = 60.0,
    max_retries: int = 2,
) -> httpx.Response:
    """Make HTTP POST with exponential backoff retries."""
    async for attempt in AsyncRetrying(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(max(1, int(max_retries))),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        reraise=True,
    ):
        with attempt:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                return await client.post(url, json=json_payload, headers=headers)


def _build_messages(prompt: str, system_prompt: Optional[str] = None) -> list[dict[str, str]]:
    """Build OpenAI-compatible messages format."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _build_headers(settings: Settings) -> dict[str, str]:
    """Build HTTP headers for OpenAI-compatible API."""
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    return headers


def _parse_stream_line(line: str) -> tuple[bool, str | None]:
    """Parse one OpenAI SSE line.

    Returns:
        (is_done, chunk)
        - is_done=True indicates stream completion token [DONE]
        - chunk contains text token when present, else None
    """
    if not line or not line.startswith("data: "):
        return False, None

    data_str = line[6:].strip()
    if data_str == "[DONE]":
        return True, None

    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return False, None

    choices = data.get("choices") or []
    if not choices:
        return False, None

    delta = choices[0].get("delta") or {}
    chunk = delta.get("content") or ""
    if isinstance(chunk, str) and chunk:
        return False, chunk
    return False, None


async def _yield_stream_chunks(
    client: httpx.AsyncClient,
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[str]:
    """Open a streaming request and yield parsed token chunks."""
    async with client.stream("POST", url, json=payload, headers=headers) as response:
        if response.status_code != 200:
            body = await response.aread()
            raise LLMProviderError(
                f"OpenAI-compatible returned {response.status_code}: {body.decode()[:500]}"
            )

        async for line in response.aiter_lines():
            done, chunk = _parse_stream_line(line)
            if done:
                return
            if chunk is not None:
                yield chunk


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible LLM provider implementation.

    Works with OpenAI, Azure OpenAI, localized OpenAI clones (vLLM, etc.).
    Uses the standard OpenAI chat completions API.

    Args:
        settings: Application settings with llm_base_url, llm_model, etc.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize OpenAI-compatible provider with settings."""
        self._settings = settings

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> LLMResult:
        """Complete a prompt using OpenAI-compatible API.

        Args:
            prompt: User prompt to complete
            system_prompt: Optional system message

        Returns:
            LLMResult with completion text and metadata

        Raises:
            LLMTimeoutError: If request times out
            LLMProviderError: If provider returns an error
        """
        base_url = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": _build_messages(prompt, system_prompt),
            "max_tokens": 2048,
        }
        timeout_seconds = self._settings.llm_timeout_seconds
        started = time.perf_counter()

        try:
            response = await _post_with_retries(
                url=url,
                json_payload=payload,
                headers=_build_headers(self._settings),
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
            raise LLMProviderError("OpenAI-compatible request failed") from exc

        if response.status_code != 200:
            error_body = sanitize_for_logging(response.text, 500)
            logger.error(
                "llm.openai_error_response",
                status_code=response.status_code,
                response_preview=error_body,
            )
            raise LLMProviderError(
                f"OpenAI-compatible returned {response.status_code}: {error_body}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("llm.openai_invalid_json", error=str(exc))
            raise LLMProviderError("OpenAI-compatible returned invalid JSON") from exc

        choices = data.get("choices") or []
        if not choices:
            logger.error("llm.openai_no_choices", data_keys=list(data.keys()))
            raise LLMProviderError("OpenAI-compatible returned no choices")

        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            logger.error("llm.openai_empty_content", message_keys=list(message.keys()))
            raise LLMProviderError("OpenAI-compatible returned empty content")

        latency_ms = (time.perf_counter() - started) * 1000
        latency_ms_int = int(round(latency_ms))
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
            latency_ms=latency_ms_int,
        )

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream-complete a prompt using OpenAI-compatible API.

        Yields tokens as they are generated by the model.

        Args:
            prompt: User prompt to complete
            system_prompt: Optional system message

        Yields:
            Individual tokens as generated

        Raises:
            LLMProviderError: If provider returns an error
        """
        base_url = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": _build_messages(prompt, system_prompt),
            "max_tokens": 2048,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            try:
                async for chunk in _yield_stream_chunks(
                    client,
                    url=url,
                    payload=payload,
                    headers=_build_headers(self._settings),
                ):
                    yield chunk
            except httpx.RequestError as exc:
                logger.warning("llm.openai_stream_error", error=str(exc))
                raise LLMProviderError("OpenAI-compatible stream request failed") from exc
