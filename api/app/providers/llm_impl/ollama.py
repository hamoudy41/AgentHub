"""Ollama LLM provider implementation."""

from __future__ import annotations

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


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation.

    Communicates with Ollama instances via HTTP API for text generation.
    Supports both streaming and non-streaming completions.

    Args:
        settings: Application settings with llm_base_url, llm_model, etc.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize Ollama provider with settings."""
        self._settings = settings

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> LLMResult:
        """Complete a prompt using Ollama.

        Args:
            prompt: User prompt to complete
            system_prompt: Optional system message

        Returns:
            LLMResult with completion text and metadata

        Raises:
            LLMTimeoutError: If request times out
            LLMProviderError: If Ollama returns an error
        """
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

        timeout_seconds = self._settings.llm_timeout_seconds
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
            raise LLMProviderError("Ollama request failed") from exc

        if response.status_code != 200:
            error_body = sanitize_for_logging(response.text, 500)
            logger.error(
                "llm.ollama_error_response",
                status_code=response.status_code,
                response_preview=error_body,
            )
            raise LLMProviderError(f"Ollama returned {response.status_code}: {error_body}")

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("llm.ollama_invalid_json", error=str(exc))
            raise LLMProviderError("Ollama returned invalid JSON") from exc

        text = data.get("response") or data.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            logger.error("llm.ollama_empty_response", data_keys=list(data.keys()))
            raise LLMProviderError("Ollama returned empty response")

        latency_ms = (time.perf_counter() - started) * 1000
        latency_ms_int = int(round(latency_ms))
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
            latency_ms=latency_ms_int,
        )

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream-complete a prompt using Ollama.

        Yields tokens as they are generated by the model.

        Args:
            prompt: User prompt to complete
            system_prompt: Optional system message

        Yields:
            Individual tokens as generated

        Raises:
            LLMProviderError: If Ollama returns an error
        """
        import json

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
                        raise LLMProviderError(
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
                raise LLMProviderError("Ollama stream request failed") from exc
