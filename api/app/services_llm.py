from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpen
from .core.config import get_settings
from .core.logging import get_logger
from .security import sanitize_for_logging


logger = get_logger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    pass


class LLMNotConfiguredError(LLMError):
    """Raised when LLM is not properly configured."""

    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""

    pass


class LLMProviderError(LLMError):
    """Raised when LLM provider returns an error."""

    def __init__(self, message: str, status_code: int | None = None, provider: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


@dataclass
class LLMResult:
    raw_text: str
    model: str
    latency_ms: float
    used_fallback: bool = False


def _get_retries() -> int:
    return max(1, get_settings().llm_max_retries)


class LLMClient:
    def __init__(self) -> None:
        self._settings = get_settings()
        # Initialize circuit breakers for each provider
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            "ollama": CircuitBreaker(
                "llm_ollama",
                CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=30.0,
                    timeout_seconds=self._settings.llm_timeout_seconds,
                ),
            ),
            "openai": CircuitBreaker(
                "llm_openai",
                CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=30.0,
                    timeout_seconds=self._settings.llm_timeout_seconds,
                ),
            ),
        }

    def is_configured(self) -> bool:
        return bool(self._settings.llm_base_url and self._settings.llm_provider)

    def get_circuit_breaker_status(self) -> dict[str, any]:
        """Get status of all circuit breakers for monitoring."""
        return {
            provider: cb.get_state()
            for provider, cb in self._circuit_breakers.items()
        }

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        tenant_id: str = "default",
        timeout: Optional[float] = None,
    ) -> LLMResult:
        if not self.is_configured():
            raise LLMNotConfiguredError(
                "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
            )
        
        # Get appropriate circuit breaker
        provider_key = "ollama" if self._settings.llm_provider == "ollama" else "openai"
        circuit_breaker = self._circuit_breakers[provider_key]
        
        # Check circuit breaker
        can_execute, reason = circuit_breaker.can_execute()
        if not can_execute:
            logger.error(
                "llm.circuit_breaker_open",
                provider=self._settings.llm_provider,
                reason=reason,
                tenant_id=tenant_id,
            )
            raise CircuitBreakerOpen(reason or "Circuit breaker is open")
        
        try:
            if self._settings.llm_provider == "ollama":
                result = await self._complete_ollama(
                    prompt, system_prompt=system_prompt, timeout=timeout
                )
            else:
                result = await self._complete_openai(
                    prompt, system_prompt=system_prompt, timeout=timeout
                )
            
            # Record success
            circuit_breaker.record_success()
            return result
            
        except (LLMError, httpx.RequestError, asyncio.TimeoutError) as e:
            # Record failure
            circuit_breaker.record_failure()
            raise

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(_get_retries()),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _complete_ollama(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        base = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base}/api/generate"
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 512},
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        timeout_value = timeout if timeout is not None else self._settings.llm_timeout_seconds
        started = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                r = await client.post(url, json=payload)
        except httpx.TimeoutException as e:
            logger.error(
                "llm.ollama_timeout",
                error=str(e),
                timeout=timeout_value,
                prompt_preview=sanitize_for_logging(prompt, 100),
            )
            raise LLMTimeoutError(f"Ollama request timed out after {timeout_value}s") from e
        except httpx.RequestError as e:
            logger.error(
                "llm.ollama_request_error",
                error=str(e),
                error_type=type(e).__name__,
                prompt_preview=sanitize_for_logging(prompt, 100),
            )
            raise LLMProviderError("Ollama request failed", provider="ollama") from e
            
        if r.status_code != 200:
            error_body = sanitize_for_logging(r.text, 500)
            logger.error(
                "llm.ollama_error_response",
                status_code=r.status_code,
                response_preview=error_body,
            )
            raise LLMProviderError(
                f"Ollama returned {r.status_code}: {error_body}",
                status_code=r.status_code,
                provider="ollama",
            )
            
        data = r.json()
        text = data.get("response") or data.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            logger.error("llm.ollama_empty_response", data_keys=list(data.keys()))
            raise LLMProviderError("Ollama returned empty response", provider="ollama")
            
        latency_ms = (time.perf_counter() - started) * 1000
        
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

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(_get_retries()),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        reraise=True,
    )
    async def _complete_openai(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> LLMResult:
        base = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base}/v1/chat/completions"
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "max_tokens": 2048,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self._settings.llm_api_key}"
        
        timeout_value = timeout if timeout is not None else self._settings.llm_timeout_seconds
        started = time.perf_counter()
        
        try:
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                r = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as e:
            logger.error(
                "llm.openai_timeout",
                error=str(e),
                timeout=timeout_value,
                prompt_preview=sanitize_for_logging(prompt, 100),
            )
            raise LLMTimeoutError(f"OpenAI request timed out after {timeout_value}s") from e
        except httpx.RequestError as e:
            logger.error(
                "llm.openai_request_error",
                error=str(e),
                error_type=type(e).__name__,
                prompt_preview=sanitize_for_logging(prompt, 100),
            )
            raise LLMProviderError("OpenAI-compatible request failed", provider="openai") from e
            
        if r.status_code != 200:
            error_body = sanitize_for_logging(r.text, 500)
            logger.error(
                "llm.openai_error_response",
                status_code=r.status_code,
                response_preview=error_body,
            )
            raise LLMProviderError(
                f"OpenAI-compatible returned {r.status_code}: {error_body}",
                status_code=r.status_code,
                provider="openai",
            )
            
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            logger.error("llm.openai_no_choices", data_keys=list(data.keys()))
            raise LLMProviderError("OpenAI-compatible returned no choices", provider="openai")
            
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            logger.error("llm.openai_empty_content", message_keys=list(msg.keys()))
            raise LLMProviderError("OpenAI-compatible returned empty content", provider="openai")
            
        latency_ms = (time.perf_counter() - started) * 1000
        
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

    async def generate_notary_summary(self, prompt: str, *, tenant_id: str) -> LLMResult:
        return await self.complete(
            prompt,
            system_prompt="You are a concise assistant for notarial document summarization. Reply only with the summary, no preamble.",
            tenant_id=tenant_id,
        )

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        tenant_id: str = "default",
    ) -> AsyncIterator[str]:
        """Stream LLM tokens one chunk at a time. Raises LLMNotConfiguredError if not configured."""
        if not self.is_configured():
            raise LLMNotConfiguredError(
                "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
            )
        if self._settings.llm_provider == "ollama":
            async for chunk in self._stream_ollama(prompt, system_prompt=system_prompt):
                yield chunk
        else:
            async for chunk in self._stream_openai(prompt, system_prompt=system_prompt):
                yield chunk

    async def _stream_ollama(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        base = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base}/api/generate"
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "prompt": prompt,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt
        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            try:
                async with client.stream("POST", url, json=payload) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        raise LLMError(f"Ollama returned {r.status_code}: {body.decode()[:500]}")
                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk = data.get("response") or data.get("text") or ""
                        if isinstance(chunk, str) and chunk:
                            yield chunk
            except httpx.RequestError as e:
                logger.warning("llm.ollama_stream_error", error=str(e))
                raise LLMError("Ollama stream request failed") from e

    async def _stream_openai(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        base = str(self._settings.llm_base_url).rstrip("/")
        url = f"{base}/v1/chat/completions"
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "max_tokens": 2048,
            "stream": True,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self._settings.llm_api_key}"
        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        raise LLMError(
                            f"OpenAI-compatible returned {r.status_code}: {body.decode()[:500]}"
                        )
                    async for line in r.aiter_lines():
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
            except httpx.RequestError as e:
                logger.warning("llm.openai_stream_error", error=str(e))
                raise LLMError("OpenAI-compatible stream request failed") from e


llm_client = LLMClient()
