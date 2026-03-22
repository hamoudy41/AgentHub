from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

import httpx

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpen
from .core.config import get_settings
from .core.logging import get_logger
from .llm.errors import LLMError, LLMNotConfiguredError, LLMProviderError, LLMTimeoutError
from .llm.providers import OllamaProvider, OpenAICompatibleProvider
from .llm.types import LLMResult


logger = get_logger(__name__)
__all__ = [
    "LLMClient",
    "LLMError",
    "LLMNotConfiguredError",
    "LLMProviderError",
    "LLMResult",
    "LLMTimeoutError",
    "llm_client",
]


class LLMClient:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._providers = {
            "ollama": OllamaProvider(self._settings),
            "openai_compatible": OpenAICompatibleProvider(self._settings),
        }
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

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        return {provider: cb.get_state() for provider, cb in self._circuit_breakers.items()}

    def _provider_key(self) -> str:
        return "ollama" if self._settings.llm_provider == "ollama" else "openai_compatible"

    def _circuit_breaker_key(self) -> str:
        return "ollama" if self._settings.llm_provider == "ollama" else "openai"

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

        provider_key = self._provider_key()
        circuit_breaker = self._circuit_breakers[self._circuit_breaker_key()]
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
            result = await self._providers[provider_key].complete(
                prompt,
                system_prompt=system_prompt,
                timeout=timeout,
            )
            circuit_breaker.record_success()
            return result
        except (LLMError, httpx.RequestError, asyncio.TimeoutError):
            circuit_breaker.record_failure()
            raise

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
        if not self.is_configured():
            raise LLMNotConfiguredError(
                "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
            )
        async for chunk in self._providers[self._provider_key()].stream_complete(
            prompt,
            system_prompt=system_prompt,
        ):
            yield chunk


llm_client = LLMClient()
