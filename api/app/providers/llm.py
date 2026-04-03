"""LLM provider base class and types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from app.core.errors import AppError


@dataclass(frozen=True)
class LLMResult:
    """Result from LLM completion."""

    raw_text: str
    model: str
    latency_ms: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


class LLMError(AppError):
    """Base error for LLM operations."""

    def __init__(self, message: str, error_code: str = "LLM_ERROR", status_code: int = 503) -> None:
        super().__init__(message, error_code=error_code, status_code=status_code)


class LLMTimeoutError(LLMError):
    """LLM request timed out."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "LLM_TIMEOUT")


class LLMProviderError(LLMError):
    """LLM provider returned an error."""

    def __init__(self, message: str, provider: str = "") -> None:
        super().__init__(
            message,
            error_code="LLM_PROVIDER_ERROR",
            status_code=503,
        )


class LLMNotConfiguredError(LLMError):
    """LLM is not properly configured."""

    def __init__(self, message: str) -> None:
        super().__init__(message, "LLM_NOT_CONFIGURED", status_code=400)


class LLMProvider(ABC):
    """Abstract base for LLM providers (Ollama, OpenAI, etc.).

    All LLM providers must implement this interface to be used
    with the LLMService. This enables provider-agnostic code that
    can work with any LLM backend.
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> LLMResult:
        """Complete a prompt and return result.

        Args:
            prompt: User prompt to complete
            system_prompt: Optional system message/context

        Returns:
            LLMResult with completion text and metadata

        Raises:
            LLMProviderError: If provider returns an error
        """
        pass

    @abstractmethod
    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream-complete a prompt, yielding tokens.

        Args:
            prompt: User prompt to complete
            system_prompt: Optional system message/context

        Yields:
            Individual tokens as they are generated

        Raises:
            LLMProviderError: If provider returns an error
        """
        pass
