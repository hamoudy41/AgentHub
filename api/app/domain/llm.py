"""Domain interfaces and entities for LLM interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass(frozen=True)
class LLMRequest:
    """Request to an LLM provider."""

    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    tenant_id: str = "default"


@dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM provider."""

    text: str
    model: str
    latency_ms: float
    provider: str
    used_fallback: bool = False


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

    pass


class ILLMProvider(ABC):
    """Interface for LLM providers."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion for the given request."""
        pass

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion for the given request."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        pass
