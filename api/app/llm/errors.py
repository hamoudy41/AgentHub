from __future__ import annotations


class LLMError(Exception):
    """Base exception for LLM-related errors."""


class LLMNotConfiguredError(LLMError):
    """Raised when LLM is not properly configured."""


class LLMTimeoutError(LLMError):
    """Raised when LLM requests exceed the configured timeout."""


class LLMProviderError(LLMError):
    """Raised when the backing LLM provider returns an invalid or failed response."""

    def __init__(self, message: str, status_code: int | None = None, provider: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
