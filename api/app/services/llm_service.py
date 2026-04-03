"""LLM service: unified interface for language model operations."""

from __future__ import annotations

from typing import AsyncIterator, Optional

from app.core.context import ExecutionContext
from app.providers.llm import LLMProvider, LLMResult

from .base_service import BaseService


class LLMService(BaseService):
    """Service for LLM operations with provider abstraction.

    Provides a unified interface for LLM completions, supporting
    both blocking and streaming modes with automatic provider selection.

    Attributes:
        provider: The LLM provider implementation to use
    """

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize LLMService with a provider.

        Args:
            provider: LLMProvider implementation (Ollama, OpenAI, etc.)
        """
        super().__init__("llm")
        self.provider = provider

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        context: ExecutionContext | None = None,
    ) -> LLMResult:
        """Complete a prompt using the configured provider."""
        return await self.provider.complete(
            prompt,
            system_prompt=system_prompt,
        )

    async def stream_complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        context: ExecutionContext | None = None,
    ) -> AsyncIterator[str]:
        """Stream-complete a prompt."""
        async for token in self.provider.stream_complete(
            prompt,
            system_prompt=system_prompt,
        ):
            yield token
