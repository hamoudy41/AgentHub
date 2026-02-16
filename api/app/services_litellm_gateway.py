"""LiteLLM gateway for multi-provider LLM routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from litellm import acompletion


class LiteLLMGatewayError(Exception):
    """Raised when LiteLLM completion fails."""

    pass


@dataclass
class LiteLLMResult:
    """Result from LiteLLM completion."""

    raw_text: str
    model: str
    total_tokens: int = 0


class LiteLLMGateway:
    """Gateway for LLM completion via LiteLLM (supports OpenAI, Anthropic, Ollama, etc.)."""

    async def complete(
        self,
        prompt: str,
        *,
        model: str,
        system_prompt: Optional[str] = None,
    ) -> LiteLLMResult:
        """Complete a prompt using the specified model (e.g. ollama/llama3.2, openai/gpt-4)."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            response = await acompletion(model=model, messages=messages)
        except Exception as e:
            raise LiteLLMGatewayError(str(e)) from e
        choices = getattr(response, "choices", []) or []
        if not choices:
            raise LiteLLMGatewayError("LiteLLM returned no choices")
        msg = getattr(choices[0], "message", None) or choices[0].get("message", {})
        content = (
            getattr(msg, "content", None)
            or (msg.get("content") if isinstance(msg, dict) else None)
            or ""
        )
        model_name = getattr(response, "model", None) or str(model)
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0
        return LiteLLMResult(
            raw_text=str(content).strip() if content else "",
            model=model_name,
            total_tokens=total_tokens,
        )


litellm_gateway = LiteLLMGateway()
