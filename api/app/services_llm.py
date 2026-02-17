from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .core.config import get_settings
from .core.logging import get_logger


logger = get_logger(__name__)


class LLMError(Exception):
    pass


class LLMNotConfiguredError(LLMError):
    pass


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

    def is_configured(self) -> bool:
        return bool(self._settings.llm_base_url and self._settings.llm_provider)

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        tenant_id: str = "default",
    ) -> LLMResult:
        if not self.is_configured():
            raise LLMNotConfiguredError(
                "LLM not configured. Set LLM_PROVIDER and LLM_BASE_URL (e.g. ollama + http://localhost:11434)."
            )
        if self._settings.llm_provider == "ollama":
            return await self._complete_ollama(prompt, system_prompt=system_prompt)
        return await self._complete_openai(prompt, system_prompt=system_prompt)

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(_get_retries()),
        reraise=True,
    )
    async def _complete_ollama(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
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
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            try:
                r = await client.post(url, json=payload)
            except httpx.RequestError as e:
                logger.warning("llm.ollama_error", error=str(e))
                raise LLMError("Ollama request failed") from e
        if r.status_code != 200:
            raise LLMError(f"Ollama returned {r.status_code}: {r.text[:500]}")
        data = r.json()
        text = data.get("response") or data.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            raise LLMError("Ollama returned empty response")
        latency_ms = (time.perf_counter() - started) * 1000
        return LLMResult(
            raw_text=text.strip(),
            model=data.get("model", self._settings.llm_model),
            latency_ms=round(latency_ms, 2),
        )

    @retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(_get_retries()),
        reraise=True,
    )
    async def _complete_openai(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
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
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self._settings.llm_timeout_seconds) as client:
            try:
                r = await client.post(url, json=payload, headers=headers)
            except httpx.RequestError as e:
                logger.warning("llm.openai_error", error=str(e))
                raise LLMError("OpenAI-compatible request failed") from e
        if r.status_code != 200:
            raise LLMError(f"OpenAI-compatible returned {r.status_code}: {r.text[:500]}")
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("OpenAI-compatible returned no choices")
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if not isinstance(text, str) or not text.strip():
            raise LLMError("OpenAI-compatible returned empty content")
        latency_ms = (time.perf_counter() - started) * 1000
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
