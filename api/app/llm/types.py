from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResult:
    raw_text: str
    model: str
    latency_ms: float
    used_fallback: bool = False
