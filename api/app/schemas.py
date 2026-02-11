from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    id: str = Field(..., min_length=1, description="Unique document ID")
    title: str
    text: str


class DocumentRead(BaseModel):
    id: str
    title: str
    text: str
    created_at: datetime


class NotarySummarizeRequest(BaseModel):
    document_id: Optional[str] = None
    text: str
    language: Literal["nl", "en"] = "nl"


class NotarySummary(BaseModel):
    title: str
    key_points: list[str]
    parties_involved: list[str]
    risks_or_warnings: list[str]
    raw_summary: str


class NotarySummarizeResponse(BaseModel):
    document_id: Optional[str]
    summary: NotarySummary
    source: Literal["llm", "fallback"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassifyRequest(BaseModel):
    text: str
    candidate_labels: list[str] = Field(
        default_factory=lambda: ["contract", "letter", "invoice", "report", "other"],
    )


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    model: str
    source: Literal["llm", "fallback"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class AskRequest(BaseModel):
    question: str
    context: str


class AskResponse(BaseModel):
    answer: str
    model: str
    source: Literal["llm", "fallback"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthStatus(BaseModel):
    status: Literal["ok"] = "ok"
    environment: str
    timestamp: datetime
    db_ok: Optional[bool] = None
    llm_ok: Optional[bool] = None
