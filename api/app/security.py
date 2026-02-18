"""Security utilities for input validation and prompt injection detection."""

from __future__ import annotations

import re
from typing import Optional

from .core.logging import get_logger

logger = get_logger(__name__)

# Patterns that indicate potential prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+(instructions|commands|prompts)",
    r"disregard\s+(all\s+)?previous\s+(instructions|commands|prompts)",
    r"forget\s+(all\s+)?previous\s+(instructions|commands|prompts)",
    r"ignore\s+the\s+above",
    r"disregard\s+the\s+above",
    r"new\s+instructions?:",
    r"system\s*:\s*",
    r"assistant\s*:\s*",
    r"<\|.*?\|>",  # Special tokens
    r"\[INST\]",  # Llama instruction markers
    r"\[/INST\]",
    r"###\s*instruction",
    r"###\s*system",
]

_COMPILED_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in _INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> tuple[bool, Optional[str]]:
    """
    Detect potential prompt injection attacks in user input.

    Args:
        text: User input text to check

    Returns:
        Tuple of (is_suspicious, matched_pattern)
        - is_suspicious: True if injection detected
        - matched_pattern: The pattern that matched, or None
    """
    if not text:
        return False, None

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(text):
            return True, pattern.pattern

    return False, None


def sanitize_user_input(
    text: str,
    *,
    max_length: Optional[int] = None,
    check_injection: bool = True,
    tenant_id: str = "default",
) -> str:
    """
    Sanitize user input for use in LLM prompts.

    Args:
        text: User input text
        max_length: Maximum allowed length
        check_injection: Whether to check for prompt injection
        tenant_id: Tenant ID for logging

    Returns:
        Sanitized text

    Raises:
        ValueError: If input fails validation
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")

    text = text.strip()

    if max_length and len(text) > max_length:
        logger.warning(
            "security.input_too_long",
            tenant_id=tenant_id,
            length=len(text),
            max_length=max_length,
        )
        raise ValueError(f"Input exceeds maximum length of {max_length} characters")

    if check_injection:
        is_suspicious, pattern = detect_prompt_injection(text)
        if is_suspicious:
            logger.warning(
                "security.potential_injection_detected",
                tenant_id=tenant_id,
                pattern=pattern,
                input_preview=text[:100],
            )
            raise ValueError("Input contains potentially malicious content")

    return text


def sanitize_for_logging(value: str, max_length: int = 200) -> str:
    """
    Sanitize a value for safe logging (truncate, remove sensitive patterns).

    Args:
        value: Value to sanitize
        max_length: Maximum length to log

    Returns:
        Sanitized value safe for logging
    """
    if not value:
        return ""

    # Truncate
    if len(value) > max_length:
        value = value[:max_length] + "..."

    # Remove potential API keys (simple pattern)
    value = re.sub(r"\bsk-[a-zA-Z0-9]{40,}\b", "[REDACTED_API_KEY]", value)
    value = re.sub(r"\bBearer\s+[a-zA-Z0-9._-]+", "Bearer [REDACTED]", value)

    return value
