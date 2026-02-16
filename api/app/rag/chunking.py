"""Document chunking for RAG."""

from __future__ import annotations


def chunk_text(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks, preferring sentence boundaries."""
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text) and chunk:
            # Prefer to split at sentence boundary (period or newline)
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            split_at = max(last_period, last_newline)
            if split_at >= 0 and split_at > chunk_size // 2:
                chunk = text[start : start + split_at + 1]
                start = start + split_at + 1 - chunk_overlap
            else:
                start = end - chunk_overlap
        else:
            start = len(text)

        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks
