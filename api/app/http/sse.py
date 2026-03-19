from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any

import orjson


def sse_event(payload: Mapping[str, Any]) -> bytes:
    return f"data: {orjson.dumps(dict(payload)).decode()}\n\n".encode()


async def stream_text_tokens(source: AsyncIterable[str]) -> AsyncIterator[bytes]:
    try:
        async for chunk in source:
            yield sse_event({"token": chunk})
    except Exception as exc:  # noqa: BLE001
        yield sse_event({"error": str(exc), "done": True})
        return

    yield sse_event({"done": True})
