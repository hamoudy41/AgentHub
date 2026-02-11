from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from redis.asyncio import Redis

from .config import get_settings


_redis: Redis | None = None


async def get_redis() -> Redis | None:
    global _redis
    settings = get_settings()
    url = getattr(settings, "redis_url", None)
    if not url:
        return None
    if _redis is None:
        _redis = Redis.from_url(str(url), decode_responses=True)
    return _redis


@asynccontextmanager
async def redis_session() -> AsyncGenerator[Redis | None, None]:
    client = await get_redis()
    try:
        yield client
    finally:
        pass


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


async def ping_redis() -> bool | None:
    """Return True if Redis is reachable, False if configured but unreachable, None if not configured."""
    client = await get_redis()
    if not client:
        return None
    try:
        await client.ping()
        return True
    except Exception:
        return False


def _rate_limit_key(tenant_id: str) -> str:
    return f"rl:{tenant_id}"


def cache_key(tenant_id: str, resource: str, resource_id: str) -> str:
    return f"cache:{tenant_id}:{resource}:{resource_id}"


async def check_rate_limit(tenant_id: str, limit: int, window_seconds: int = 60) -> bool:
    client = await get_redis()
    if not client:
        return True
    key = _rate_limit_key(tenant_id)
    try:
        now = int(time.time())
        window_start = now - window_seconds
        member = f"{now}:{uuid.uuid4().hex}"
        pipe = client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zadd(key, {member: now})
        pipe.zcard(key)
        pipe.expire(key, window_seconds + 1)
        results = await pipe.execute()
        count = results[2]
        if count > limit:
            await client.zrem(key, member)
            return False
        return True
    except Exception:
        return True


async def get_cached(key_prefix: str) -> str | None:
    client = await get_redis()
    if not client:
        return None
    try:
        return await client.get(key_prefix)
    except Exception:
        return None


async def set_cached(key_prefix: str, value: str, ttl_seconds: int = 300) -> None:
    client = await get_redis()
    if not client:
        return
    try:
        await client.setex(key_prefix, ttl_seconds, value)
    except Exception:
        pass
