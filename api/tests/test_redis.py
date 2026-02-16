"""Tests for Redis module (cache, rate limit, ping)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.redis import (
    cache_key,
    check_rate_limit,
    close_redis,
    get_cached,
    ping_redis,
    set_cached,
)


@pytest.mark.asyncio
async def test_cache_key():
    """cache_key returns formatted key."""
    assert cache_key("t1", "document", "doc-1") == "cache:t1:document:doc-1"


@pytest.mark.asyncio
async def test_ping_redis_not_configured():
    """ping_redis returns None when Redis not configured."""
    with patch("app.core.redis.get_settings") as mock_settings:
        mock_settings.return_value.redis_url = None
        with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            result = await ping_redis()
            assert result is None


@pytest.mark.asyncio
async def test_ping_redis_success():
    """ping_redis returns True when Redis is reachable."""
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_client
        result = await ping_redis()
        assert result is True


@pytest.mark.asyncio
async def test_ping_redis_failure():
    """ping_redis returns False when Redis raises."""
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(side_effect=ConnectionError("Connection refused"))
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_client
        result = await ping_redis()
        assert result is False


@pytest.mark.asyncio
async def test_get_cached_not_configured():
    """get_cached returns None when Redis not configured."""
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        result = await get_cached("key")
        assert result is None


@pytest.mark.asyncio
async def test_get_cached_hit():
    """get_cached returns value when key exists."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value='{"id":"d1"}')
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_client
        result = await get_cached("cache:t1:document:d1")
        assert result == '{"id":"d1"}'


@pytest.mark.asyncio
async def test_get_cached_exception_returns_none():
    """get_cached returns None on Redis exception."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=ConnectionError("Redis down"))
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_client
        result = await get_cached("key")
        assert result is None


@pytest.mark.asyncio
async def test_set_cached_not_configured():
    """set_cached does nothing when Redis not configured."""
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        await set_cached("key", "value", ttl_seconds=300)
        mock_get.assert_called_once()


@pytest.mark.asyncio
async def test_set_cached_success():
    """set_cached calls setex when Redis configured."""
    mock_client = AsyncMock()
    mock_client.setex = AsyncMock(return_value=True)
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_client
        await set_cached("key", "value", ttl_seconds=300)
        mock_client.setex.assert_called_once_with("key", 300, "value")


@pytest.mark.asyncio
async def test_check_rate_limit_no_redis_returns_true():
    """check_rate_limit returns True when Redis not configured (allow all)."""
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = None
        result = await check_rate_limit("t1", limit=10, window_seconds=60)
        assert result is True


@pytest.mark.asyncio
async def test_check_rate_limit_under_limit():
    """check_rate_limit returns True when under limit."""
    mock_client = MagicMock()
    pipe = MagicMock()
    pipe.zremrangebyscore = MagicMock(return_value=pipe)
    pipe.zadd = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    mock_client.pipeline.return_value = pipe
    mock_client.zrem = AsyncMock()
    pipe.execute = AsyncMock(return_value=[None, None, 3])  # count=3
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_client
        result = await check_rate_limit("t1", limit=10, window_seconds=60)
        assert result is True


@pytest.mark.asyncio
async def test_check_rate_limit_over_limit():
    """check_rate_limit returns False when over limit."""
    mock_client = MagicMock()
    pipe = MagicMock()
    pipe.zremrangebyscore = MagicMock(return_value=pipe)
    pipe.zadd = MagicMock(return_value=pipe)
    pipe.zcard = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    mock_client.pipeline.return_value = pipe
    mock_client.zrem = AsyncMock()
    # count=15 > limit 10 -> returns False
    pipe.execute = AsyncMock(return_value=[None, None, 15])
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_client
        result = await check_rate_limit("t1", limit=10, window_seconds=60)
        assert result is False


@pytest.mark.asyncio
async def test_check_rate_limit_exception_returns_true():
    """check_rate_limit returns True on exception (fail open)."""
    mock_client = AsyncMock()
    mock_client.pipeline.side_effect = ConnectionError("Redis down")
    with patch("app.core.redis.get_redis", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_client
        result = await check_rate_limit("t1", limit=10, window_seconds=60)
        assert result is True


@pytest.mark.asyncio
async def test_close_redis():
    """close_redis closes client and sets to None."""
    mock_client = AsyncMock()
    with patch("app.core.redis._redis", mock_client):
        await close_redis()
        mock_client.aclose.assert_called_once()
