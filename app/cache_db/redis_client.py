"""Redis client for token blacklist and session invalidation checks.

Mirrors the Java Gateway's RedisService for compatibility:
- BL_{token} → token blacklist (EXISTS check)
- TOKEN_VERSION:{userId} → session invalidation timestamp (GET + compare)
"""

import logging
from typing import Optional

import redis

from app.core.config import settings

logger = logging.getLogger("kb_review")

BLACKLIST_PREFIX = "BL_"
TOKEN_VERSION_PREFIX = "TOKEN_VERSION:"

_redis_client: Optional[redis.Redis] = None


class RedisCheckError(Exception):
    """Raised when Redis is unavailable and we fail-closed."""
    pass


def _get_client() -> Optional[redis.Redis]:
    """Return the Redis client singleton, or None if not configured."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if not settings.REDIS_HOST:
        return None
    _redis_client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=0.5,
        socket_timeout=0.2,
        max_connections=20,
    )
    return _redis_client


def is_redis_configured() -> bool:
    """True if REDIS_HOST is set."""
    return bool(settings.REDIS_HOST)


def check_redis_health() -> str:
    """Return 'connected', 'disconnected', or 'not_configured'."""
    if not is_redis_configured():
        return "not_configured"
    try:
        client = _get_client()
        if client and client.ping():
            return "connected"
    except Exception:
        pass
    return "disconnected"


def is_token_blacklisted(token: str) -> bool:
    """Check if token is in the Redis blacklist (BL_{token} key).

    Returns True if blacklisted, False if not.
    Raises RedisCheckError if Redis is unavailable (fail-closed).
    """
    client = _get_client()
    if client is None:
        raise RedisCheckError("Redis not configured")
    key = BLACKLIST_PREFIX + token
    result = client.exists(key)
    if result:
        logger.info("Token is blacklisted (Redis key exists)")
    return bool(result)


def is_token_invalidated(user_id: int, iat_seconds: Optional[int]) -> bool:
    """Check if token was issued before a session invalidation event.

    Compares the JWT iat (issued-at, in seconds) against the Redis
    TOKEN_VERSION:{userId} timestamp (in milliseconds, written by Auth Service).

    Returns True if token is invalidated, False if valid.
    Raises RedisCheckError if Redis is unavailable (fail-closed).
    """
    client = _get_client()
    if client is None:
        raise RedisCheckError("Redis not configured")
    if iat_seconds is None:
        logger.warning(f"JWT missing iat claim for user_id={user_id} — skipping session invalidation check")
        return False

    key = TOKEN_VERSION_PREFIX + str(user_id)
    timestamp_str = client.get(key)
    if timestamp_str is None:
        return False  # No invalidation record → token is valid

    try:
        invalidation_time_ms = int(timestamp_str)
    except (ValueError, TypeError):
        logger.error(f"Invalid timestamp in Redis for key {key}: {timestamp_str} — treating as invalidated (fail-closed)")
        return True

    # Convert JWT iat (seconds) to milliseconds to match Java's System.currentTimeMillis()
    iat_ms = iat_seconds * 1000
    is_invalid = iat_ms < invalidation_time_ms
    if is_invalid:
        logger.info(
            f"Token invalidated | user_id={user_id} | iat_ms={iat_ms} "
            f"| invalidation_ms={invalidation_time_ms}"
        )
    return is_invalid
