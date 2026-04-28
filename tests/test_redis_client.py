"""Unit tests for Redis client functions (mocked Redis, no real connection)."""

import pytest
from unittest.mock import patch, MagicMock

from app.cache_db.redis_client import (
    is_token_blacklisted,
    is_token_invalidated,
    check_redis_health,
    RedisCheckError,
)


class TestIsTokenBlacklisted:

    def test_blacklisted_token(self):
        mock_client = MagicMock()
        mock_client.exists.return_value = 1
        with patch("app.cache_db.redis_client._get_client", return_value=mock_client):
            assert is_token_blacklisted("abc123") is True
        mock_client.exists.assert_called_once_with("BL_abc123")

    def test_not_blacklisted(self):
        mock_client = MagicMock()
        mock_client.exists.return_value = 0
        with patch("app.cache_db.redis_client._get_client", return_value=mock_client):
            assert is_token_blacklisted("abc123") is False

    def test_redis_not_configured(self):
        with patch("app.cache_db.redis_client._get_client", return_value=None):
            with pytest.raises(RedisCheckError):
                is_token_blacklisted("abc123")


class TestIsTokenInvalidated:

    def test_token_issued_before_invalidation(self):
        """iat=1000s (1_000_000ms) < invalidation=2_000_000ms → invalidated."""
        mock_client = MagicMock()
        mock_client.get.return_value = "2000000"  # ms
        with patch("app.cache_db.redis_client._get_client", return_value=mock_client):
            assert is_token_invalidated(42, iat_seconds=1000) is True
        mock_client.get.assert_called_once_with("TOKEN_VERSION:42")

    def test_token_issued_after_invalidation(self):
        """iat=3000s (3_000_000ms) > invalidation=2_000_000ms → valid."""
        mock_client = MagicMock()
        mock_client.get.return_value = "2000000"
        with patch("app.cache_db.redis_client._get_client", return_value=mock_client):
            assert is_token_invalidated(42, iat_seconds=3000) is False

    def test_token_issued_at_exact_invalidation_time(self):
        """iat=2000s (2_000_000ms) == invalidation=2_000_000ms → valid (not <)."""
        mock_client = MagicMock()
        mock_client.get.return_value = "2000000"
        with patch("app.cache_db.redis_client._get_client", return_value=mock_client):
            assert is_token_invalidated(42, iat_seconds=2000) is False

    def test_no_invalidation_record(self):
        """No TOKEN_VERSION key → token is valid."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        with patch("app.cache_db.redis_client._get_client", return_value=mock_client):
            assert is_token_invalidated(42, iat_seconds=1000) is False

    def test_no_iat_claim(self):
        """Missing iat → treated as valid (logged as warning)."""
        mock_client = MagicMock()
        mock_client.get.return_value = "2000000"
        with patch("app.cache_db.redis_client._get_client", return_value=mock_client):
            assert is_token_invalidated(42, iat_seconds=None) is False
        # Should not even query Redis when iat is None
        mock_client.get.assert_not_called()

    def test_malformed_redis_timestamp(self):
        """Non-numeric Redis value → treated as invalidated (fail-closed)."""
        mock_client = MagicMock()
        mock_client.get.return_value = "not-a-number"
        with patch("app.cache_db.redis_client._get_client", return_value=mock_client):
            assert is_token_invalidated(42, iat_seconds=1000) is True

    def test_redis_not_configured(self):
        with patch("app.cache_db.redis_client._get_client", return_value=None):
            with pytest.raises(RedisCheckError):
                is_token_invalidated(42, iat_seconds=1000)


class TestCheckRedisHealth:

    def test_connected(self):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        with patch("app.cache_db.redis_client.is_redis_configured", return_value=True), \
             patch("app.cache_db.redis_client._get_client", return_value=mock_client):
            assert check_redis_health() == "connected"

    def test_disconnected(self):
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("refused")
        with patch("app.cache_db.redis_client.is_redis_configured", return_value=True), \
             patch("app.cache_db.redis_client._get_client", return_value=mock_client):
            assert check_redis_health() == "disconnected"

    def test_not_configured(self):
        with patch("app.cache_db.redis_client.is_redis_configured", return_value=False):
            assert check_redis_health() == "not_configured"
