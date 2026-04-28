"""Authentication tests.

Tests auth paths without dependency_overrides — the real require_auth runs.
Uses test JWT secret and patches _build_user_caller for user context tests.
"""

import base64
import time

import jwt
import pytest
from unittest.mock import patch, MagicMock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.auth import CallerContext
from tests.conftest import make_fake_agent

# ── Test constants ──────────────────────────────────────────────────────

TEST_JWT_SECRET_B64 = base64.b64encode(b"test-secret-key-at-least-32-bytes!!").decode()
TEST_API_KEY = "test-api-key-for-auth-tests"

AGENT_COMPANY_100 = make_fake_agent(9999, "test-tenant", company_id=100)
AGENT_COMPANY_200 = make_fake_agent(9998, "test-tenant-2", company_id=200)

AGENT_MAP = {
    9999: AGENT_COMPANY_100,
    9998: AGENT_COMPANY_200,
}


# ── JWT helper ──────────────────────────────────────────────────────────

def _make_jwt(user_id, expired=False):
    """Create a signed JWT with the test secret."""
    now = int(time.time())
    payload = {
        "id": user_id,
        "sub": "test@example.com",
        "iat": now,
        "exp": now + (-3600 if expired else 3600),
    }
    secret = base64.b64decode(TEST_JWT_SECRET_B64)
    return jwt.encode(payload, secret, algorithm="HS256")


def _make_jwt_custom(payload, secret_b64=TEST_JWT_SECRET_B64):
    """Create a JWT with custom payload and optional different secret."""
    secret = base64.b64decode(secret_b64)
    return jwt.encode(payload, secret, algorithm="HS256")


# ── Mock helpers ────────────────────────────────────────────────────────

def _mock_lookup(db, agent_id):
    return AGENT_MAP.get(agent_id)


def _patch_user_caller(company_id, is_super_admin=False):
    """Patch _build_user_caller to return a CallerContext without DB access."""
    def _fake_build(user_id, auth_type, db):
        return CallerContext(
            auth_type=auth_type,
            user_id=user_id,
            company_id=company_id,
            is_super_admin=is_super_admin,
        )
    return patch("app.api.auth._build_user_caller", side_effect=_fake_build)


def _patch_user_caller_not_found():
    """Patch _build_user_caller to raise 401 (user not found/inactive)."""
    def _fake_build(user_id, auth_type, db):
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return patch("app.api.auth._build_user_caller", side_effect=_fake_build)


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def auth_client():
    """Test client with real auth (no dependency_overrides for require_auth).

    Patches settings and agent lookup. DB user lookups are mocked per-test
    via _patch_user_caller.
    """
    mock_reader = MagicMock()
    mock_reader.get_document_chunks.return_value = []
    mock_reader.get_entity_chunk_mappings.return_value = []
    mock_reader.get_entity_relationships.return_value = []

    with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
         patch("app.graph_db.neo4j_reader.Neo4jReader"), \
         patch("app.api.endpoints.review_api._lookup_agent", side_effect=_mock_lookup), \
         patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup), \
         patch("app.core.config.settings.X_API_KEY", TEST_API_KEY), \
         patch("app.core.config.settings.JWT_SECRET", TEST_JWT_SECRET_B64), \
         patch("app.cache_db.redis_client.is_redis_configured", return_value=True), \
         patch("app.cache_db.redis_client.is_token_blacklisted", return_value=False), \
         patch("app.cache_db.redis_client.is_token_invalidated", return_value=False):
        from app.main import app
        with TestClient(app) as c:
            yield c


# ── Tests ───────────────────────────────────────────────────────────────

class TestNoAuth:
    """Requests without any credentials."""

    def test_health_no_auth(self, auth_client):
        """Health endpoint requires no auth."""
        resp = auth_client.get("/review-api/health")
        assert resp.status_code == 200

    def test_reviews_no_auth(self, auth_client):
        """Protected endpoints return 401 without credentials."""
        resp = auth_client.get("/review-api/reviews")
        assert resp.status_code == 401
        assert "Authentication required" in resp.json()["detail"]

    def test_post_review_no_auth(self, auth_client):
        """POST reviews returns 401 without credentials."""
        resp = auth_client.post("/review-api/reviews", json={"agent_id": 9999})
        assert resp.status_code == 401


class TestApiKeyAuth:
    """API key authentication (service-to-service)."""

    def test_valid_api_key(self, auth_client):
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 200

    def test_invalid_api_key(self, auth_client):
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403
        assert "Invalid API key" in resp.json()["detail"]

    def test_api_key_bypasses_company_check(self, auth_client):
        """API key callers can access any agent regardless of company."""
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 202


class TestJwtAuth:
    """Direct JWT authentication."""

    def test_expired_token(self, auth_client):
        token = _make_jwt(user_id=1, expired=True)
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"].lower()

    def test_malformed_bearer(self, auth_client):
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"Authorization": "NotBearer token"},
        )
        assert resp.status_code == 401
        assert "Bearer" in resp.json()["detail"]

    def test_empty_bearer(self, auth_client):
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"Authorization": "Bearer "},
        )
        assert resp.status_code == 401

    def test_valid_jwt(self, auth_client):
        """Valid JWT with mocked user lookup succeeds."""
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100):
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200

    def test_jwt_inactive_user(self, auth_client):
        """Valid JWT but user is inactive -> 401."""
        token = _make_jwt(user_id=1)
        with _patch_user_caller_not_found():
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 401
        assert "not found or inactive" in resp.json()["detail"]


class TestGatewayAuth:
    """Gateway-forwarded X-User-Id authentication (requires API key)."""

    def test_api_key_plus_user_id(self, auth_client):
        """Valid API key + X-User-Id → Gateway-forwarded auth."""
        with _patch_user_caller(company_id=100):
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"X-API-Key": TEST_API_KEY, "X-User-Id": "42"},
            )
        assert resp.status_code == 200

    def test_bare_user_id_rejected(self, auth_client):
        """X-User-Id without API key is rejected (prevents spoofing)."""
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"X-User-Id": "42"},
        )
        assert resp.status_code == 401
        assert "X-API-Key" in resp.json()["detail"]

    def test_invalid_user_id(self, auth_client):
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"X-API-Key": TEST_API_KEY, "X-User-Id": "not-a-number"},
        )
        assert resp.status_code == 400
        assert "integer" in resp.json()["detail"].lower()

    def test_user_id_inactive(self, auth_client):
        with _patch_user_caller_not_found():
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"X-API-Key": TEST_API_KEY, "X-User-Id": "999"},
            )
        assert resp.status_code == 401

    def test_gateway_user_company_scoped(self, auth_client):
        """Gateway-forwarded user is company-scoped, not full access."""
        with _patch_user_caller(company_id=100):
            resp = auth_client.post(
                "/review-api/reviews",
                json={"agent_id": 9998},  # company_id=200
                headers={"X-API-Key": TEST_API_KEY, "X-User-Id": "42"},
            )
        assert resp.status_code == 404


class TestCompanyIsolation:
    """JWT/Gateway users can only access their own company's agents."""

    def test_own_company_post(self, auth_client):
        """User with company_id=100 can start review for agent with company_id=100."""
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100):
            resp = auth_client.post(
                "/review-api/reviews",
                json={"agent_id": 9999},  # company_id=100
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 202

    def test_other_company_post(self, auth_client):
        """User with company_id=100 cannot start review for agent with company_id=200."""
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100):
            resp = auth_client.post(
                "/review-api/reviews",
                json={"agent_id": 9998},  # company_id=200
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 404

    def test_own_company_get_review(self, auth_client):
        """User can GET a review belonging to their company's agent."""
        # First create a review with API key
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999},  # company_id=100
            headers={"X-API-Key": TEST_API_KEY},
        )
        review_id = resp.json()["id"]

        # Now access it as a company_id=100 user
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100):
            resp = auth_client.get(
                f"/review-api/reviews/{review_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200

    def test_other_company_get_review(self, auth_client):
        """User cannot GET a review belonging to another company's agent."""
        # Create a review for agent 9999 (company_id=100)
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999},
            headers={"X-API-Key": TEST_API_KEY},
        )
        review_id = resp.json()["id"]

        # Try to access as a company_id=200 user — gets 404 (not 403) to
        # avoid leaking that the review exists.
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=200):
            resp = auth_client.get(
                f"/review-api/reviews/{review_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 404

    def test_other_company_get_issues(self, auth_client):
        """User cannot GET issues for a review belonging to another company's agent."""
        # Create a review for agent 9999 (company_id=100)
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999},
            headers={"X-API-Key": TEST_API_KEY},
        )
        review_id = resp.json()["id"]

        # Try to access issues as a company_id=200 user — gets 404
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=200):
            resp = auth_client.get(
                f"/review-api/reviews/{review_id}/issues",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 404

    def test_list_reviews_filtered_by_agent_wrong_company(self, auth_client):
        """User cannot list reviews for another company's agent."""
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100):
            resp = auth_client.get(
                "/review-api/reviews?agent_id=9998",  # company_id=200
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 404


class TestSuperAdmin:
    """Super admins bypass company restrictions."""

    def test_super_admin_post_cross_company(self, auth_client):
        """Super admin with company_id=100 can start review for company_id=200 agent."""
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100, is_super_admin=True):
            resp = auth_client.post(
                "/review-api/reviews",
                json={"agent_id": 9998},  # company_id=200
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 202

    def test_super_admin_get_cross_company(self, auth_client):
        """Super admin can GET a review belonging to another company's agent."""
        # Create a review for agent 9998 (company_id=200) with API key
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9998},
            headers={"X-API-Key": TEST_API_KEY},
        )
        review_id = resp.json()["id"]

        # Access as super admin with company_id=100
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100, is_super_admin=True):
            resp = auth_client.get(
                f"/review-api/reviews/{review_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200


class TestAuthPriority:
    """When multiple auth headers are present, priority applies."""

    def test_api_key_wins_over_jwt(self, auth_client):
        """X-API-Key takes priority even when JWT is also present."""
        token = _make_jwt(user_id=1, expired=True)  # expired — would fail
        resp = auth_client.get(
            "/review-api/reviews",
            headers={
                "X-API-Key": TEST_API_KEY,
                "Authorization": f"Bearer {token}",
            },
        )
        assert resp.status_code == 200

    def test_api_key_plus_user_id_is_company_scoped(self, auth_client):
        """API key + X-User-Id gives company-scoped access, not full access."""
        with _patch_user_caller(company_id=100):
            # Can access own company's agent
            resp = auth_client.post(
                "/review-api/reviews",
                json={"agent_id": 9999},  # company_id=100
                headers={"X-API-Key": TEST_API_KEY, "X-User-Id": "42"},
            )
        assert resp.status_code == 202

        with _patch_user_caller(company_id=100):
            # Cannot access other company's agent — gets 404 (not 403)
            resp = auth_client.post(
                "/review-api/reviews",
                json={"agent_id": 9998},  # company_id=200
                headers={"X-API-Key": TEST_API_KEY, "X-User-Id": "42"},
            )
        assert resp.status_code == 404


class TestApiKeyFullAccess:
    """API key callers (service-to-service) can access all GET endpoints."""

    def test_api_key_get_review(self, auth_client):
        """API key caller can GET /reviews/{id} for any agent."""
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999},
            headers={"X-API-Key": TEST_API_KEY},
        )
        review_id = resp.json()["id"]

        resp = auth_client.get(
            f"/review-api/reviews/{review_id}",
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 200

    def test_api_key_get_issues(self, auth_client):
        """API key caller can GET /reviews/{id}/issues for any agent."""
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999},
            headers={"X-API-Key": TEST_API_KEY},
        )
        review_id = resp.json()["id"]

        resp = auth_client.get(
            f"/review-api/reviews/{review_id}/issues",
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 200


class TestJwtEdgeCases:
    """JWT validation edge cases."""

    def test_jwt_missing_id_claim(self, auth_client):
        """JWT without 'id' claim → 401."""
        token = _make_jwt_custom({
            "sub": "test@example.com",
            "exp": int(time.time()) + 3600,
        })
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401
        assert "id" in resp.json()["detail"].lower()

    def test_jwt_non_integer_id(self, auth_client):
        """JWT with non-integer 'id' claim → 401."""
        token = _make_jwt_custom({
            "id": "not-an-int",
            "sub": "test@example.com",
            "exp": int(time.time()) + 3600,
        })
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401

    def test_jwt_non_integer_iat(self, auth_client):
        """JWT with non-integer 'iat' claim → 401 (PyJWT rejects it)."""
        token = _make_jwt_custom({
            "id": 1,
            "sub": "test@example.com",
            "iat": "not-a-number",
            "exp": int(time.time()) + 3600,
        })
        with _patch_user_caller(company_id=100):
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 401

    def test_jwt_wrong_secret(self, auth_client):
        """JWT signed with a different secret → 401."""
        wrong_secret = base64.b64encode(b"wrong-secret-key-at-least-32-bytes!!").decode()
        token = _make_jwt_custom(
            {"id": 1, "sub": "test@example.com", "exp": int(time.time()) + 3600},
            secret_b64=wrong_secret,
        )
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401


class TestCompanyIsolationGetIssues:
    """Positive test: own-company user can access their review's issues."""

    def test_own_company_get_issues(self, auth_client):
        """User can GET issues for a review belonging to their company's agent."""
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999},  # company_id=100
            headers={"X-API-Key": TEST_API_KEY},
        )
        review_id = resp.json()["id"]

        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100):
            resp = auth_client.get(
                f"/review-api/reviews/{review_id}/issues",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200


class TestSuperAdminFullAccess:
    """Super admins bypass company restrictions on all endpoints."""

    def test_super_admin_get_issues_cross_company(self, auth_client):
        """Super admin can GET issues for another company's review."""
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9998},  # company_id=200
            headers={"X-API-Key": TEST_API_KEY},
        )
        review_id = resp.json()["id"]

        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100, is_super_admin=True):
            resp = auth_client.get(
                f"/review-api/reviews/{review_id}/issues",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200

    def test_super_admin_list_all_reviews(self, auth_client):
        """Super admin listing reviews without agent_id sees all companies."""
        # Create reviews for both companies
        auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999},  # company_id=100
            headers={"X-API-Key": TEST_API_KEY},
        )
        auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9998},  # company_id=200
            headers={"X-API-Key": TEST_API_KEY},
        )

        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100, is_super_admin=True):
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200
        agent_ids = {r["agent_id"] for r in resp.json()}
        # Super admin sees reviews for BOTH companies
        assert 9999 in agent_ids
        assert 9998 in agent_ids


class TestAutoFiltering:
    """List endpoint auto-filters by company for non-service callers.

    Tests the agent_id-param company isolation path. The auto-filter subquery
    path (listing without agent_id) hits the real Agents table and is not
    covered here.
    """
    # TODO: The auto-filter subquery path (no agent_id param, JWT caller)
    # lacks test coverage — it queries Agents.company_id which requires
    # real DB rows or a more involved mock setup.

    def test_list_auto_filtered_by_company(self, auth_client):
        """JWT user with agent_id param only sees own company's reviews."""
        # Ensure reviews exist for both companies
        auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999},  # company_id=100
            headers={"X-API-Key": TEST_API_KEY},
        )
        auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9998},  # company_id=200
            headers={"X-API-Key": TEST_API_KEY},
        )

        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100):
            # With agent_id — uses _lookup_agent mock
            resp_own = auth_client.get(
                "/review-api/reviews?agent_id=9999",
                headers={"Authorization": f"Bearer {token}"},
            )
            resp_other = auth_client.get(
                "/review-api/reviews?agent_id=9998",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp_own.status_code == 200
        assert any(r["agent_id"] == 9999 for r in resp_own.json())
        # Cross-company agent returns 404 (not 403, to avoid info leak)
        assert resp_other.status_code == 404


class TestConcurrencyGuard:
    """Prevent duplicate active reviews for the same agent."""

    def test_duplicate_review_rejected(self, auth_client):
        """Second POST for same agent while active review exists → 409."""
        from app.database_layer.db_config import SessionLocal
        from app.database_layer.db_models import KbReview

        # Patch run_review to no-op so the review stays at PENDING
        with patch("app.api.endpoints.review_api.run_review"):
            resp = auth_client.post(
                "/review-api/reviews",
                json={"agent_id": 9999},
                headers={"X-API-Key": TEST_API_KEY},
            )
            assert resp.status_code == 202
            review_id = resp.json()["id"]

            # Second POST — should be rejected
            resp = auth_client.post(
                "/review-api/reviews",
                json={"agent_id": 9999},
                headers={"X-API-Key": TEST_API_KEY},
            )
            assert resp.status_code == 409
            assert "already has an active review" in resp.json()["detail"]

        # Cleanup: mark the stuck review as COMPLETED
        db = SessionLocal()
        try:
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            review.status = "COMPLETED"
            db.commit()
        finally:
            db.close()


class TestAuditTrail:
    """Review records track who started them."""

    def test_created_by_populated_for_jwt_caller(self, auth_client):
        """JWT caller's user_id is stored in created_by_user_id."""
        token = _make_jwt(user_id=42)
        with _patch_user_caller(company_id=100):
            resp = auth_client.post(
                "/review-api/reviews",
                json={"agent_id": 9999},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 202
        assert resp.json()["created_by_user_id"] == 42

    def test_created_by_null_for_api_key_caller(self, auth_client):
        """API key caller has no user_id → created_by_user_id is null."""
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 202
        assert resp.json()["created_by_user_id"] is None

    def test_created_by_populated_for_gateway_caller(self, auth_client):
        """Gateway (X-API-Key + X-User-Id) caller's user_id is stored."""
        with _patch_user_caller(company_id=100):
            resp = auth_client.post(
                "/review-api/reviews",
                json={"agent_id": 9999},
                headers={"X-API-Key": TEST_API_KEY, "X-User-Id": "55"},
            )
        assert resp.status_code == 202
        assert resp.json()["created_by_user_id"] == 55


class TestPatchIssueStatusAuth:
    """Auth-specific tests for the PATCH /reviews/{id}/issues/{id} endpoint."""

    def test_jwt_caller_patch_populates_status_updated_by(self, auth_client):
        """JWT caller's user_id is stored in status_updated_by on PATCH."""
        from app.database_layer.db_config import SessionLocal
        from app.database_layer.db_models import KbReview, KbReviewIssue

        db = SessionLocal()
        review_id = None
        try:
            # Create review + issue directly in DB (API key, so it completes fast)
            review = KbReview(agent_id=9999, status="COMPLETED", progress=100, issues_found=1)
            db.add(review)
            db.flush()
            review_id = review.id
            issue = KbReviewIssue(
                review_id=review.id, issue_type="CONTRADICTION", severity="HIGH",
                confidence=0.9, title="Test", consensus="UNANIMOUS",
                judges_flagged=3, judges_total=3,
            )
            db.add(issue)
            db.commit()
            db.refresh(issue)
            issue_id = issue.id

            # PATCH as JWT user_id=42
            token = _make_jwt(user_id=42)
            with _patch_user_caller(company_id=100):
                resp = auth_client.patch(
                    f"/review-api/reviews/{review_id}/issues/{issue_id}",
                    json={"status": "RESOLVED", "note": "Fixed"},
                    headers={"Authorization": f"Bearer {token}"},
                )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "RESOLVED"
            assert data["status_updated_by"] == 42
            assert data["status_updated_at"] is not None
            assert data["status_note"] == "Fixed"
        finally:
            if review_id:
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()

    def test_cross_company_patch_returns_404(self, auth_client):
        """User from company_id=200 cannot PATCH issues on company_id=100 review."""
        from app.database_layer.db_config import SessionLocal
        from app.database_layer.db_models import KbReview, KbReviewIssue

        db = SessionLocal()
        review_id = None
        try:
            # Create review for agent 9999 (company_id=100)
            review = KbReview(agent_id=9999, status="COMPLETED", progress=100, issues_found=1)
            db.add(review)
            db.flush()
            review_id = review.id
            issue = KbReviewIssue(
                review_id=review.id, issue_type="CONTRADICTION", severity="HIGH",
                confidence=0.9, title="Test", consensus="UNANIMOUS",
                judges_flagged=3, judges_total=3,
            )
            db.add(issue)
            db.commit()
            db.refresh(issue)
            issue_id = issue.id

            # PATCH as company_id=200 user — should get 404 (not 403)
            token = _make_jwt(user_id=1)
            with _patch_user_caller(company_id=200):
                resp = auth_client.patch(
                    f"/review-api/reviews/{review_id}/issues/{issue_id}",
                    json={"status": "RESOLVED"},
                    headers={"Authorization": f"Bearer {token}"},
                )
            assert resp.status_code == 404
        finally:
            if review_id:
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()

    def test_cross_company_patch_active_review_returns_404_not_409(self, auth_client):
        """Cross-company PATCH on RUNNING review returns 404, not 409 (no info leak)."""
        from app.database_layer.db_config import SessionLocal
        from app.database_layer.db_models import KbReview, KbReviewIssue

        db = SessionLocal()
        review_id = None
        try:
            # Create a RUNNING review for agent 9999 (company_id=100)
            review = KbReview(agent_id=9999, status="RUNNING", progress=50, issues_found=1)
            db.add(review)
            db.flush()
            review_id = review.id
            issue = KbReviewIssue(
                review_id=review.id, issue_type="CONTRADICTION", severity="HIGH",
                confidence=0.9, title="Test", consensus="UNANIMOUS",
                judges_flagged=3, judges_total=3,
            )
            db.add(issue)
            db.commit()
            db.refresh(issue)
            issue_id = issue.id

            # PATCH as company_id=200 user on a RUNNING review
            # Must get 404 (not 409) to avoid leaking that the review is active
            token = _make_jwt(user_id=1)
            with _patch_user_caller(company_id=200):
                resp = auth_client.patch(
                    f"/review-api/reviews/{review_id}/issues/{issue_id}",
                    json={"status": "RESOLVED"},
                    headers={"Authorization": f"Bearer {token}"},
                )
            assert resp.status_code == 404, (
                f"Expected 404 but got {resp.status_code} — "
                "cross-company PATCH on active review must not leak 409"
            )
        finally:
            if review_id:
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()


class TestUnconfiguredAuth:
    """Server returns 500 when auth env vars are missing."""

    def test_missing_api_key_config(self):
        """Empty X_API_KEY env var → 500 when caller sends an API key."""
        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = []
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
             patch("app.graph_db.neo4j_reader.Neo4jReader"), \
             patch("app.api.endpoints.review_api._lookup_agent", side_effect=_mock_lookup), \
             patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup), \
             patch("app.core.config.settings.X_API_KEY", ""), \
             patch("app.core.config.settings.JWT_SECRET", TEST_JWT_SECRET_B64):
            from app.main import app
            with TestClient(app) as c:
                resp = c.get(
                    "/review-api/reviews",
                    headers={"X-API-Key": "any-key"},
                )
        assert resp.status_code == 500
        assert "not configured" in resp.json()["detail"]

    def test_missing_jwt_secret_config(self):
        """Empty JWT_SECRET env var → 500 when caller sends a JWT."""
        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = []
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
             patch("app.graph_db.neo4j_reader.Neo4jReader"), \
             patch("app.api.endpoints.review_api._lookup_agent", side_effect=_mock_lookup), \
             patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup), \
             patch("app.core.config.settings.X_API_KEY", TEST_API_KEY), \
             patch("app.core.config.settings.JWT_SECRET", ""):
            from app.main import app
            with TestClient(app) as c:
                token = _make_jwt(user_id=1)
                resp = c.get(
                    "/review-api/reviews",
                    headers={"Authorization": f"Bearer {token}"},
                )
        assert resp.status_code == 500
        assert "not configured" in resp.json()["detail"]


class TestRequestBodyValidation:
    """Pydantic validation returns 422 for invalid request bodies."""

    def test_similarity_threshold_too_high(self, auth_client):
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999, "similarity_threshold": 1.5},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 422

    def test_max_candidate_pairs_too_low(self, auth_client):
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999, "max_candidate_pairs": 0},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 422

    def test_max_candidate_pairs_too_high(self, auth_client):
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999, "max_candidate_pairs": 501},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 422

    def test_missing_agent_id(self, auth_client):
        resp = auth_client.post(
            "/review-api/reviews",
            json={},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 422

    def test_invalid_analysis_types(self, auth_client):
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999, "analysis_types": ["CONTRADICTION", "MADE_UP_TYPE"]},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 422
        assert "MADE_UP_TYPE" in resp.text

    def test_empty_analysis_types(self, auth_client):
        resp = auth_client.post(
            "/review-api/reviews",
            json={"agent_id": 9999, "analysis_types": []},
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert resp.status_code == 422


class TestApiKeyWhitespace:
    """API key with leading/trailing whitespace should still authenticate."""

    def test_api_key_with_whitespace(self, auth_client):
        resp = auth_client.get(
            "/review-api/reviews",
            headers={"X-API-Key": f"  {TEST_API_KEY}  "},
        )
        assert resp.status_code == 200


class TestStartupRecovery:
    """Tests for _recover_stale_reviews() on startup."""

    def test_pending_review_recovered_as_failed(self, auth_client):
        """PENDING review from before process start → FAILED."""
        from datetime import datetime, timedelta
        from app.database_layer.db_config import SessionLocal
        from app.database_layer.db_models import KbReview

        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(
                agent_id=9999, status="PENDING", progress=0, issues_found=0,
            )
            db.add(review)
            db.commit()
            db.refresh(review)
            review_id = review.id

            # Backdate created_at to simulate old review
            db.execute(
                __import__("sqlalchemy").text(
                    "UPDATE kb_reviews SET created_at = :ts WHERE id = :id"
                ),
                {"ts": datetime.now() - timedelta(hours=1), "id": review.id},
            )
            db.commit()

            from app.main import _recover_stale_reviews
            import app.main
            old_start = app.main._PROCESS_START
            app.main._PROCESS_START = datetime.now()
            try:
                _recover_stale_reviews()
            finally:
                app.main._PROCESS_START = old_start

            db.expire_all()
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            assert review.status == "FAILED"
            assert review.completed_at is not None
            assert review.error_message is not None

        finally:
            if review_id:
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()

    def test_running_review_with_issues_recovered_as_partial(self, auth_client):
        """RUNNING review with persisted findings → PARTIAL."""
        from datetime import datetime, timedelta
        from app.database_layer.db_config import SessionLocal
        from app.database_layer.db_models import KbReview, KbReviewIssue

        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(
                agent_id=9999, status="RUNNING", progress=50, issues_found=1,
                started_at=datetime.now() - timedelta(hours=1),
            )
            db.add(review)
            db.flush()
            review_id = review.id

            issue = KbReviewIssue(
                review_id=review.id, issue_type="CONTRADICTION", severity="HIGH",
                confidence=0.9, title="Found something", consensus="UNANIMOUS",
                judges_flagged=3, judges_total=3,
            )
            db.add(issue)
            db.commit()

            from app.main import _recover_stale_reviews
            import app.main
            old_start = app.main._PROCESS_START
            app.main._PROCESS_START = datetime.now()
            try:
                _recover_stale_reviews()
            finally:
                app.main._PROCESS_START = old_start

            db.expire_all()
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            assert review.status == "PARTIAL"
            assert review.issues_found == 1
            assert review.completed_at is not None

        finally:
            if review_id:
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()

    def test_recent_running_review_not_recovered(self, auth_client):
        """RUNNING review started AFTER process start → left alone."""
        from datetime import datetime
        from app.database_layer.db_config import SessionLocal
        from app.database_layer.db_models import KbReview

        db = SessionLocal()
        review_id = None
        try:
            from app.main import _recover_stale_reviews
            import app.main
            # Set process start to 1 hour ago
            old_start = app.main._PROCESS_START
            app.main._PROCESS_START = datetime.now()

            # Create review AFTER process start
            import time
            time.sleep(0.1)
            review = KbReview(
                agent_id=9999, status="RUNNING", progress=30, issues_found=0,
                started_at=datetime.now(),
            )
            db.add(review)
            db.commit()
            db.refresh(review)
            review_id = review.id

            try:
                _recover_stale_reviews()
            finally:
                app.main._PROCESS_START = old_start

            db.expire_all()
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            assert review.status == "RUNNING", "Recent review should not be recovered"

        finally:
            if review_id:
                # Clean up — mark completed so it doesn't block concurrency guard
                review = db.query(KbReview).filter(KbReview.id == review_id).first()
                if review:
                    review.status = "COMPLETED"
                    db.commit()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()

    def test_minority_excluded_from_recovered_count(self, auth_client):
        """Recovered review's issues_found should exclude MINORITY."""
        from datetime import datetime, timedelta
        from app.database_layer.db_config import SessionLocal
        from app.database_layer.db_models import KbReview, KbReviewIssue

        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(
                agent_id=9999, status="RUNNING", progress=80, issues_found=0,
                started_at=datetime.now() - timedelta(hours=1),
            )
            db.add(review)
            db.flush()
            review_id = review.id

            db.add_all([
                KbReviewIssue(
                    review_id=review.id, issue_type="CONTRADICTION", severity="HIGH",
                    confidence=0.9, title="Unanimous", consensus="UNANIMOUS",
                    judges_flagged=3, judges_total=3,
                ),
                KbReviewIssue(
                    review_id=review.id, issue_type="CONTRADICTION", severity="LOW",
                    confidence=0.3, title="Minority", consensus="MINORITY",
                    judges_flagged=1, judges_total=3,
                ),
            ])
            db.commit()

            from app.main import _recover_stale_reviews
            import app.main
            old_start = app.main._PROCESS_START
            app.main._PROCESS_START = datetime.now()
            try:
                _recover_stale_reviews()
            finally:
                app.main._PROCESS_START = old_start

            db.expire_all()
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            assert review.status == "PARTIAL"
            assert review.issues_found == 1, "MINORITY should be excluded from count"

        finally:
            if review_id:
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()


class TestRedisBlacklist:
    """Redis token blacklist and session invalidation checks."""

    def test_blacklisted_token_rejected(self, auth_client):
        """Blacklisted token returns 401."""
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100), \
             patch("app.cache_db.redis_client.is_token_blacklisted", return_value=True):
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 401
        assert "revoked" in resp.json()["detail"].lower()

    def test_invalidated_session_rejected(self, auth_client):
        """Token issued before session invalidation returns 401."""
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100), \
             patch("app.cache_db.redis_client.is_token_blacklisted", return_value=False), \
             patch("app.cache_db.redis_client.is_token_invalidated", return_value=True):
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 401
        assert "security update" in resp.json()["detail"].lower()

    def test_blacklist_checked_before_invalidation(self, auth_client):
        """Both blacklisted AND invalidated → blacklist 401 wins (checked first)."""
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100), \
             patch("app.cache_db.redis_client.is_token_blacklisted", return_value=True), \
             patch("app.cache_db.redis_client.is_token_invalidated", return_value=True):
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 401
        assert "revoked" in resp.json()["detail"].lower()

    def test_redis_unavailable_returns_503(self, auth_client):
        """Redis connection failure returns 503 (fail-closed)."""
        from app.cache_db.redis_client import RedisCheckError
        token = _make_jwt(user_id=1)
        with _patch_user_caller(company_id=100), \
             patch("app.cache_db.redis_client.is_token_blacklisted",
                   side_effect=RedisCheckError("connection refused")):
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    def test_redis_not_configured_dev_allows_jwt(self):
        """In dev mode, missing Redis config warns but allows JWT through."""
        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = []
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
             patch("app.graph_db.neo4j_reader.Neo4jReader"), \
             patch("app.api.endpoints.review_api._lookup_agent", side_effect=_mock_lookup), \
             patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup), \
             patch("app.core.config.settings.X_API_KEY", TEST_API_KEY), \
             patch("app.core.config.settings.JWT_SECRET", TEST_JWT_SECRET_B64), \
             patch("app.core.config.settings.APP_ENV", "dev"), \
             patch("app.cache_db.redis_client.is_redis_configured", return_value=False):
            from app.main import app
            with TestClient(app) as c:
                token = _make_jwt(user_id=1)
                with _patch_user_caller(company_id=100):
                    resp = c.get(
                        "/review-api/reviews",
                        headers={"Authorization": f"Bearer {token}"},
                    )
        assert resp.status_code == 200

    def test_redis_not_configured_production_returns_503(self):
        """In production, missing Redis config returns 503."""
        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = []
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
             patch("app.graph_db.neo4j_reader.Neo4jReader"), \
             patch("app.api.endpoints.review_api._lookup_agent", side_effect=_mock_lookup), \
             patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup), \
             patch("app.core.config.settings.X_API_KEY", TEST_API_KEY), \
             patch("app.core.config.settings.JWT_SECRET", TEST_JWT_SECRET_B64), \
             patch("app.core.config.settings.APP_ENV", "production"), \
             patch("app.cache_db.redis_client.is_redis_configured", return_value=False):
            from app.main import app
            with TestClient(app) as c:
                token = _make_jwt(user_id=1)
                resp = c.get(
                    "/review-api/reviews",
                    headers={"Authorization": f"Bearer {token}"},
                )
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    def test_api_key_path_unaffected_by_redis(self, auth_client):
        """API key auth works even when Redis is down."""
        with patch("app.cache_db.redis_client.is_redis_configured", return_value=False):
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"X-API-Key": TEST_API_KEY},
            )
        assert resp.status_code == 200

    def test_gateway_path_unaffected_by_redis(self, auth_client):
        """Gateway auth (X-API-Key + X-User-Id) works even when Redis is down."""
        with patch("app.cache_db.redis_client.is_redis_configured", return_value=False), \
             _patch_user_caller(company_id=100):
            resp = auth_client.get(
                "/review-api/reviews",
                headers={"X-API-Key": TEST_API_KEY, "X-User-Id": "42"},
            )
        assert resp.status_code == 200
