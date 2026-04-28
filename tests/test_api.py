"""Integration tests for API endpoints.

Uses the MySQL database configured in .env.
Mocks Neo4j reader, LLM calls, and agent lookups (agents table is read-only).
"""

import json
import time
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.database_layer.db_config import SessionLocal
from app.database_layer.db_models import KbReview, KbReviewIssue, KbReviewJudgeResult
from tests.conftest import make_fake_agent

AGENT_WITH_TENANT = make_fake_agent(9999, "test-tenant-001", "Test Agent")
AGENT_NO_TENANT = make_fake_agent(9998, None, "No Tenant Agent")

AGENT_MAP = {
    9999: AGENT_WITH_TENANT,
    9998: AGENT_NO_TENANT,
}


def _mock_lookup(db, agent_id):
    return AGENT_MAP.get(agent_id)


@pytest.fixture(scope="module")
def client():
    """FastAPI test client. Patches neo4j_reader and agent lookup."""
    mock_reader = MagicMock()
    mock_reader.get_document_chunks.return_value = []
    mock_reader.get_entity_chunk_mappings.return_value = []
    mock_reader.get_entity_relationships.return_value = []

    with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
         patch("app.graph_db.neo4j_reader.Neo4jReader"), \
         patch("app.api.endpoints.review_api._lookup_agent", side_effect=_mock_lookup), \
         patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup):
        from app.main import app
        from app.api.auth import require_auth
        from tests.conftest import test_caller_override
        app.dependency_overrides[require_auth] = test_caller_override
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.pop(require_auth, None)


class TestHealthCheck:
    def test_health(self, client):
        resp = client.get("/review-api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "KB Review" in data["service"]


class TestStartReviewCustom:
    """Tests for custom mode (judges provided)."""

    def test_start_review_success(self, client):
        resp = client.post("/review-api/reviews", json={
            "agent_id": 9999,
            "judges": [{"provider": "litellm", "model": "test-model"}],
        })
        assert resp.status_code == 202
        data = resp.json()
        assert data["agent_id"] == 9999
        assert data["status"] == "PENDING"
        assert data["progress"] == 0
        assert "id" in data

    def test_agent_not_found(self, client):
        resp = client.post("/review-api/reviews", json={
            "agent_id": 777777,
            "judges": [{"provider": "litellm", "model": "test"}],
        })
        assert resp.status_code == 404

    def test_agent_no_tenant(self, client):
        resp = client.post("/review-api/reviews", json={
            "agent_id": 9998,
            "judges": [{"provider": "litellm", "model": "test"}],
        })
        assert resp.status_code == 400
        assert "tenant_id" in resp.json()["detail"]

    def test_invalid_provider(self, client):
        resp = client.post("/review-api/reviews", json={
            "agent_id": 9999,
            "judges": [{"provider": "invalid_provider", "model": "test"}],
        })
        assert resp.status_code == 400
        assert "invalid_provider" in resp.json()["detail"]

    def test_empty_judges_list(self, client):
        resp = client.post("/review-api/reviews", json={
            "agent_id": 9999,
            "judges": [],
        })
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_multiple_judges(self, client):
        resp = client.post("/review-api/reviews", json={
            "agent_id": 9999,
            "judges": [
                {"provider": "litellm", "model": "model-a"},
                {"provider": "fireworks", "model": "model-b"},
                {"provider": "openrouter", "model": "model-c"},
            ],
            "analysis_types": ["CONTRADICTION"],
            "similarity_threshold": 0.7,
            "max_candidate_pairs": 50,
        })
        assert resp.status_code == 202

    def test_custom_no_fallback_in_config(self, client):
        """Custom mode should NOT include fallback_judges in config."""
        resp = client.post("/review-api/reviews", json={
            "agent_id": 9999,
            "judges": [{"provider": "litellm", "model": "test"}],
        })
        review_id = resp.json()["id"]
        db = SessionLocal()
        try:
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            config = json.loads(review.config_json)
            assert "fallback_judges" not in config
        finally:
            db.close()


class TestStartReviewAuto:
    """Tests for auto mode (judges omitted)."""

    def test_auto_review_async(self, client):
        """POST /reviews with just agent_id returns 202."""
        resp = client.post("/review-api/reviews", json={"agent_id": 9999})
        assert resp.status_code == 202
        data = resp.json()
        assert data["agent_id"] == 9999
        assert data["status"] == "PENDING"

    def test_auto_review_stores_config(self, client):
        """Config JSON should contain primary judges and fallback_judges."""
        resp = client.post("/review-api/reviews", json={"agent_id": 9999})
        review_id = resp.json()["id"]

        db = SessionLocal()
        try:
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            config = json.loads(review.config_json)
            assert "judges" in config
            assert "fallback_judges" in config
            assert len(config["judges"]) == 3
            assert len(config["fallback_judges"]) == 3
            # Primary is Gemini, fallback is Haiku
            assert "gemini" in config["judges"][0]["model"]
            assert "haiku" in config["fallback_judges"][0]["model"]
        finally:
            db.close()

    def test_auto_agent_not_found(self, client):
        resp = client.post("/review-api/reviews", json={"agent_id": 777777})
        assert resp.status_code == 404

    def test_auto_agent_no_tenant(self, client):
        resp = client.post("/review-api/reviews", json={"agent_id": 9998})
        assert resp.status_code == 400


class TestAutoReviewSync:
    """Tests for sync mode (?wait=true)."""

    def test_sync_returns_200_on_completion(self, client):
        """With ?wait=true and empty KB, review completes immediately -> 200."""
        resp = client.post(
            "/review-api/reviews?wait=true",
            json={"agent_id": 9999},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "review" in data
        assert "issues" in data
        assert data["review"]["status"] == "COMPLETED"
        assert isinstance(data["issues"], list)

    def test_sync_custom_also_works(self, client):
        """Sync mode works with custom judges too."""
        resp = client.post(
            "/review-api/reviews?wait=true",
            json={
                "agent_id": 9999,
                "judges": [{"provider": "litellm", "model": "test"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["review"]["status"] == "COMPLETED"


class TestSyncTimeout:
    """Test that sync mode returns 202 on timeout instead of 500."""

    def test_timeout_returns_202(self, client):
        """When run_review takes longer than timeout, return 202."""
        def slow_run_review(review_id):
            time.sleep(3)
            # Update status so the review is "still running" from client's POV
            db = SessionLocal()
            try:
                review = db.query(KbReview).filter(KbReview.id == review_id).first()
                if review:
                    review.status = "COMPLETED"
                    review.progress = 100
                    db.commit()
            finally:
                db.close()

        with patch("app.api.endpoints.review_api.run_review", side_effect=slow_run_review), \
             patch("app.api.endpoints.review_api.SYNC_TIMEOUT_SECONDS", 1):
            resp = client.post(
                "/review-api/reviews?wait=true",
                json={"agent_id": 9999},
            )
            assert resp.status_code == 202
            data = resp.json()
            assert "id" in data
            # Review is still PENDING/RUNNING at this point (hasn't completed yet)

            # Wait for the background thread to finish
            time.sleep(4)

            # Now the review should be completed in DB
            review_id = data["id"]
            db = SessionLocal()
            try:
                review = db.query(KbReview).filter(KbReview.id == review_id).first()
                assert review.status == "COMPLETED"
            finally:
                db.close()


class TestGetReview:
    def test_get_review(self, client):
        create_resp = client.post("/review-api/reviews", json={
            "agent_id": 9999,
            "judges": [{"provider": "litellm", "model": "test"}],
        })
        review_id = create_resp.json()["id"]

        resp = client.get(f"/review-api/reviews/{review_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == review_id
        assert data["agent_id"] == 9999
        assert "issues_by_type" in data
        assert "issues_by_severity" in data

    def test_get_nonexistent_review(self, client):
        resp = client.get("/review-api/reviews/999999")
        assert resp.status_code == 404


class TestListReviews:
    def test_list_all(self, client):
        resp = client.get("/review-api/reviews")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_filter_by_agent(self, client):
        resp = client.get("/review-api/reviews?agent_id=9999")
        assert resp.status_code == 200
        data = resp.json()
        for item in data:
            assert item["agent_id"] == 9999

    def test_filter_nonexistent_agent(self, client):
        resp = client.get("/review-api/reviews?agent_id=888888")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetReviewIssues:
    def test_get_issues_empty(self, client):
        create_resp = client.post("/review-api/reviews", json={
            "agent_id": 9999,
            "judges": [{"provider": "litellm", "model": "test"}],
        })
        review_id = create_resp.json()["id"]

        resp = client.get(f"/review-api/reviews/{review_id}/issues")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_issues_nonexistent_review(self, client):
        resp = client.get("/review-api/reviews/999999/issues")
        assert resp.status_code == 404

    def test_filter_params_accepted(self, client):
        create_resp = client.post("/review-api/reviews", json={
            "agent_id": 9999,
            "judges": [{"provider": "litellm", "model": "test"}],
        })
        review_id = create_resp.json()["id"]

        resp = client.get(
            f"/review-api/reviews/{review_id}/issues"
            "?issue_type=CONTRADICTION&severity=HIGH&min_confidence=0.5"
        )
        assert resp.status_code == 200


class TestCarryForwardParams:
    """Test carry-forward query params and response fields."""

    def test_carryforward_true_stored_in_config(self, client):
        """Default carryforward=true is stored in config_json."""
        resp = client.post("/review-api/reviews", json={
            "agent_id": 9999,
            "judges": [{"provider": "litellm", "model": "test"}],
        })
        review_id = resp.json()["id"]
        db = SessionLocal()
        try:
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            config = json.loads(review.config_json)
            assert config["carryforward"] is True
        finally:
            db.close()

    def test_carryforward_false_stored_in_config(self, client):
        """?carryforward=false is stored in config_json."""
        resp = client.post(
            "/review-api/reviews?carryforward=false",
            json={
                "agent_id": 9999,
                "judges": [{"provider": "litellm", "model": "test"}],
            },
        )
        review_id = resp.json()["id"]
        db = SessionLocal()
        try:
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            config = json.loads(review.config_json)
            assert config["carryforward"] is False
        finally:
            db.close()

    def test_carried_forward_filter_on_issues(self, client):
        """?carried_forward= filters issues by carried_forward flag."""
        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(agent_id=9999, status="COMPLETED", progress=100, issues_found=2)
            db.add(review)
            db.flush()
            review_id = review.id

            # Fresh issue
            fresh = KbReviewIssue(
                review_id=review.id, issue_type="CONTRADICTION", severity="HIGH",
                confidence=0.9, title="Fresh finding", consensus="UNANIMOUS",
                judges_flagged=3, judges_total=3, carried_forward=False,
            )
            # Carried-forward issue
            carried = KbReviewIssue(
                review_id=review.id, issue_type="ENTITY_INCONSISTENCY", severity="MEDIUM",
                confidence=0.8, title="Carried forward finding", consensus="MAJORITY",
                judges_flagged=2, judges_total=3,
                carried_forward=True, original_review_id=1,
            )
            db.add_all([fresh, carried])
            db.commit()

            # Default: all issues
            resp = client.get(f"/review-api/reviews/{review.id}/issues")
            assert len(resp.json()) == 2

            # Only fresh
            resp_fresh = client.get(f"/review-api/reviews/{review.id}/issues?carried_forward=false")
            issues_fresh = resp_fresh.json()
            assert len(issues_fresh) == 1
            assert issues_fresh[0]["title"] == "Fresh finding"
            assert issues_fresh[0]["carried_forward"] is False

            # Only carried forward
            resp_cf = client.get(f"/review-api/reviews/{review.id}/issues?carried_forward=true")
            issues_cf = resp_cf.json()
            assert len(issues_cf) == 1
            assert issues_cf[0]["title"] == "Carried forward finding"
            assert issues_cf[0]["carried_forward"] is True
            assert issues_cf[0]["original_review_id"] == 1

        finally:
            if review_id:
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()

    def test_summary_includes_carry_forward_fields(self, client):
        """GET /reviews/{id} returns carry-forward summary fields."""
        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(
                agent_id=9999, status="COMPLETED", progress=100, issues_found=5,
                previous_review_id=42, pairs_reused=8, pairs_analyzed=2,
                docs_changed=1, docs_unchanged=9,
            )
            db.add(review)
            db.commit()
            db.refresh(review)
            review_id = review.id

            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["previous_review_id"] == 42
            assert data["pairs_reused"] == 8
            assert data["pairs_analyzed"] == 2
            assert data["docs_changed"] == 1
            assert data["docs_unchanged"] == 9

        finally:
            if review_id:
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()

    def test_sync_response_includes_carry_forward_fields(self, client):
        """Sync mode ?wait=true returns carry-forward fields in review summary."""
        resp = client.post(
            "/review-api/reviews?wait=true",
            json={"agent_id": 9999},
        )
        assert resp.status_code == 200
        data = resp.json()
        review = data["review"]
        # Empty KB: no previous, all zeros
        assert review["previous_review_id"] is None
        assert review["pairs_reused"] == 0
        assert review["pairs_analyzed"] == 0
        assert review["docs_changed"] == 0
        assert review["docs_unchanged"] == 0


class TestMinorityFiltering:
    """Test that MINORITY findings are excluded by default."""

    def test_minority_excluded_by_default(self, client):
        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(agent_id=9999, status="COMPLETED", progress=100, issues_found=2)
            db.add(review)
            db.flush()
            review_id = review.id

            # UNANIMOUS issue
            issue1 = KbReviewIssue(
                review_id=review.id, issue_type="CONTRADICTION", severity="HIGH",
                confidence=0.9, title="Unanimous finding", consensus="UNANIMOUS",
                judges_flagged=3, judges_total=3,
            )
            # MINORITY issue
            issue2 = KbReviewIssue(
                review_id=review.id, issue_type="CONTRADICTION", severity="LOW",
                confidence=0.4, title="Minority finding", consensus="MINORITY",
                judges_flagged=1, judges_total=3,
            )
            db.add_all([issue1, issue2])
            db.commit()

            # Default: MINORITY excluded
            resp = client.get(f"/review-api/reviews/{review.id}/issues")
            assert resp.status_code == 200
            issues = resp.json()
            assert len(issues) == 1
            assert issues[0]["title"] == "Unanimous finding"

            # With include_minority=true: both returned
            resp2 = client.get(f"/review-api/reviews/{review.id}/issues?include_minority=true")
            assert resp2.status_code == 200
            assert len(resp2.json()) == 2

        finally:
            if review_id:
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()

    def test_majority_included_by_default(self, client):
        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(agent_id=9999, status="COMPLETED", progress=100, issues_found=1)
            db.add(review)
            db.flush()
            review_id = review.id

            issue = KbReviewIssue(
                review_id=review.id, issue_type="ENTITY_INCONSISTENCY", severity="MEDIUM",
                confidence=0.7, title="Majority finding", consensus="MAJORITY",
                judges_flagged=2, judges_total=3,
            )
            db.add(issue)
            db.commit()

            resp = client.get(f"/review-api/reviews/{review.id}/issues")
            assert resp.status_code == 200
            assert len(resp.json()) == 1
            assert resp.json()[0]["consensus"] == "MAJORITY"

        finally:
            if review_id:
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()


class TestMinoritySummaryCounts:
    """Summary counts (issues_found, issues_by_type, issues_by_severity) should
    exclude MINORITY by default and match the issue list the caller sees."""

    def test_summary_excludes_minority_counts(self, client):
        """GET /reviews/{id} summary counts should exclude MINORITY findings."""
        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(agent_id=9999, status="COMPLETED", progress=100, issues_found=3)
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
                    review_id=review.id, issue_type="ENTITY_INCONSISTENCY", severity="MEDIUM",
                    confidence=0.7, title="Majority", consensus="MAJORITY",
                    judges_flagged=2, judges_total=3,
                ),
                KbReviewIssue(
                    review_id=review.id, issue_type="CONTRADICTION", severity="LOW",
                    confidence=0.3, title="Minority", consensus="MINORITY",
                    judges_flagged=1, judges_total=3,
                ),
            ])
            db.commit()

            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.status_code == 200
            data = resp.json()
            # issues_found should exclude MINORITY (2, not 3)
            assert data["issues_found"] == 2
            # Breakdowns should only count the 2 non-MINORITY issues
            assert data["issues_by_type"] == {"CONTRADICTION": 1, "ENTITY_INCONSISTENCY": 1}
            assert data["issues_by_severity"] == {"HIGH": 1, "MEDIUM": 1}

        finally:
            if review_id:
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()


class TestSummaryIncludeMinority:
    """GET /reviews/{id}?include_minority=true includes MINORITY in counts."""

    def test_summary_includes_minority_when_requested(self, client):
        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(agent_id=9999, status="COMPLETED", progress=100, issues_found=2)
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
                    review_id=review.id, issue_type="ENTITY_INCONSISTENCY", severity="LOW",
                    confidence=0.3, title="Minority", consensus="MINORITY",
                    judges_flagged=1, judges_total=3,
                ),
            ])
            db.commit()

            # Default: MINORITY excluded from counts
            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.json()["issues_found"] == 1
            assert resp.json()["issues_by_type"] == {"CONTRADICTION": 1}

            # include_minority=true: counts include MINORITY
            resp2 = client.get(f"/review-api/reviews/{review.id}?include_minority=true")
            assert resp2.status_code == 200
            data = resp2.json()
            assert data["issues_found"] == 2
            assert data["issues_by_type"] == {"CONTRADICTION": 1, "ENTITY_INCONSISTENCY": 1}
            assert data["issues_by_severity"] == {"HIGH": 1, "LOW": 1}

        finally:
            if review_id:
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()


class TestErrorMessageInResponse:
    """error_message is returned in API responses for FAILED/PARTIAL reviews."""

    def test_error_message_in_get_review(self, client):
        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(
                agent_id=9999, status="FAILED", progress=0, issues_found=0,
                error_message="Agent 9999 not found or has no tenant_id",
            )
            db.add(review)
            db.commit()
            db.refresh(review)
            review_id = review.id

            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["error_message"] == "Agent 9999 not found or has no tenant_id"
            assert data["status"] == "FAILED"

        finally:
            if review_id:
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()

    def test_error_message_null_for_completed(self, client):
        db = SessionLocal()
        review_id = None
        try:
            review = KbReview(agent_id=9999, status="COMPLETED", progress=100, issues_found=0)
            db.add(review)
            db.commit()
            db.refresh(review)
            review_id = review.id

            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.status_code == 200
            assert resp.json()["error_message"] is None

        finally:
            if review_id:
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()


class TestSanitizeJudgeConfig:
    """api_key must not be stored in config_json (security)."""

    def test_api_key_stripped_from_config_json(self, client):
        resp = client.post("/review-api/reviews", json={
            "agent_id": 9999,
            "judges": [{"provider": "litellm", "model": "test", "api_key": "sk-secret-key"}],
        })
        assert resp.status_code == 202
        review_id = resp.json()["id"]

        db = SessionLocal()
        try:
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            config = json.loads(review.config_json)
            for j in config["judges"]:
                assert "api_key" not in j, "api_key should be stripped from stored config"
                assert "sk-secret-key" not in json.dumps(j)
        finally:
            db.close()

    def test_auto_mode_config_has_no_api_key(self, client):
        resp = client.post("/review-api/reviews", json={"agent_id": 9999})
        assert resp.status_code == 202
        review_id = resp.json()["id"]

        db = SessionLocal()
        try:
            review = db.query(KbReview).filter(KbReview.id == review_id).first()
            config = json.loads(review.config_json)
            for j in config["judges"]:
                assert "api_key" not in j
            for j in config.get("fallback_judges", []):
                assert "api_key" not in j
        finally:
            db.close()


class TestReviewWithIssues:
    """Test that manually-inserted issues are correctly returned."""

    def test_issues_returned_with_judge_results(self, client):
        db = SessionLocal()
        review_id = None
        try:
            # Create review
            review = KbReview(agent_id=9999, status="COMPLETED", progress=100, issues_found=1)
            db.add(review)
            db.flush()
            review_id = review.id

            # Create issue
            issue = KbReviewIssue(
                review_id=review.id,
                issue_type="CONTRADICTION",
                severity="CRITICAL",
                confidence=0.95,
                title="Price conflict",
                description="Conflicting prices",
                doc_a_name="Pricing.pdf",
                doc_a_page=3,
                doc_a_excerpt="$99/month",
                doc_b_name="FAQ.docx",
                doc_b_page=12,
                doc_b_excerpt="$149/month",
                consensus="UNANIMOUS",
                judges_flagged=2,
                judges_total=2,
            )
            db.add(issue)
            db.flush()

            # Create judge result
            jr = KbReviewJudgeResult(
                issue_id=issue.id,
                judge_index=0,
                judge_provider="litellm",
                judge_model="test-model",
                detected=True,
                severity="CRITICAL",
                confidence=0.95,
                reasoning="Price differs between documents",
            )
            db.add(jr)
            db.commit()

            # Fetch via API
            resp = client.get(f"/review-api/reviews/{review.id}/issues")
            assert resp.status_code == 200
            issues = resp.json()
            assert len(issues) == 1

            api_issue = issues[0]
            assert api_issue["issue_type"] == "CONTRADICTION"
            assert api_issue["severity"] == "CRITICAL"
            assert api_issue["doc_a_name"] == "Pricing.pdf"
            assert api_issue["doc_b_name"] == "FAQ.docx"
            assert api_issue["consensus"] == "UNANIMOUS"
            assert len(api_issue["judge_results"]) == 1
            assert api_issue["judge_results"][0]["reasoning"] == "Price differs between documents"

            # Verify summary counts
            resp2 = client.get(f"/review-api/reviews/{review.id}")
            assert resp2.status_code == 200
            summary = resp2.json()
            assert summary["issues_by_type"]["CONTRADICTION"] == 1
            assert summary["issues_by_severity"]["CRITICAL"] == 1

        finally:
            if review_id:
                db.query(KbReviewJudgeResult).filter(
                    KbReviewJudgeResult.issue_id.in_(
                        db.query(KbReviewIssue.id).filter(KbReviewIssue.review_id == review_id)
                    )
                ).delete(synchronize_session=False)
                db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
                db.query(KbReview).filter(KbReview.id == review_id).delete()
                db.commit()
            db.close()
