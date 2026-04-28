"""Tests for issue status management (PATCH endpoint + filtering).

Covers:
- PATCH status transitions (RESOLVED, DISMISSED, ACKNOWLEDGED, re-open)
- Validation (invalid status, wrong review, non-existent issue)
- Blocking during PENDING/RUNNING reviews (409)
- Atomic count recalculation (issues_found / issues_resolved)
- GET ?status= filter with 422 on invalid values
- Filter composition (status + include_minority + carried_forward)
- API key caller (status_updated_by is None)
- PATCH on carried-forward issues
"""

import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.database_layer.db_config import SessionLocal
from app.database_layer.db_models import KbReview, KbReviewIssue, KbReviewJudgeResult
from tests.conftest import make_fake_agent

AGENT = make_fake_agent(9999, "test-tenant-status", "Status Test Agent")

AGENT_MAP = {9999: AGENT}


def _mock_lookup(db, agent_id):
    return AGENT_MAP.get(agent_id)


@pytest.fixture(scope="module")
def client():
    """FastAPI test client with auth bypassed (API key caller)."""
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


def _create_review_with_issues(db, status="COMPLETED", issues=None):
    """Helper: create a review with issues. Returns (review, issue_list)."""
    review = KbReview(agent_id=9999, status=status, progress=100, issues_found=0)
    db.add(review)
    db.flush()

    issue_list = []
    if issues is None:
        issues = [
            {"issue_type": "CONTRADICTION", "severity": "HIGH", "confidence": 0.9,
             "title": "Price conflict", "consensus": "UNANIMOUS",
             "judges_flagged": 3, "judges_total": 3},
            {"issue_type": "ENTITY_INCONSISTENCY", "severity": "MEDIUM", "confidence": 0.7,
             "title": "Entity mismatch", "consensus": "MAJORITY",
             "judges_flagged": 2, "judges_total": 3},
        ]
    for issue_data in issues:
        issue = KbReviewIssue(review_id=review.id, **issue_data)
        db.add(issue)
        issue_list.append(issue)
    db.flush()

    # Update issues_found to match non-MINORITY active count
    active = sum(1 for i in issue_list
                 if getattr(i, "consensus", "") != "MINORITY"
                 and getattr(i, "status", "OPEN") in ("OPEN", "ACKNOWLEDGED"))
    review.issues_found = active
    db.commit()

    return review, issue_list


def _cleanup(db, review_id):
    """Clean up test data."""
    db.query(KbReviewJudgeResult).filter(
        KbReviewJudgeResult.issue_id.in_(
            db.query(KbReviewIssue.id).filter(KbReviewIssue.review_id == review_id)
        )
    ).delete(synchronize_session=False)
    db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
    db.query(KbReview).filter(KbReview.id == review_id).delete()
    db.commit()


class TestPatchIssueStatus:
    """PATCH /reviews/{review_id}/issues/{issue_id}"""

    def test_resolve_issue(self, client):
        """PATCH to RESOLVED updates status fields and returns updated issue."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            resp = client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "RESOLVED", "note": "Fixed in latest training"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "RESOLVED"
            assert data["status_note"] == "Fixed in latest training"
            assert data["status_updated_at"] is not None
            # API key caller -> status_updated_by is None
            assert data["status_updated_by"] is None
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_dismiss_issue_with_note(self, client):
        """PATCH to DISMISSED with note."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            resp = client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "DISMISSED", "note": "False positive"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "DISMISSED"
            assert resp.json()["status_note"] == "False positive"
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_acknowledge_issue(self, client):
        """PATCH to ACKNOWLEDGED."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            resp = client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "ACKNOWLEDGED"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "ACKNOWLEDGED"
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_reopen_issue(self, client):
        """PATCH back to OPEN (re-open a resolved issue)."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            # First resolve
            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "RESOLVED"},
            )
            # Then re-open
            resp = client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "OPEN", "note": "Not actually fixed"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "OPEN"
            assert resp.json()["status_note"] == "Not actually fixed"
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_invalid_status_returns_422(self, client):
        """PATCH with invalid status returns 422."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            resp = client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "INVALID"},
            )
            assert resp.status_code == 422
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_nonexistent_issue_returns_404(self, client):
        """PATCH non-existent issue returns 404."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            resp = client.patch(
                f"/review-api/reviews/{review.id}/issues/999999",
                json={"status": "RESOLVED"},
            )
            assert resp.status_code == 404
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_issue_from_wrong_review_returns_404(self, client):
        """PATCH issue that exists but belongs to a different review returns 404."""
        db = SessionLocal()
        review_id_1 = None
        review_id_2 = None
        try:
            review1, issues1 = _create_review_with_issues(db)
            review_id_1 = review1.id
            review2, issues2 = _create_review_with_issues(db)
            review_id_2 = review2.id

            # Try to PATCH issue from review1 using review2's URL
            resp = client.patch(
                f"/review-api/reviews/{review2.id}/issues/{issues1[0].id}",
                json={"status": "RESOLVED"},
            )
            assert resp.status_code == 404
        finally:
            if review_id_1:
                _cleanup(db, review_id_1)
            if review_id_2:
                _cleanup(db, review_id_2)
            db.close()

    def test_nonexistent_review_returns_404(self, client):
        """PATCH on non-existent review returns 404."""
        resp = client.patch(
            "/review-api/reviews/999999/issues/1",
            json={"status": "RESOLVED"},
        )
        assert resp.status_code == 404


class TestPatchBlocksDuringActiveReview:
    """PATCH is blocked while review is PENDING or RUNNING."""

    def test_patch_blocked_during_running(self, client):
        """PATCH returns 409 when review is RUNNING."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db, status="RUNNING")
            review_id = review.id

            resp = client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "RESOLVED"},
            )
            assert resp.status_code == 409
            assert "RUNNING" in resp.json()["detail"]
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_patch_blocked_during_pending(self, client):
        """PATCH returns 409 when review is PENDING."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db, status="PENDING")
            review_id = review.id

            resp = client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "RESOLVED"},
            )
            assert resp.status_code == 409
            assert "PENDING" in resp.json()["detail"]
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_patch_allowed_on_failed_review(self, client):
        """PATCH is allowed on FAILED reviews (issues may still exist for PARTIAL)."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db, status="PARTIAL")
            review_id = review.id

            resp = client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "DISMISSED", "note": "False positive from partial run"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "DISMISSED"
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()


class TestCountRecalculation:
    """Verify issues_found and issues_resolved are recalculated atomically."""

    def test_resolve_decrements_issues_found(self, client):
        """Resolving an issue decrements issues_found and increments issues_resolved."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            # Before: 2 active, 0 resolved
            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.json()["issues_found"] == 2
            assert resp.json()["issues_resolved"] == 0

            # Resolve one
            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "RESOLVED"},
            )

            # After: 1 active, 1 resolved
            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.json()["issues_found"] == 1
            assert resp.json()["issues_resolved"] == 1
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_dismiss_decrements_issues_found(self, client):
        """Dismissing an issue also decrements issues_found."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "DISMISSED"},
            )

            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.json()["issues_found"] == 1
            assert resp.json()["issues_resolved"] == 1
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_acknowledge_keeps_active(self, client):
        """Acknowledging an issue keeps it in issues_found (still active)."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "ACKNOWLEDGED"},
            )

            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.json()["issues_found"] == 2  # still active
            assert resp.json()["issues_resolved"] == 0
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_reopen_increments_issues_found(self, client):
        """Re-opening a resolved issue increments issues_found back."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            # Resolve both
            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "RESOLVED"},
            )
            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[1].id}",
                json={"status": "RESOLVED"},
            )

            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.json()["issues_found"] == 0
            assert resp.json()["issues_resolved"] == 2

            # Re-open one
            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "OPEN"},
            )

            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.json()["issues_found"] == 1
            assert resp.json()["issues_resolved"] == 1
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_minority_excluded_from_counts(self, client):
        """MINORITY issues don't affect issues_found/issues_resolved counts."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db, issues=[
                {"issue_type": "CONTRADICTION", "severity": "HIGH", "confidence": 0.9,
                 "title": "Unanimous", "consensus": "UNANIMOUS",
                 "judges_flagged": 3, "judges_total": 3},
                {"issue_type": "CONTRADICTION", "severity": "LOW", "confidence": 0.3,
                 "title": "Minority", "consensus": "MINORITY",
                 "judges_flagged": 1, "judges_total": 3},
            ])
            review_id = review.id

            # Resolve the MINORITY issue
            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[1].id}",
                json={"status": "RESOLVED"},
            )

            # MINORITY doesn't count — still 1 active, 0 resolved
            resp = client.get(f"/review-api/reviews/{review.id}")
            assert resp.json()["issues_found"] == 1
            assert resp.json()["issues_resolved"] == 0
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_list_reviews_includes_issues_resolved(self, client):
        """GET /reviews includes issues_resolved in list items."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "RESOLVED"},
            )

            resp = client.get("/review-api/reviews?agent_id=9999")
            assert resp.status_code == 200
            matching = [r for r in resp.json() if r["id"] == review.id]
            assert len(matching) == 1
            assert matching[0]["issues_resolved"] == 1
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()


class TestStatusFilter:
    """GET /reviews/{id}/issues?status= filtering."""

    def test_filter_by_open(self, client):
        """?status=OPEN returns only open issues."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            # Resolve first issue
            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "RESOLVED"},
            )

            resp = client.get(f"/review-api/reviews/{review.id}/issues?status=OPEN")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["status"] == "OPEN"
            assert data[0]["title"] == "Entity mismatch"
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_filter_by_resolved(self, client):
        """?status=RESOLVED returns only resolved issues."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "RESOLVED"},
            )

            resp = client.get(f"/review-api/reviews/{review.id}/issues?status=RESOLVED")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["status"] == "RESOLVED"
            assert data[0]["title"] == "Price conflict"
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_invalid_status_filter_returns_422(self, client):
        """?status=INVALID returns 422, not empty list."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db)
            review_id = review.id

            resp = client.get(f"/review-api/reviews/{review.id}/issues?status=INVALID")
            assert resp.status_code == 422
            assert "Invalid status filter" in resp.json()["detail"]
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_status_with_include_minority(self, client):
        """?status=OPEN&include_minority=true includes MINORITY open issues."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db, issues=[
                {"issue_type": "CONTRADICTION", "severity": "HIGH", "confidence": 0.9,
                 "title": "Unanimous open", "consensus": "UNANIMOUS",
                 "judges_flagged": 3, "judges_total": 3},
                {"issue_type": "CONTRADICTION", "severity": "LOW", "confidence": 0.3,
                 "title": "Minority open", "consensus": "MINORITY",
                 "judges_flagged": 1, "judges_total": 3},
            ])
            review_id = review.id

            # Default: MINORITY excluded
            resp = client.get(f"/review-api/reviews/{review.id}/issues?status=OPEN")
            assert len(resp.json()) == 1

            # With include_minority: both returned
            resp2 = client.get(
                f"/review-api/reviews/{review.id}/issues?status=OPEN&include_minority=true"
            )
            assert len(resp2.json()) == 2
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()

    def test_status_with_carried_forward(self, client):
        """?status=OPEN&carried_forward=false returns only fresh open issues."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db, issues=[
                {"issue_type": "CONTRADICTION", "severity": "HIGH", "confidence": 0.9,
                 "title": "Fresh open", "consensus": "UNANIMOUS",
                 "judges_flagged": 3, "judges_total": 3, "carried_forward": False},
                {"issue_type": "ENTITY_INCONSISTENCY", "severity": "MEDIUM", "confidence": 0.8,
                 "title": "Carried open", "consensus": "MAJORITY",
                 "judges_flagged": 2, "judges_total": 3,
                 "carried_forward": True, "original_review_id": 1},
            ])
            review_id = review.id

            # Only fresh open issues (most common real-world query)
            resp = client.get(
                f"/review-api/reviews/{review.id}/issues?status=OPEN&carried_forward=false"
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["title"] == "Fresh open"
            assert data[0]["carried_forward"] is False
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()


class TestPatchCarriedForwardIssue:
    """PATCH on carried-forward issues should work normally."""

    def test_patch_carried_forward_issue(self, client):
        """Carried-forward issues are fully mutable."""
        db = SessionLocal()
        review_id = None
        try:
            review, issues = _create_review_with_issues(db, issues=[
                {"issue_type": "CONTRADICTION", "severity": "HIGH", "confidence": 0.9,
                 "title": "Carried issue", "consensus": "UNANIMOUS",
                 "judges_flagged": 3, "judges_total": 3,
                 "carried_forward": True, "original_review_id": 1},
            ])
            review_id = review.id

            resp = client.patch(
                f"/review-api/reviews/{review.id}/issues/{issues[0].id}",
                json={"status": "RESOLVED", "note": "Finally fixed"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "RESOLVED"
            assert resp.json()["carried_forward"] is True
        finally:
            if review_id:
                _cleanup(db, review_id)
            db.close()
