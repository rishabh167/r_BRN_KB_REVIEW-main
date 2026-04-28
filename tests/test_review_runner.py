"""Integration tests for the review_runner pipeline.

Uses MySQL configured in .env, mocks Neo4j, LLM calls, and agent lookups.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.database_layer.db_config import SessionLocal
from app.database_layer.db_models import (
    KbReview, KbReviewIssue, KbReviewJudgeResult, KbReviewJudgeStat,
    KbReviewDocHash,
)
from tests.conftest import make_fake_agent

FAKE_AGENT = make_fake_agent(8888, "test-tenant-runner", "Runner Test Agent")
FAKE_AGENT_NO_TENANT = make_fake_agent(7777, None, "No Tenant")


def _mock_lookup(db, agent_id):
    return {8888: FAKE_AGENT, 7777: FAKE_AGENT_NO_TENANT}.get(agent_id)


def _fresh_review(review_id: int) -> KbReview:
    """Read a review from a fresh session to avoid REPEATABLE READ staleness."""
    db = SessionLocal()
    try:
        return db.query(KbReview).filter(KbReview.id == review_id).first()
    finally:
        db.close()


def _fresh_issues(review_id: int) -> list[KbReviewIssue]:
    """Read issues from a fresh session."""
    db = SessionLocal()
    try:
        return db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).all()
    finally:
        db.close()


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def review_record(db_session):
    """Create a PENDING review record."""
    config = {
        "judges": [{"provider": "litellm", "model": "test-model"}],
        "analysis_types": ["CONTRADICTION", "ENTITY_INCONSISTENCY"],
        "similarity_threshold": 0.85,
        "max_candidate_pairs": 50,
    }
    review = KbReview(
        agent_id=8888,
        status="PENDING",
        config_json=json.dumps(config),
    )
    db_session.add(review)
    db_session.commit()
    db_session.refresh(review)
    review_id = review.id
    yield review
    # Cleanup with fresh session
    cleanup_db = SessionLocal()
    try:
        cleanup_db.query(KbReviewJudgeResult).filter(
            KbReviewJudgeResult.issue_id.in_(
                cleanup_db.query(KbReviewIssue.id).filter(KbReviewIssue.review_id == review_id)
            )
        ).delete(synchronize_session=False)
        cleanup_db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
        cleanup_db.query(KbReviewDocHash).filter(
            KbReviewDocHash.agent_id == 8888
        ).delete()
        cleanup_db.query(KbReview).filter(KbReview.id == review_id).delete()
        cleanup_db.commit()
    finally:
        cleanup_db.close()


class TestRunReviewEmptyKB:
    """Test pipeline with an agent that has no KB data."""

    def test_completes_with_zero_issues(self, review_record):
        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = []
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
             patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup):
            from app.analysis.review_runner import run_review
            run_review(review_record.id)

        updated = _fresh_review(review_record.id)
        assert updated.status == "COMPLETED"
        assert updated.progress == 100
        assert updated.issues_found == 0
        assert updated.total_chunks == 0


class TestRunReviewStructuralIssues:
    """Test pipeline with structural issues only (no LLM needed)."""

    def test_detects_missing_embeddings(self, review_record):
        chunks = [
            {"text": "content", "source": "Doc.pdf", "page": 1, "split_id": 0,
             "document_id": "d1", "document_content_id": "dc1", "embedding": None, "total_pages": 1},
        ]
        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = chunks
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        mock_judge = MagicMock()
        mock_judge.ainvoke = AsyncMock(return_value=MagicMock(content="[]", usage_metadata=None))

        with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
             patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
             patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup):
            from app.analysis.review_runner import run_review
            run_review(review_record.id)

        updated = _fresh_review(review_record.id)
        assert updated.status == "COMPLETED"
        assert updated.issues_found >= 1

        issues = _fresh_issues(review_record.id)
        structural = [i for i in issues if i.issue_type == "MISSING_EMBEDDINGS"]
        assert len(structural) >= 1
        assert structural[0].consensus == "STRUCTURAL"
        assert structural[0].confidence == 1.0


class TestRunReviewWithLLMFindings:
    """Test pipeline with mocked LLM that returns findings."""

    def test_persists_llm_findings(self, review_record):
        chunks = [
            {"text": "Enterprise plan is $99/month", "source": "Pricing.pdf", "page": 3,
             "split_id": 0, "document_id": "d1", "document_content_id": "dc1",
             "embedding": [0.1] * 10, "total_pages": 5},
            {"text": "Enterprise costs $149/month", "source": "FAQ.docx", "page": 12,
             "split_id": 0, "document_id": "d2", "document_content_id": "dc2",
             "embedding": [0.1] * 10, "total_pages": 20},
        ]
        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = chunks
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        llm_response = json.dumps([{
            "detected": True,
            "pair_index": 0,
            "issue_type": "CONTRADICTION",
            "severity": "CRITICAL",
            "confidence": 0.95,
            "title": "Conflicting enterprise pricing",
            "description": "Price conflict between documents",
            "reasoning": "One says $99, other says $149",
            "claim_a": "$99/month",
            "claim_b": "$149/month",
        }])

        mock_ai_response = MagicMock()
        mock_ai_response.content = llm_response
        mock_ai_response.usage_metadata = {"input_tokens": 500, "output_tokens": 200}

        mock_judge = MagicMock()
        mock_judge.ainvoke = AsyncMock(return_value=mock_ai_response)

        with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
             patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
             patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup):
            from app.analysis.review_runner import run_review
            run_review(review_record.id)

        updated = _fresh_review(review_record.id)
        assert updated.status == "COMPLETED"

        issues = _fresh_issues(review_record.id)
        contradictions = [i for i in issues if i.issue_type == "CONTRADICTION"]
        for issue in contradictions:
            assert issue.severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
            assert issue.consensus in ("UNANIMOUS", "MAJORITY", "MINORITY", "SINGLE_JUDGE")


class TestRunReviewFailure:
    """Test pipeline error handling."""

    def test_nonexistent_review(self):
        """Running with a non-existent review_id should not crash."""
        mock_reader = MagicMock()
        with patch("app.analysis.review_runner.neo4j_reader", mock_reader):
            from app.analysis.review_runner import run_review
            run_review(999999)

    def test_agent_no_tenant_marks_failed(self):
        """Agent without tenant_id should mark review as FAILED."""
        db = SessionLocal()
        try:
            config = {"judges": [{"provider": "litellm", "model": "test"}],
                      "analysis_types": ["CONTRADICTION"], "similarity_threshold": 0.85,
                      "max_candidate_pairs": 50}
            review = KbReview(agent_id=7777, status="PENDING", config_json=json.dumps(config))
            db.add(review)
            db.commit()
            db.refresh(review)
            review_id = review.id
        finally:
            db.close()

        mock_reader = MagicMock()
        with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
             patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup):
            from app.analysis.review_runner import run_review
            run_review(review_id)

        updated = _fresh_review(review_id)
        assert updated.status == "FAILED"
        assert "tenant_id" in updated.error_message

        # Cleanup
        db = SessionLocal()
        try:
            db.query(KbReview).filter(KbReview.id == review_id).delete()
            db.commit()
        finally:
            db.close()


class TestAutoFailover:
    """Test that run_review retries with fallback judges when primaries fail."""

    @pytest.fixture
    def failover_review(self, db_session):
        """Create a review with primary + fallback judge config."""
        config = {
            "judges": [{"provider": "litellm", "model": "gemini/gemini-3-flash-preview"}],
            "fallback_judges": [{"provider": "litellm", "model": "anthropic/claude-haiku-4-5"}],
            "analysis_types": ["CONTRADICTION"],
            "similarity_threshold": 0.85,
            "max_candidate_pairs": 50,
        }
        review = KbReview(
            agent_id=8888,
            status="PENDING",
            config_json=json.dumps(config),
        )
        db_session.add(review)
        db_session.commit()
        db_session.refresh(review)
        review_id = review.id
        yield review
        # Cleanup
        cleanup_db = SessionLocal()
        try:
            cleanup_db.query(KbReviewJudgeResult).filter(
                KbReviewJudgeResult.issue_id.in_(
                    cleanup_db.query(KbReviewIssue.id).filter(KbReviewIssue.review_id == review_id)
                )
            ).delete(synchronize_session=False)
            cleanup_db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id).delete()
            cleanup_db.query(KbReviewJudgeStat).filter(KbReviewJudgeStat.review_id == review_id).delete()
            cleanup_db.query(KbReviewDocHash).filter(
                KbReviewDocHash.agent_id == 8888
            ).delete()
            cleanup_db.query(KbReview).filter(KbReview.id == review_id).delete()
            cleanup_db.commit()
        finally:
            cleanup_db.close()

    def test_failover_to_fallback_judges(self, failover_review):
        """When all primary judges fail on probe, fallback judges are used."""
        chunks = [
            {"text": "Enterprise plan is $99/month", "source": "Pricing.pdf", "page": 3,
             "split_id": 0, "document_id": "d1", "document_content_id": "dc1",
             "embedding": [0.1] * 10, "total_pages": 5},
            {"text": "Enterprise costs $149/month", "source": "FAQ.docx", "page": 12,
             "split_id": 0, "document_id": "d2", "document_content_id": "dc2",
             "embedding": [0.1] * 10, "total_pages": 20},
        ]
        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = chunks
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        call_count = 0

        def mock_create_judge(config, rate_limiter=None):
            nonlocal call_count
            call_count += 1
            judge = MagicMock()
            if "gemini" in config.model:
                # Primary judge — fails on probe batch
                judge.ainvoke = AsyncMock(side_effect=Exception("Gemini unavailable"))
            else:
                # Fallback judge — succeeds with empty findings
                judge.ainvoke = AsyncMock(return_value=MagicMock(
                    content="[]", usage_metadata=None,
                ))
            return judge

        with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
             patch("app.analysis.review_runner.create_judge", side_effect=mock_create_judge), \
             patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup):
            from app.analysis.review_runner import run_review
            run_review(failover_review.id)

        updated = _fresh_review(failover_review.id)
        assert updated.status == "COMPLETED"

        # Verify judge stats are from the fallback model, not primary
        stats_db = SessionLocal()
        try:
            stats = stats_db.query(KbReviewJudgeStat).filter(
                KbReviewJudgeStat.review_id == failover_review.id
            ).all()
            for stat in stats:
                assert "haiku" in stat.judge_model, (
                    f"Expected fallback model (haiku) but got {stat.judge_model}"
                )
        finally:
            stats_db.close()

        # create_judge was called twice: once for primary, once for fallback
        assert call_count == 2

    def test_no_fallback_marks_failed(self, db_session):
        """When all judges fail and no fallback_judges exist, review is FAILED."""
        config = {
            "judges": [{"provider": "litellm", "model": "gemini/gemini-3-flash-preview"}],
            # No fallback_judges
            "analysis_types": ["CONTRADICTION"],
            "similarity_threshold": 0.85,
            "max_candidate_pairs": 50,
        }
        review = KbReview(
            agent_id=8888,
            status="PENDING",
            config_json=json.dumps(config),
        )
        db_session.add(review)
        db_session.commit()
        db_session.refresh(review)
        review_id = review.id

        chunks = [
            {"text": "Content A", "source": "A.pdf", "page": 1, "split_id": 0,
             "document_id": "d1", "document_content_id": "dc1",
             "embedding": [0.1] * 10, "total_pages": 1},
            {"text": "Content B", "source": "B.pdf", "page": 1, "split_id": 0,
             "document_id": "d2", "document_content_id": "dc2",
             "embedding": [0.1] * 10, "total_pages": 1},
        ]
        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = chunks
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        mock_judge = MagicMock()
        mock_judge.ainvoke = AsyncMock(side_effect=Exception("Model unavailable"))

        with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
             patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
             patch("app.analysis.review_runner._lookup_agent", side_effect=_mock_lookup):
            from app.analysis.review_runner import run_review
            run_review(review_id)

        updated = _fresh_review(review_id)
        assert updated.status == "FAILED"

        # Cleanup
        cleanup_db = SessionLocal()
        try:
            cleanup_db.query(KbReviewJudgeStat).filter(
                KbReviewJudgeStat.review_id == review_id).delete()
            cleanup_db.query(KbReview).filter(KbReview.id == review_id).delete()
            cleanup_db.commit()
        finally:
            cleanup_db.close()


# ── Change detection / carry-forward integration tests ────────────────────


CARRY_FORWARD_AGENT_ID = 6666
CARRY_FORWARD_AGENT = make_fake_agent(CARRY_FORWARD_AGENT_ID, "test-tenant-cf", "CF Agent")


def _cf_mock_lookup(db, agent_id):
    if agent_id == CARRY_FORWARD_AGENT_ID:
        return CARRY_FORWARD_AGENT
    return _mock_lookup(db, agent_id)


def _cf_cleanup(review_ids: list[int]):
    """Clean up reviews, issues, judge results, stats, and doc hashes."""
    cleanup_db = SessionLocal()
    try:
        for rid in review_ids:
            cleanup_db.query(KbReviewJudgeResult).filter(
                KbReviewJudgeResult.issue_id.in_(
                    cleanup_db.query(KbReviewIssue.id).filter(KbReviewIssue.review_id == rid)
                )
            ).delete(synchronize_session=False)
            cleanup_db.query(KbReviewIssue).filter(KbReviewIssue.review_id == rid).delete()
            cleanup_db.query(KbReviewJudgeStat).filter(KbReviewJudgeStat.review_id == rid).delete()
            cleanup_db.query(KbReview).filter(KbReview.id == rid).delete()
        # Clean doc hashes for this agent
        cleanup_db.query(KbReviewDocHash).filter(
            KbReviewDocHash.agent_id == CARRY_FORWARD_AGENT_ID
        ).delete()
        cleanup_db.commit()
    finally:
        cleanup_db.close()


def _make_chunks(sources_content: dict[str, str]) -> list[dict]:
    """Build chunk list from {source: text} dict. Each source gets 1 chunk."""
    chunks = []
    for i, (source, text) in enumerate(sources_content.items()):
        chunks.append({
            "text": text,
            "source": source,
            "page": 1,
            "split_id": 0,
            "document_id": f"d{i}",
            "document_content_id": f"dc{i}",
            "embedding": [0.1] * 10,
            "total_pages": 1,
        })
    return chunks


def _create_review(db, config_overrides=None):
    """Create a PENDING review for the carry-forward agent."""
    config = {
        "judges": [{"provider": "litellm", "model": "test-model"}],
        "analysis_types": ["CONTRADICTION", "ENTITY_INCONSISTENCY"],
        "similarity_threshold": 0.85,
        "max_candidate_pairs": 50,
        "carryforward": True,
    }
    if config_overrides:
        config.update(config_overrides)
    review = KbReview(
        agent_id=CARRY_FORWARD_AGENT_ID,
        status="PENDING",
        config_json=json.dumps(config),
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


def _make_llm_finding_response(pair_index=0, title="Test contradiction"):
    """Build a mock LLM response with a single finding."""
    return json.dumps([{
        "detected": True,
        "pair_index": pair_index,
        "issue_type": "CONTRADICTION",
        "severity": "HIGH",
        "confidence": 0.9,
        "title": title,
        "description": "Test description",
        "reasoning": "Test reasoning",
        "claim_a": "Claim A",
        "claim_b": "Claim B",
    }])


class TestCarryForward:
    """Integration tests for carry-forward / change detection."""

    def test_second_review_carries_forward_all(self):
        """When all docs are unchanged, second review carries forward all findings."""
        chunks = _make_chunks({"Pricing.pdf": "Enterprise $99/mo", "FAQ.docx": "Enterprise $149/mo"})

        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = chunks
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        mock_ai_response = MagicMock()
        mock_ai_response.content = _make_llm_finding_response()
        mock_ai_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        mock_judge = MagicMock()
        mock_judge.ainvoke = AsyncMock(return_value=mock_ai_response)

        review_ids = []
        db = SessionLocal()
        try:
            # Run first review
            review1 = _create_review(db)
            review_ids.append(review1.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                from app.analysis.review_runner import run_review
                run_review(review1.id)

            r1 = _fresh_review(review1.id)
            assert r1.status == "COMPLETED"
            r1_issues = _fresh_issues(review1.id)
            r1_llm_issues = [i for i in r1_issues if i.judges_total > 0]
            assert len(r1_llm_issues) > 0

            # Reset mock call count for second review
            mock_judge.ainvoke.reset_mock()

            # Run second review with same data
            review2 = _create_review(db)
            review_ids.append(review2.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                run_review(review2.id)

            r2 = _fresh_review(review2.id)
            assert r2.status == "COMPLETED"
            assert r2.previous_review_id == review1.id
            assert r2.docs_unchanged > 0
            assert r2.pairs_reused > 0

            # LLM should NOT have been called for unchanged pairs
            # (ainvoke may be called 0 times if all pairs reused)
            r2_issues = _fresh_issues(review2.id)
            carried = [i for i in r2_issues if i.carried_forward]
            assert len(carried) > 0
            for cf_issue in carried:
                assert cf_issue.original_review_id == review1.id

        finally:
            db.close()
            _cf_cleanup(review_ids)

    def test_carry_forward_preserves_status_fields(self):
        """Carry-forward copies status, status_updated_by, status_updated_at, status_note."""
        from datetime import datetime

        chunks = _make_chunks({"Pricing.pdf": "Enterprise $99/mo", "FAQ.docx": "Enterprise $149/mo"})

        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = chunks
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        mock_ai_response = MagicMock()
        mock_ai_response.content = _make_llm_finding_response()
        mock_ai_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        mock_judge = MagicMock()
        mock_judge.ainvoke = AsyncMock(return_value=mock_ai_response)

        review_ids = []
        db = SessionLocal()
        try:
            # Run first review
            review1 = _create_review(db)
            review_ids.append(review1.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                from app.analysis.review_runner import run_review
                run_review(review1.id)

            r1 = _fresh_review(review1.id)
            assert r1.status == "COMPLETED"

            # Manually update an issue's status to simulate user resolving it
            r1_issues = _fresh_issues(review1.id)
            llm_issues = [i for i in r1_issues if i.judges_total > 0]
            assert len(llm_issues) > 0

            update_db = SessionLocal()
            try:
                issue = update_db.query(KbReviewIssue).filter(
                    KbReviewIssue.id == llm_issues[0].id
                ).first()
                issue.status = "RESOLVED"
                issue.status_updated_by = 42
                issue.status_updated_at = datetime.now()
                issue.status_note = "Fixed in latest training"
                update_db.commit()
            finally:
                update_db.close()

            # Run second review with same data — should carry forward with status
            mock_judge.ainvoke.reset_mock()
            review2 = _create_review(db)
            review_ids.append(review2.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                run_review(review2.id)

            r2 = _fresh_review(review2.id)
            assert r2.status == "COMPLETED"

            r2_issues = _fresh_issues(review2.id)
            carried = [i for i in r2_issues if i.carried_forward]
            assert len(carried) > 0

            # Find the carried issue that corresponds to our resolved one
            resolved_carried = [i for i in carried if i.status == "RESOLVED"]
            assert len(resolved_carried) > 0, "Carried-forward issue should preserve RESOLVED status"
            cf = resolved_carried[0]
            assert cf.status_updated_by == 42
            assert cf.status_updated_at is not None
            assert cf.status_note == "Fixed in latest training"

        finally:
            db.close()
            _cf_cleanup(review_ids)

    def test_changed_doc_causes_reanalysis(self):
        """When a doc changes, pairs involving it are re-analyzed."""
        chunks_v1 = _make_chunks({
            "Pricing.pdf": "Enterprise $99/mo",
            "FAQ.docx": "Enterprise $149/mo",
        })
        chunks_v2 = _make_chunks({
            "Pricing.pdf": "Enterprise $199/mo",  # changed
            "FAQ.docx": "Enterprise $149/mo",      # unchanged
        })

        mock_reader = MagicMock()
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        mock_ai_response = MagicMock()
        mock_ai_response.content = _make_llm_finding_response()
        mock_ai_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}
        mock_judge = MagicMock()
        mock_judge.ainvoke = AsyncMock(return_value=mock_ai_response)

        review_ids = []
        db = SessionLocal()
        try:
            # First review
            mock_reader.get_document_chunks.return_value = chunks_v1
            review1 = _create_review(db)
            review_ids.append(review1.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                from app.analysis.review_runner import run_review
                run_review(review1.id)

            assert _fresh_review(review1.id).status == "COMPLETED"

            # Second review with changed Pricing.pdf
            mock_reader.get_document_chunks.return_value = chunks_v2
            mock_judge.ainvoke.reset_mock()

            review2 = _create_review(db)
            review_ids.append(review2.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                run_review(review2.id)

            r2 = _fresh_review(review2.id)
            assert r2.status == "COMPLETED"
            assert r2.docs_changed >= 1
            assert r2.pairs_analyzed >= 1  # changed doc pairs re-analyzed

            # LLM was called because doc changed
            assert mock_judge.ainvoke.call_count > 0

        finally:
            db.close()
            _cf_cleanup(review_ids)

    def test_config_change_forces_full_analysis(self):
        """When config changes, no carry-forward — full analysis."""
        chunks = _make_chunks({"A.pdf": "Content A", "B.pdf": "Content B"})

        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = chunks
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        mock_ai_response = MagicMock()
        mock_ai_response.content = "[]"
        mock_ai_response.usage_metadata = None
        mock_judge = MagicMock()
        mock_judge.ainvoke = AsyncMock(return_value=mock_ai_response)

        review_ids = []
        db = SessionLocal()
        try:
            # First review with threshold 0.85
            review1 = _create_review(db)
            review_ids.append(review1.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                from app.analysis.review_runner import run_review
                run_review(review1.id)

            # Second review with different threshold
            review2 = _create_review(db, config_overrides={"similarity_threshold": 0.90})
            review_ids.append(review2.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                run_review(review2.id)

            r2 = _fresh_review(review2.id)
            assert r2.status == "COMPLETED"
            assert r2.previous_review_id is None  # no compatible previous review
            assert r2.pairs_reused == 0

        finally:
            db.close()
            _cf_cleanup(review_ids)

    def test_carryforward_false_forces_full(self):
        """?carryforward=false skips change detection entirely."""
        chunks = _make_chunks({"A.pdf": "Content A", "B.pdf": "Content B"})

        mock_reader = MagicMock()
        mock_reader.get_document_chunks.return_value = chunks
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        mock_ai_response = MagicMock()
        mock_ai_response.content = "[]"
        mock_ai_response.usage_metadata = None
        mock_judge = MagicMock()
        mock_judge.ainvoke = AsyncMock(return_value=mock_ai_response)

        review_ids = []
        db = SessionLocal()
        try:
            # First review
            review1 = _create_review(db)
            review_ids.append(review1.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                from app.analysis.review_runner import run_review
                run_review(review1.id)

            # Second review with carryforward=false
            review2 = _create_review(db, config_overrides={"carryforward": False})
            review_ids.append(review2.id)

            mock_judge.ainvoke.reset_mock()

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                run_review(review2.id)

            r2 = _fresh_review(review2.id)
            assert r2.status == "COMPLETED"
            assert r2.previous_review_id is None
            assert r2.pairs_reused == 0
            # LLM was called because carry-forward was disabled
            assert mock_judge.ainvoke.call_count > 0

        finally:
            db.close()
            _cf_cleanup(review_ids)

    def test_removed_doc_findings_not_carried(self):
        """Findings from removed documents should not be carried forward."""
        chunks_v1 = _make_chunks({
            "A.pdf": "Content A",
            "B.pdf": "Content B",
            "C.pdf": "Content C",
        })
        # C.pdf removed in v2
        chunks_v2 = _make_chunks({
            "A.pdf": "Content A",
            "B.pdf": "Content B",
        })

        mock_reader = MagicMock()
        mock_reader.get_entity_chunk_mappings.return_value = []
        mock_reader.get_entity_relationships.return_value = []

        mock_ai_response = MagicMock()
        mock_ai_response.content = "[]"
        mock_ai_response.usage_metadata = None
        mock_judge = MagicMock()
        mock_judge.ainvoke = AsyncMock(return_value=mock_ai_response)

        review_ids = []
        db = SessionLocal()
        try:
            mock_reader.get_document_chunks.return_value = chunks_v1
            review1 = _create_review(db)
            review_ids.append(review1.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                from app.analysis.review_runner import run_review
                run_review(review1.id)

            assert _fresh_review(review1.id).status == "COMPLETED"

            # Verify C.pdf hash exists (use fresh session to avoid REPEATABLE READ staleness)
            check_db = SessionLocal()
            try:
                c_hash = check_db.query(KbReviewDocHash).filter(
                    KbReviewDocHash.agent_id == CARRY_FORWARD_AGENT_ID,
                    KbReviewDocHash.source_canonical == "C.pdf",
                ).first()
                assert c_hash is not None
            finally:
                check_db.close()

            # Second review without C.pdf
            mock_reader.get_document_chunks.return_value = chunks_v2
            review2 = _create_review(db)
            review_ids.append(review2.id)

            with patch("app.analysis.review_runner.neo4j_reader", mock_reader), \
                 patch("app.analysis.review_runner.create_judge", return_value=mock_judge), \
                 patch("app.analysis.review_runner._lookup_agent", side_effect=_cf_mock_lookup):
                run_review(review2.id)

            assert _fresh_review(review2.id).status == "COMPLETED"

            # C.pdf hash should be deleted
            db2 = SessionLocal()
            try:
                c_hash_after = db2.query(KbReviewDocHash).filter(
                    KbReviewDocHash.agent_id == CARRY_FORWARD_AGENT_ID,
                    KbReviewDocHash.source_canonical == "C.pdf",
                ).first()
                assert c_hash_after is None
            finally:
                db2.close()

        finally:
            db.close()
            _cf_cleanup(review_ids)
