"""Unit tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError
from app.database_layer.db_schemas import (
    JudgeConfig,
    ReviewRequest,
    ReviewSummary,
    IssueOut,
)
from datetime import datetime


class TestJudgeConfig:
    def test_valid_config(self):
        jc = JudgeConfig(provider="litellm", model="test-model")
        assert jc.provider == "litellm"
        assert jc.model == "test-model"
        assert jc.api_base is None
        assert jc.api_key is None
        assert jc.temperature is None
        assert jc.max_tokens is None

    def test_full_config(self):
        jc = JudgeConfig(
            provider="fireworks",
            model="llama-70b",
            api_base="http://custom:4000/v1",
            api_key="sk-test",
            temperature=0.2,
            max_tokens=8000,
        )
        assert jc.api_base == "http://custom:4000/v1"
        assert jc.temperature == 0.2

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            JudgeConfig(provider="litellm")  # missing model

        with pytest.raises(ValidationError):
            JudgeConfig(model="test")  # missing provider


class TestReviewRequest:
    def test_valid_request(self):
        req = ReviewRequest(
            agent_id=42,
            judges=[JudgeConfig(provider="litellm", model="test")],
        )
        assert req.agent_id == 42
        assert len(req.judges) == 1
        assert req.similarity_threshold == 0.85
        assert req.max_candidate_pairs == 50
        assert len(req.analysis_types) == 2

    def test_multiple_judges(self):
        req = ReviewRequest(
            agent_id=1,
            judges=[
                JudgeConfig(provider="litellm", model="a"),
                JudgeConfig(provider="fireworks", model="b"),
                JudgeConfig(provider="openrouter", model="c"),
            ],
        )
        assert len(req.judges) == 3

    def test_empty_judges_accepted_at_schema_level(self):
        """Empty list is valid at schema level; API endpoint rejects it."""
        req = ReviewRequest(agent_id=1, judges=[])
        assert req.judges == []

    def test_omitted_judges_is_auto_mode(self):
        """Omitting judges sets None (auto mode)."""
        req = ReviewRequest(agent_id=1)
        assert req.judges is None

    def test_custom_thresholds(self):
        req = ReviewRequest(
            agent_id=1,
            judges=[JudgeConfig(provider="litellm", model="t")],
            similarity_threshold=0.7,
            max_candidate_pairs=50,
            analysis_types=["CONTRADICTION"],
        )
        assert req.similarity_threshold == 0.7
        assert req.max_candidate_pairs == 50
        assert req.analysis_types == ["CONTRADICTION"]

    def test_invalid_threshold(self):
        with pytest.raises(ValidationError):
            ReviewRequest(
                agent_id=1,
                judges=[JudgeConfig(provider="litellm", model="t")],
                similarity_threshold=1.5,  # > 1.0
            )

    def test_invalid_max_pairs(self):
        with pytest.raises(ValidationError):
            ReviewRequest(
                agent_id=1,
                judges=[JudgeConfig(provider="litellm", model="t")],
                max_candidate_pairs=0,  # < 1
            )

    def test_max_candidate_pairs_too_high(self):
        with pytest.raises(ValidationError):
            ReviewRequest(
                agent_id=1,
                judges=[JudgeConfig(provider="litellm", model="t")],
                max_candidate_pairs=501,  # > 500
            )

    def test_empty_analysis_types_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(agent_id=1, analysis_types=[])

    def test_invalid_analysis_types_rejected(self):
        with pytest.raises(ValidationError):
            ReviewRequest(agent_id=1, analysis_types=["CONTRADICTION", "FAKE_TYPE"])


class TestReviewSummary:
    def test_from_dict(self):
        summary = ReviewSummary(
            id=1, agent_id=42, status="COMPLETED", progress=100,
            total_documents=8, total_chunks=142, issues_found=7,
            issues_by_type={"CONTRADICTION": 2, "AMBIGUITY": 5},
            issues_by_severity={"CRITICAL": 1, "MEDIUM": 6},
            created_at=datetime.now(),
        )
        assert summary.issues_found == 7
        assert summary.issues_by_type["CONTRADICTION"] == 2

    def test_optional_fields_default(self):
        summary = ReviewSummary(
            id=1, agent_id=1, status="PENDING", progress=0,
            total_documents=0, total_chunks=0, issues_found=0,
            created_at=datetime.now(),
        )
        assert summary.started_at is None
        assert summary.completed_at is None
        assert summary.issues_by_type == {}
        assert summary.previous_review_id is None
        assert summary.pairs_reused == 0
        assert summary.pairs_analyzed == 0
        assert summary.docs_changed == 0
        assert summary.docs_unchanged == 0

    def test_carry_forward_fields(self):
        summary = ReviewSummary(
            id=2, agent_id=1, status="COMPLETED", progress=100,
            total_documents=10, total_chunks=50, issues_found=5,
            previous_review_id=1,
            pairs_reused=8, pairs_analyzed=2,
            docs_changed=1, docs_unchanged=9,
            created_at=datetime.now(),
        )
        assert summary.previous_review_id == 1
        assert summary.pairs_reused == 8
        assert summary.pairs_analyzed == 2
        assert summary.docs_changed == 1
        assert summary.docs_unchanged == 9


class TestIssueOut:
    def test_complete_issue(self):
        issue = IssueOut(
            id=1, review_id=1, issue_type="CONTRADICTION", severity="CRITICAL",
            confidence=0.95, title="Price conflict",
            doc_a_name="A.pdf", doc_a_page=3, doc_a_excerpt="$99/month",
            doc_b_name="B.pdf", doc_b_page=12, doc_b_excerpt="$149/month",
            consensus="UNANIMOUS", judges_flagged=3, judges_total=3,
            created_at=datetime.now(),
        )
        assert issue.issue_type == "CONTRADICTION"
        assert issue.judge_results == []
        assert issue.carried_forward is False
        assert issue.original_review_id is None

    def test_structural_issue_no_doc_b(self):
        issue = IssueOut(
            id=2, review_id=1, issue_type="MISSING_EMBEDDINGS", severity="HIGH",
            confidence=1.0, title="15 sections missing embeddings",
            doc_a_name="FAQ.pdf", doc_a_page=1,
            consensus="STRUCTURAL",
            created_at=datetime.now(),
        )
        assert issue.doc_b_name is None
        assert issue.doc_b_page is None

    def test_carried_forward_issue(self):
        issue = IssueOut(
            id=3, review_id=2, issue_type="CONTRADICTION", severity="HIGH",
            confidence=0.9, title="Carried forward finding",
            doc_a_name="A.pdf", doc_a_page=1,
            doc_b_name="B.pdf", doc_b_page=2,
            consensus="UNANIMOUS", judges_flagged=3, judges_total=3,
            carried_forward=True, original_review_id=1,
            created_at=datetime.now(),
        )
        assert issue.carried_forward is True
        assert issue.original_review_id == 1
