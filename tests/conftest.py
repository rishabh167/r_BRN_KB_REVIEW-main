"""Shared test fixtures.

Integration tests connect to the MySQL configured in .env (or env vars).
Unit tests use no DB at all — they test pure functions.
Neo4j and LLM calls are always mocked.
Agents table is READ-ONLY — tests must never write to it.
"""

import logging
import pytest
from unittest.mock import MagicMock

from sqlalchemy import text, inspect
from app.database_layer.db_config import Base, engine, SessionLocal
from app.database_layer.db_models import (
    KbReview, KbReviewIssue, KbReviewJudgeResult, KbReviewJudgeStat, KbReviewDocHash,
)

logger = logging.getLogger("kb_review.tests")

# Tables owned by this service — safe to create_all. Excludes read-only
# tables (agents, users, permissions, roles_permissions) that the test DB
# user has no access to.
_OUR_TABLES = [
    KbReview.__table__,
    KbReviewIssue.__table__,
    KbReviewJudgeResult.__table__,
    KbReviewJudgeStat.__table__,
    KbReviewDocHash.__table__,
]


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Ensure kb_review tables exist before the test session.

    We only call create_all (IF NOT EXISTS) for our owned tables.
    No teardown — individual test fixtures clean up their own rows.
    We never touch read-only tables (agents, users, permissions, etc.).
    """
    Base.metadata.create_all(bind=engine, tables=_OUR_TABLES)

    # Add new columns to existing tables (Hibernate-style auto-update).
    # Needed because create_all won't ALTER existing tables.
    with engine.connect() as conn:
        insp = inspect(engine)

        # kb_reviews columns
        review_cols = {c["name"] for c in insp.get_columns("kb_reviews")}
        for col_name, col_sql in [
            ("created_by_user_id", "ALTER TABLE kb_reviews ADD COLUMN created_by_user_id BIGINT"),
            ("issues_resolved", "ALTER TABLE kb_reviews ADD COLUMN issues_resolved INT DEFAULT 0"),
        ]:
            if col_name not in review_cols:
                try:
                    conn.execute(text(col_sql))
                    conn.commit()
                    logger.info(f"Added {col_name} column to kb_reviews")
                except Exception as e:
                    conn.rollback()
                    logger.warning(f"Could not add {col_name} column to kb_reviews: {e}")

        # kb_review_issues columns
        issue_cols = {c["name"] for c in insp.get_columns("kb_review_issues")}
        for col_name, col_sql in [
            ("status", "ALTER TABLE kb_review_issues ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'OPEN'"),
            ("status_updated_by", "ALTER TABLE kb_review_issues ADD COLUMN status_updated_by BIGINT"),
            ("status_updated_at", "ALTER TABLE kb_review_issues ADD COLUMN status_updated_at DATETIME"),
            ("status_note", "ALTER TABLE kb_review_issues ADD COLUMN status_note TEXT"),
        ]:
            if col_name not in issue_cols:
                try:
                    conn.execute(text(col_sql))
                    conn.commit()
                    logger.info(f"Added {col_name} column to kb_review_issues")
                except Exception as e:
                    conn.rollback()
                    logger.warning(f"Could not add {col_name} column to kb_review_issues: {e}")

        # Index on kb_review_issues.status (for UI filtering)
        existing_indexes = {idx["name"] for idx in insp.get_indexes("kb_review_issues")}
        if "ix_kb_review_issues_status" not in existing_indexes:
            try:
                conn.execute(text(
                    "CREATE INDEX ix_kb_review_issues_status ON kb_review_issues (status)"
                ))
                conn.commit()
                logger.info("Added status index to kb_review_issues")
            except Exception as e:
                conn.rollback()
                logger.warning(f"Could not add status index to kb_review_issues: {e}")

    yield


def make_fake_agent(agent_id, tenant_id, name="Test Agent", company_id=1):
    """Create a fake Agents-like object for mocking. No DB writes."""
    agent = MagicMock()
    agent.id = agent_id
    agent.tenant_id = tenant_id
    agent.name = name
    agent.company_id = company_id
    agent.is_active = True
    return agent


def test_caller_override():
    """Return a CallerContext that bypasses auth for existing tests."""
    from app.api.auth import CallerContext
    return CallerContext(auth_type="api_key")


@pytest.fixture
def sample_chunks():
    """Realistic chunk data as returned by Neo4j reader."""
    return [
        {
            "text": "The enterprise plan is priced at $99 per month and includes unlimited users.",
            "source": "Product_Pricing_2024.pdf",
            "page": 3,
            "split_id": 0,
            "document_id": "doc-001",
            "document_content_id": "dc-001",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
            "total_pages": 10,
        },
        {
            "text": "Our enterprise tier is available at $149/month with premium support included.",
            "source": "Company_FAQ.docx",
            "page": 12,
            "split_id": 0,
            "document_id": "doc-002",
            "document_content_id": "dc-002",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
            "total_pages": 20,
        },
        {
            "text": "Refund requests must be submitted within 30 days of purchase.",
            "source": "Return_Policy.pdf",
            "page": 1,
            "split_id": 0,
            "document_id": "doc-003",
            "document_content_id": "dc-003",
            "embedding": [0.9, 0.8, 0.7, 0.6, 0.5] * 20,
            "total_pages": 5,
        },
        {
            "text": "All refunds are processed within 14 business days of the return.",
            "source": "Return_Policy.pdf",
            "page": 2,
            "split_id": 0,
            "document_id": "doc-003",
            "document_content_id": "dc-004",
            "embedding": [0.85, 0.75, 0.65, 0.55, 0.45] * 20,
            "total_pages": 5,
        },
    ]


@pytest.fixture
def chunks_with_issues():
    """Chunks that have structural problems."""
    return [
        {
            "text": "Normal content here.",
            "source": "Good_Doc.pdf",
            "page": 1,
            "split_id": 0,
            "document_id": "doc-ok",
            "document_content_id": "dc-ok",
            "embedding": [0.1] * 100,
            "total_pages": 5,
        },
        {
            "text": "This section has no embedding.",
            "source": "Bad_Doc.pdf",
            "page": 1,
            "split_id": 0,
            "document_id": "doc-bad1",
            "document_content_id": "dc-bad1",
            "embedding": None,
            "total_pages": 3,
        },
        {
            "text": "",
            "source": "Empty_Doc.pdf",
            "page": 1,
            "split_id": 0,
            "document_id": "doc-bad2",
            "document_content_id": "dc-bad2",
            "embedding": [0.2] * 100,
            "total_pages": 2,
        },
        {
            "text": "   ",
            "source": "Empty_Doc.pdf",
            "page": 2,
            "split_id": 0,
            "document_id": "doc-bad2",
            "document_content_id": "dc-bad3",
            "embedding": [0.3] * 100,
            "total_pages": 2,
        },
        {
            "text": None,
            "source": "Null_Doc.pdf",
            "page": 1,
            "split_id": 0,
            "document_id": "doc-bad3",
            "document_content_id": "dc-bad4",
            "embedding": None,
            "total_pages": 1,
        },
    ]


@pytest.fixture
def sample_entity_mappings():
    """Entity-chunk mappings as returned by Neo4j reader."""
    return [
        {"entity_id": "enterprise plan", "entity_labels": ["__Entity__", "Product"],
         "document_id": "doc-001", "source": "Product_Pricing_2024.pdf", "page": 3, "split_id": 0},
        {"entity_id": "enterprise plan", "entity_labels": ["__Entity__", "Product"],
         "document_id": "doc-002", "source": "Company_FAQ.docx", "page": 12, "split_id": 0},
        {"entity_id": "pricing", "entity_labels": ["__Entity__", "Concept"],
         "document_id": "doc-001", "source": "Product_Pricing_2024.pdf", "page": 3, "split_id": 0},
        {"entity_id": "pricing", "entity_labels": ["__Entity__", "Concept"],
         "document_id": "doc-002", "source": "Company_FAQ.docx", "page": 12, "split_id": 0},
    ]


@pytest.fixture
def sample_entity_relationships():
    """Entity relationships as returned by Neo4j reader."""
    return [
        {"source_entity": "enterprise plan", "target_entity": "pricing", "relationship_type": "HAS_PRICE"},
    ]


@pytest.fixture
def mock_neo4j_reader(sample_chunks, sample_entity_mappings, sample_entity_relationships):
    """Mock the neo4j_reader module-level instance."""
    reader = MagicMock()
    reader.get_document_chunks.return_value = sample_chunks
    reader.get_entity_chunk_mappings.return_value = sample_entity_mappings
    reader.get_entity_relationships.return_value = sample_entity_relationships
    return reader
