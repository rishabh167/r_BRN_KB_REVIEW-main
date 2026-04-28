from sqlalchemy import (
    Column, String, Text, BigInteger, Integer, DateTime, Boolean, Float, ForeignKey,
    UniqueConstraint, Index,
)
from sqlalchemy.orm import relationship
from datetime import datetime
from .db_config import Base


# ---------- Read-only: existing tables ----------

class Agents(Base):
    __tablename__ = "agents"
    __table_args__ = {"extend_existing": True}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id = Column(String(255), unique=True, nullable=True)
    name = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    company_id = Column(BigInteger, nullable=True)
    is_active = Column(Boolean, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=True)


class Users(Base):
    """Read-only mapping for auth: company lookup + role check."""
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}

    id = Column(BigInteger, primary_key=True)
    company_id = Column(BigInteger, nullable=False)
    role_id = Column(BigInteger, nullable=False)
    status = Column(String(20), nullable=False)


class Permissions(Base):
    """Read-only mapping for auth: super_admin_access check."""
    __tablename__ = "permissions"
    __table_args__ = {"extend_existing": True}

    id = Column(BigInteger, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)


class RolesPermissions(Base):
    """Read-only mapping for auth: role → permission join table."""
    __tablename__ = "roles_permissions"
    __table_args__ = {"extend_existing": True}

    id = Column(BigInteger, primary_key=True)
    roles_id = Column(BigInteger, nullable=False)
    permissions_id = Column(BigInteger, nullable=False)


# ---------- KB Review tables ----------

class KbReview(Base):
    __tablename__ = "kb_reviews"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    agent_id = Column(BigInteger, nullable=False, index=True)
    status = Column(String(20), nullable=False, default="PENDING")
    config_json = Column(Text, nullable=True)
    total_documents = Column(Integer, nullable=True, default=0)
    total_chunks = Column(Integer, nullable=True, default=0)
    chunks_with_issues = Column(Integer, nullable=True, default=0)
    candidate_pairs = Column(Integer, nullable=True, default=0)
    url_duplicates_removed = Column(Integer, nullable=True, default=0)
    issues_found = Column(Integer, nullable=True, default=0)
    issues_resolved = Column(Integer, nullable=True, default=0)
    progress = Column(Integer, nullable=True, default=0)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    previous_review_id = Column(BigInteger, nullable=True)
    pairs_reused = Column(Integer, nullable=True, default=0)
    pairs_analyzed = Column(Integer, nullable=True, default=0)
    docs_changed = Column(Integer, nullable=True, default=0)
    docs_unchanged = Column(Integer, nullable=True, default=0)
    created_by_user_id = Column(BigInteger, nullable=True)

    issues = relationship("KbReviewIssue", back_populates="review", cascade="all, delete-orphan")
    judge_stats = relationship("KbReviewJudgeStat", back_populates="review", cascade="all, delete-orphan")


class KbReviewIssue(Base):
    __tablename__ = "kb_review_issues"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    review_id = Column(BigInteger, ForeignKey("kb_reviews.id"), nullable=False, index=True)
    issue_type = Column(String(30), nullable=False)
    severity = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    doc_a_name = Column(String(500), nullable=True)
    doc_a_page = Column(Integer, nullable=True)
    doc_a_excerpt = Column(Text, nullable=True)
    doc_b_name = Column(String(500), nullable=True)
    doc_b_page = Column(Integer, nullable=True)
    doc_b_excerpt = Column(Text, nullable=True)
    entities_involved = Column(Text, nullable=True)
    consensus = Column(String(20), nullable=True)
    judges_flagged = Column(Integer, nullable=True, default=0)
    judges_total = Column(Integer, nullable=True, default=0)
    carried_forward = Column(Boolean, nullable=False, default=False)
    original_review_id = Column(BigInteger, nullable=True)
    status = Column(String(20), nullable=False, default="OPEN", index=True)
    status_updated_by = Column(BigInteger, nullable=True)
    status_updated_at = Column(DateTime, nullable=True)
    status_note = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    review = relationship("KbReview", back_populates="issues")
    judge_results = relationship("KbReviewJudgeResult", back_populates="issue", cascade="all, delete-orphan")


class KbReviewJudgeResult(Base):
    __tablename__ = "kb_review_judge_results"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    issue_id = Column(BigInteger, ForeignKey("kb_review_issues.id"), nullable=False, index=True)
    judge_index = Column(Integer, nullable=False)
    judge_provider = Column(String(30), nullable=True)
    judge_model = Column(String(100), nullable=True)
    detected = Column(Boolean, nullable=False, default=False)
    severity = Column(String(10), nullable=True)
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)
    input_tokens = Column(Integer, nullable=True, default=0)
    output_tokens = Column(Integer, nullable=True, default=0)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    issue = relationship("KbReviewIssue", back_populates="judge_results")


class KbReviewJudgeStat(Base):
    __tablename__ = "kb_review_judge_stats"
    __table_args__ = (
        UniqueConstraint("review_id", "judge_index", name="idx_judge_stats_review_judge"),
        Index("idx_judge_stats_review_id", "review_id"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    review_id = Column(BigInteger, ForeignKey("kb_reviews.id", ondelete="CASCADE"), nullable=False)
    judge_index = Column(Integer, nullable=False)
    judge_provider = Column(String(30), nullable=True)
    judge_model = Column(String(100), nullable=True)
    total_input_tokens = Column(Integer, nullable=True, default=0)
    total_output_tokens = Column(Integer, nullable=True, default=0)
    total_llm_calls = Column(Integer, nullable=True, default=0)
    total_findings = Column(Integer, nullable=True, default=0)
    duration_ms = Column(Integer, nullable=True, default=0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)

    review = relationship("KbReview", back_populates="judge_stats")


class KbReviewDocHash(Base):
    __tablename__ = "kb_review_doc_hashes"
    __table_args__ = (
        UniqueConstraint("agent_id", "source_canonical", name="uq_agent_source"),
        Index("idx_doc_hashes_agent_id", "agent_id"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    agent_id = Column(BigInteger, nullable=False)
    source_canonical = Column(String(500), nullable=False)
    content_hash = Column(String(64), nullable=False)
    review_id = Column(BigInteger, ForeignKey("kb_reviews.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=True)
