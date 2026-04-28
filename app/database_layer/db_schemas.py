from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional
from datetime import datetime

VALID_ANALYSIS_TYPES = {"CONTRADICTION", "ENTITY_INCONSISTENCY", "SEMANTIC_DUPLICATION", "AMBIGUITY"}
VALID_ISSUE_STATUSES = {"OPEN", "RESOLVED", "DISMISSED", "ACKNOWLEDGED"}

# ---------- Request models ----------

class JudgeConfig(BaseModel):
    provider: str = Field(..., description="litellm | fireworks | openrouter")
    model: str = Field(..., description="Model name for the provider")
    api_base: Optional[str] = Field(None, description="Override provider base URL")
    api_key: Optional[str] = Field(None, description="Override provider API key")
    temperature: Optional[float] = Field(None, description="LLM temperature")
    max_tokens: Optional[int] = Field(None, description="Max output tokens")
    reasoning_effort: Optional[str] = Field(None, description="Reasoning effort: low | medium | high (for thinking models like Gemini 3)")
    rate_limit_rpm: Optional[int] = Field(None, description="Rate limit in requests/min for this model. Judges sharing the same (provider, model) share one limiter.")


class ReviewRequest(BaseModel):
    agent_id: int
    judges: Optional[list[JudgeConfig]] = Field(
        None, description="Judge configs. Omit for auto mode (3x Gemini + Haiku fallback).",
    )
    analysis_types: list[str] = Field(
        default=["CONTRADICTION", "ENTITY_INCONSISTENCY"],
        description="Which LLM-based checks to run (optional: SEMANTIC_DUPLICATION, AMBIGUITY)",
    )
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_candidate_pairs: int = Field(default=50, ge=1, le=500)

    @field_validator("analysis_types")
    @classmethod
    def validate_analysis_types(cls, v):
        if not v:
            raise ValueError("analysis_types cannot be empty")
        invalid = set(v) - VALID_ANALYSIS_TYPES
        if invalid:
            raise ValueError(f"Invalid analysis types: {invalid}. Must be one of {VALID_ANALYSIS_TYPES}")
        return sorted(set(v))


class IssueStatusUpdate(BaseModel):
    status: str = Field(..., description="OPEN | RESOLVED | DISMISSED | ACKNOWLEDGED")
    note: Optional[str] = Field(None, max_length=1000, description="Optional note explaining the status change")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        if v not in VALID_ISSUE_STATUSES:
            raise ValueError(f"Invalid status: {v}. Must be one of {VALID_ISSUE_STATUSES}")
        return v


# ---------- Response models ----------

class JudgeStatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    judge_index: int
    judge_provider: Optional[str] = None
    judge_model: Optional[str] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_calls: int = 0
    total_findings: int = 0
    duration_ms: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ReviewSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_id: int
    status: str
    progress: int
    total_documents: int
    total_chunks: int
    issues_found: int
    issues_resolved: int = 0
    url_duplicates_removed: int = 0
    issues_by_type: dict[str, int] = {}
    issues_by_severity: dict[str, int] = {}
    judge_stats: list[JudgeStatOut] = []
    previous_review_id: Optional[int] = None
    pairs_reused: int = 0
    pairs_analyzed: int = 0
    docs_changed: int = 0
    docs_unchanged: int = 0
    created_by_user_id: Optional[int] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime


class ReviewListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    agent_id: int
    status: str
    progress: int
    issues_found: int
    issues_resolved: int = 0
    created_by_user_id: Optional[int] = None
    created_at: datetime


class JudgeResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    judge_index: int
    judge_provider: Optional[str] = None
    judge_model: Optional[str] = None
    detected: bool
    severity: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    created_at: Optional[datetime] = None


class IssueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    review_id: int
    issue_type: str
    severity: str
    confidence: float
    title: str
    description: Optional[str] = None
    doc_a_name: Optional[str] = None
    doc_a_page: Optional[int] = None
    doc_a_excerpt: Optional[str] = None
    doc_b_name: Optional[str] = None
    doc_b_page: Optional[int] = None
    doc_b_excerpt: Optional[str] = None
    entities_involved: Optional[str] = None
    consensus: Optional[str] = None
    judges_flagged: Optional[int] = None
    judges_total: Optional[int] = None
    carried_forward: bool = False
    original_review_id: Optional[int] = None
    status: str = "OPEN"
    status_updated_by: Optional[int] = None
    status_updated_at: Optional[datetime] = None
    status_note: Optional[str] = None
    judge_results: list[JudgeResultOut] = []
    created_at: datetime


class SyncReviewResponse(BaseModel):
    """Full review result returned by sync mode (?wait=true)."""
    model_config = ConfigDict(from_attributes=True)

    review: ReviewSummary
    issues: list[IssueOut] = []
