import asyncio
import json
import logging
from collections import Counter
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database_layer.db_config import get_db, SessionLocal
from app.database_layer.db_models import KbReview, KbReviewIssue, KbReviewJudgeStat, Agents
from app.database_layer.db_schemas import (
    ReviewRequest, ReviewSummary, ReviewListItem, IssueOut, JudgeResultOut,
    JudgeStatOut, JudgeConfig, SyncReviewResponse, IssueStatusUpdate,
    VALID_ISSUE_STATUSES,
)
from app.analysis.review_runner import run_review
from app.api.auth import require_auth, CallerContext, authorize_agent_access

logger = logging.getLogger("kb_review")

router = APIRouter()

# ── Auto mode defaults ───────────────────────────────────────────────────
# Judges are selected at runtime based on which API keys are present in .env.
# Priority: Anthropic > Google Gemini > LiteLLM proxy (kept for backward compat).

def _get_auto_judges() -> tuple[list[JudgeConfig], list[JudgeConfig]]:
    """Return (primary_judges, fallback_judges) based on available API keys."""
    from app.core.config import settings

    anthropic_available = bool(settings.ANTHROPIC_API_KEY)
    google_available = bool(settings.GOOGLE_API_KEY)

    if anthropic_available:
        primary = [
            JudgeConfig(provider="anthropic", model="claude-haiku-4-5"),
            JudgeConfig(provider="anthropic", model="claude-haiku-4-5"),
            JudgeConfig(provider="anthropic", model="claude-haiku-4-5"),
        ]
    elif google_available:
        primary = [
            JudgeConfig(provider="google", model="gemini-2.0-flash"),
            JudgeConfig(provider="google", model="gemini-2.0-flash"),
            JudgeConfig(provider="google", model="gemini-2.0-flash"),
        ]
    else:
        # LiteLLM proxy fallback — code preserved for backward compatibility
        primary = [
            JudgeConfig(provider="litellm", model="gemini/gemini-2.0-flash"),
            JudgeConfig(provider="litellm", model="gemini/gemini-2.0-flash"),
            JudgeConfig(provider="litellm", model="gemini/gemini-2.0-flash"),
        ]

    if anthropic_available and google_available:
        # Both keys present — use Google as fallback for diversity
        fallback = [
            JudgeConfig(provider="google", model="gemini-2.0-flash"),
            JudgeConfig(provider="google", model="gemini-2.0-flash"),
            JudgeConfig(provider="google", model="gemini-2.0-flash"),
        ]
    elif google_available and not anthropic_available:
        # Only Google — reuse same for fallback
        fallback = primary
    else:
        # LiteLLM as fallback — code preserved
        fallback = [
            JudgeConfig(provider="litellm", model="anthropic/claude-haiku-4-5"),
            JudgeConfig(provider="litellm", model="anthropic/claude-haiku-4-5"),
            JudgeConfig(provider="litellm", model="anthropic/claude-haiku-4-5"),
        ]

    return primary, fallback

SYNC_TIMEOUT_SECONDS = 300  # 5 minutes


# ── Helpers ──────────────────────────────────────────────────────────────


def _lookup_agent(db: Session, agent_id: int):
    """Read-only lookup of an agent by ID."""
    return db.query(Agents).filter(Agents.id == agent_id).first()


def _sanitize_judge_config(j: dict) -> dict:
    """Remove secrets before persisting to config_json."""
    return {k: v for k, v in j.items() if k != "api_key"}


def _build_review_summary(
    review: KbReview, db: Session, include_minority: bool = False,
) -> ReviewSummary:
    """Build a ReviewSummary from a KbReview record.

    issues_by_type/issues_by_severity counts match the issue list the caller
    will see: MINORITY excluded by default, included when requested.
    """
    issues = db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review.id).all()
    visible = issues if include_minority else [i for i in issues if i.consensus != "MINORITY"]
    active = [i for i in visible if getattr(i, "status", "OPEN") in ("OPEN", "ACKNOWLEDGED")]
    resolved = [i for i in visible if getattr(i, "status", "OPEN") in ("RESOLVED", "DISMISSED")]
    by_type = dict(Counter(i.issue_type for i in active))
    by_severity = dict(Counter(i.severity for i in active))

    stats = (
        db.query(KbReviewJudgeStat)
        .filter(KbReviewJudgeStat.review_id == review.id)
        .order_by(KbReviewJudgeStat.judge_index)
        .all()
    )
    judge_stats = [JudgeStatOut.model_validate(s) for s in stats]

    return ReviewSummary(
        id=review.id,
        agent_id=review.agent_id,
        status=review.status,
        progress=review.progress or 0,
        total_documents=review.total_documents or 0,
        total_chunks=review.total_chunks or 0,
        issues_found=len(active),
        issues_resolved=len(resolved),
        url_duplicates_removed=review.url_duplicates_removed or 0,
        issues_by_type=by_type,
        issues_by_severity=by_severity,
        judge_stats=judge_stats,
        previous_review_id=review.previous_review_id,
        pairs_reused=review.pairs_reused or 0,
        pairs_analyzed=review.pairs_analyzed or 0,
        docs_changed=review.docs_changed or 0,
        docs_unchanged=review.docs_unchanged or 0,
        created_by_user_id=review.created_by_user_id,
        error_message=review.error_message,
        started_at=review.started_at,
        completed_at=review.completed_at,
        created_at=review.created_at,
    )


def _serialize_issues(issues: list, include_minority: bool = False) -> list[IssueOut]:
    """Convert KbReviewIssue list to IssueOut list, optionally filtering MINORITY."""
    result = []
    for issue in issues:
        if not include_minority and issue.consensus == "MINORITY":
            continue
        judge_results = [
            JudgeResultOut(
                judge_index=jr.judge_index,
                judge_provider=jr.judge_provider,
                judge_model=jr.judge_model,
                detected=jr.detected,
                severity=jr.severity,
                confidence=jr.confidence,
                reasoning=jr.reasoning,
                created_at=jr.created_at,
            )
            for jr in issue.judge_results
        ]
        result.append(IssueOut(
            id=issue.id,
            review_id=issue.review_id,
            issue_type=issue.issue_type,
            severity=issue.severity,
            confidence=issue.confidence,
            title=issue.title,
            description=issue.description,
            doc_a_name=issue.doc_a_name,
            doc_a_page=issue.doc_a_page,
            doc_a_excerpt=issue.doc_a_excerpt,
            doc_b_name=issue.doc_b_name,
            doc_b_page=issue.doc_b_page,
            doc_b_excerpt=issue.doc_b_excerpt,
            entities_involved=issue.entities_involved,
            consensus=issue.consensus,
            judges_flagged=issue.judges_flagged,
            judges_total=issue.judges_total,
            carried_forward=issue.carried_forward or False,
            original_review_id=issue.original_review_id,
            status=getattr(issue, "status", "OPEN") or "OPEN",
            status_updated_by=getattr(issue, "status_updated_by", None),
            status_updated_at=getattr(issue, "status_updated_at", None),
            status_note=getattr(issue, "status_note", None),
            judge_results=judge_results,
            created_at=issue.created_at,
        ))
    return result


def _read_completed_review(
    review_id: int, include_minority: bool = False,
) -> SyncReviewResponse:
    """Read the completed review from a fresh DB session.

    Opens a new SessionLocal() to avoid stale reads from MySQL's REPEATABLE
    READ isolation — the endpoint's session may have cached pre-run_review data.
    """
    fresh_db = SessionLocal()
    try:
        review = fresh_db.query(KbReview).filter(KbReview.id == review_id).first()
        if not review:
            raise HTTPException(status_code=500, detail="Review not found after completion")
        summary = _build_review_summary(review, fresh_db, include_minority=include_minority)

        issues = (
            fresh_db.query(KbReviewIssue)
            .filter(KbReviewIssue.review_id == review_id)
            .order_by(KbReviewIssue.id)
            .all()
        )
        issue_list = _serialize_issues(issues, include_minority=include_minority)

        return SyncReviewResponse(review=summary, issues=issue_list)
    finally:
        fresh_db.close()


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("/reviews")
async def start_review(
    request: ReviewRequest,
    background_tasks: BackgroundTasks,
    caller: CallerContext = Depends(require_auth),
    wait: bool = Query(False, description="Block until review completes (max 5 min)"),
    include_minority: bool = Query(
        False, description="Include MINORITY consensus findings in sync response",
    ),
    carryforward: bool = Query(
        True, description="Carry forward findings for unchanged document pairs (default true)",
    ),
    db: Session = Depends(get_db),
):
    """Start a KB review.

    - **Auto mode** (judges omitted): 3x Gemini no-reasoning + Haiku fallback.
    - **Custom mode** (judges provided): uses specified judges, no fallback.
    - **?wait=true**: blocks until completion and returns full results.
    - **?carryforward=false**: force full LLM analysis, ignore previous review.
    """
    # Validate agent exists
    agent = _lookup_agent(db, request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
    if not agent.tenant_id:
        raise HTTPException(status_code=400, detail=f"Agent {request.agent_id} has no tenant_id")

    authorize_agent_access(caller, agent)

    # Concurrency guard — reject if agent already has an active review.
    active = db.query(KbReview).filter(
        KbReview.agent_id == request.agent_id,
        KbReview.status.in_(["PENDING", "RUNNING"]),
    ).first()
    if active:
        raise HTTPException(
            status_code=409,
            detail=f"Agent {request.agent_id} already has an active review (id={active.id}, status={active.status})",
        )

    if request.judges is not None:
        # Custom mode — validate providers
        valid_providers = {"litellm", "fireworks", "openrouter", "anthropic", "google"}
        for j in request.judges:
            if j.provider not in valid_providers:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid provider: {j.provider}. Must be one of {valid_providers}",
                )
        if len(request.judges) == 0:
            raise HTTPException(status_code=400, detail="judges list cannot be empty")

        config = {
            "judges": [_sanitize_judge_config(j.model_dump()) for j in request.judges],
            "analysis_types": request.analysis_types,
            "similarity_threshold": request.similarity_threshold,
            "max_candidate_pairs": request.max_candidate_pairs,
            "carryforward": carryforward,
        }
    else:
        # Auto mode — judges selected dynamically from available API keys
        _primary, _fallback = _get_auto_judges()
        config = {
            "judges": [_sanitize_judge_config(j.model_dump()) for j in _primary],
            "fallback_judges": [_sanitize_judge_config(j.model_dump()) for j in _fallback],
            "analysis_types": request.analysis_types,
            "similarity_threshold": request.similarity_threshold,
            "max_candidate_pairs": request.max_candidate_pairs,
            "carryforward": carryforward,
        }

    # Create review record
    review = KbReview(
        agent_id=request.agent_id,
        status="PENDING",
        config_json=json.dumps(config),
        created_by_user_id=caller.user_id,
    )
    db.add(review)
    db.commit()
    db.refresh(review)

    pending_summary = ReviewSummary(
        id=review.id,
        agent_id=review.agent_id,
        status=review.status,
        progress=review.progress or 0,
        total_documents=review.total_documents or 0,
        total_chunks=review.total_chunks or 0,
        issues_found=review.issues_found or 0,
        url_duplicates_removed=0,
        created_by_user_id=review.created_by_user_id,
        created_at=review.created_at,
    )

    if not wait:
        # Async mode — fire and forget
        background_tasks.add_task(run_review, review.id)
        return JSONResponse(
            status_code=202,
            content=pending_summary.model_dump(mode="json"),
        )

    # Sync mode — run review in thread pool and wait for completion.
    # run_in_executor is needed because BackgroundTasks fires AFTER the
    # response is sent. run_review() is sync and creates its own event loop
    # via asyncio.run() in the executor thread — no conflict with ours.
    loop = asyncio.get_running_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, run_review, review.id),
            timeout=SYNC_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        # The executor thread keeps running — Python threads can't be
        # cancelled. The review will complete in the background. Return 202
        # so the client can poll GET /reviews/{id} for the final result.
        logger.info(
            f"Review {review.id}: sync timeout after {SYNC_TIMEOUT_SECONDS}s, "
            f"returning 202 — review continues in background"
        )
        # Read fresh state from DB — review may have progressed to RUNNING
        # since we created pending_summary.
        fresh_db = SessionLocal()
        try:
            fresh_review = fresh_db.query(KbReview).filter(KbReview.id == review.id).first()
            timeout_summary = _build_review_summary(fresh_review, fresh_db)
        finally:
            fresh_db.close()
        return JSONResponse(
            status_code=202,
            content=timeout_summary.model_dump(mode="json"),
        )

    # Completed within timeout — return full results
    result = _read_completed_review(review.id, include_minority=include_minority)
    return JSONResponse(
        status_code=200,
        content=result.model_dump(mode="json"),
    )


@router.get("/reviews/{review_id}", response_model=ReviewSummary)
async def get_review(
    review_id: int,
    caller: CallerContext = Depends(require_auth),
    include_minority: bool = Query(
        False, description="Include MINORITY consensus findings in counts",
    ),
    db: Session = Depends(get_db),
):
    """Get review status and summary."""
    review = db.query(KbReview).filter(KbReview.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")

    if not caller.is_service and not caller.is_super_admin:
        agent = _lookup_agent(db, review.agent_id)
        if not agent or not caller.can_access_agent(agent.company_id):
            raise HTTPException(
                status_code=404, detail=f"Review {review_id} not found",
            )

    return _build_review_summary(review, db, include_minority=include_minority)


@router.get("/reviews", response_model=list[ReviewListItem])
async def list_reviews(
    agent_id: Optional[int] = Query(None),
    caller: CallerContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """List reviews, optionally filtered by agent_id."""
    query = db.query(KbReview)

    if agent_id is not None:
        # If a specific agent is requested, verify access for non-service callers
        if not caller.is_service and not caller.is_super_admin:
            agent = _lookup_agent(db, agent_id)
            if not agent or not caller.can_access_agent(agent.company_id):
                raise HTTPException(
                    status_code=404, detail=f"Agent {agent_id} not found",
                )
        query = query.filter(KbReview.agent_id == agent_id)
    elif not caller.is_service and not caller.is_super_admin:
        # No agent_id specified — auto-filter to caller's company agents
        company_agent_ids = (
            db.query(Agents.id)
            .filter(Agents.company_id == caller.company_id)
            .scalar_subquery()
        )
        query = query.filter(KbReview.agent_id.in_(company_agent_ids))

    reviews = query.order_by(KbReview.created_at.desc()).all()

    return [
        ReviewListItem(
            id=r.id,
            agent_id=r.agent_id,
            status=r.status,
            progress=r.progress or 0,
            issues_found=r.issues_found or 0,
            issues_resolved=r.issues_resolved or 0,
            created_by_user_id=r.created_by_user_id,
            created_at=r.created_at,
        )
        for r in reviews
    ]


@router.get("/reviews/{review_id}/issues", response_model=list[IssueOut])
async def get_review_issues(
    review_id: int,
    caller: CallerContext = Depends(require_auth),
    issue_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    include_minority: bool = Query(
        False, description="Include MINORITY consensus findings (excluded by default)",
    ),
    carried_forward: Optional[bool] = Query(
        None, description="Filter by carried_forward: true=only carried forward, false=only fresh",
    ),
    status: Optional[str] = Query(
        None, description="Filter by status: OPEN, RESOLVED, DISMISSED, ACKNOWLEDGED",
    ),
    db: Session = Depends(get_db),
):
    """Get detailed issues for a review with optional filters."""
    review = db.query(KbReview).filter(KbReview.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")

    if not caller.is_service and not caller.is_super_admin:
        agent = _lookup_agent(db, review.agent_id)
        if not agent or not caller.can_access_agent(agent.company_id):
            raise HTTPException(
                status_code=404, detail=f"Review {review_id} not found",
            )

    query = db.query(KbReviewIssue).filter(KbReviewIssue.review_id == review_id)

    if not include_minority:
        query = query.filter(KbReviewIssue.consensus != "MINORITY")

    if carried_forward is not None:
        query = query.filter(KbReviewIssue.carried_forward == carried_forward)

    if status is not None:
        if status not in VALID_ISSUE_STATUSES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status filter: {status}. Must be one of {sorted(VALID_ISSUE_STATUSES)}",
            )
        query = query.filter(KbReviewIssue.status == status)

    if issue_type:
        query = query.filter(KbReviewIssue.issue_type == issue_type)
    if severity:
        query = query.filter(KbReviewIssue.severity == severity)
    if min_confidence is not None:
        query = query.filter(KbReviewIssue.confidence >= min_confidence)

    issues = query.order_by(KbReviewIssue.id).all()
    return _serialize_issues(issues, include_minority=True)  # already filtered by query


@router.patch("/reviews/{review_id}/issues/{issue_id}", response_model=IssueOut)
async def update_issue_status(
    review_id: int,
    issue_id: int,
    body: IssueStatusUpdate,
    caller: CallerContext = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Update the status of an issue (OPEN, RESOLVED, DISMISSED, ACKNOWLEDGED)."""
    review = db.query(KbReview).filter(KbReview.id == review_id).first()
    if not review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")

    # Company access check BEFORE active-status check to prevent info leak:
    # without this order, cross-company users could distinguish active vs
    # completed reviews by getting 409 vs 404.
    if not caller.is_service and not caller.is_super_admin:
        agent = _lookup_agent(db, review.agent_id)
        if not agent or not caller.can_access_agent(agent.company_id):
            raise HTTPException(
                status_code=404, detail=f"Review {review_id} not found",
            )

    # Block status changes while review is still running
    if review.status in ("PENDING", "RUNNING"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot update issue status while review is {review.status}. Wait for the review to complete.",
        )

    issue = (
        db.query(KbReviewIssue)
        .filter(KbReviewIssue.id == issue_id, KbReviewIssue.review_id == review_id)
        .first()
    )
    if not issue:
        raise HTTPException(status_code=404, detail=f"Issue {issue_id} not found in review {review_id}")

    issue.status = body.status
    issue.status_updated_by = caller.user_id
    issue.status_updated_at = datetime.now()
    issue.status_note = body.note
    db.flush()

    # Atomic count recalculation to avoid race conditions with concurrent PATCHes.
    # NOTE: if adding bulk status updates, move recalculation outside the loop
    # to avoid N redundant COUNT queries. Single atomic UPDATE at the end suffices.
    db.execute(text("""
        UPDATE kb_reviews SET
            issues_found = (SELECT COUNT(*) FROM kb_review_issues
                            WHERE review_id = :rid AND consensus != 'MINORITY'
                            AND status IN ('OPEN', 'ACKNOWLEDGED')),
            issues_resolved = (SELECT COUNT(*) FROM kb_review_issues
                               WHERE review_id = :rid AND consensus != 'MINORITY'
                               AND status IN ('RESOLVED', 'DISMISSED'))
        WHERE id = :rid
    """), {"rid": review_id})
    db.commit()
    db.refresh(issue)

    return _serialize_issues([issue], include_minority=True)[0]
