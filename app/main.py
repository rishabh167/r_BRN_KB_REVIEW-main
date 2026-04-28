import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import review_api
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kb_review")


_PROCESS_START = None  # set in lifespan before calling recovery


def _recover_stale_reviews():
    """Mark any RUNNING or orphaned PENDING reviews as PARTIAL/FAILED on startup.

    Any RUNNING/PENDING review from before this process started is definitely
    orphaned (the previous process is dead). Uses _PROCESS_START as the cutoff
    so even reviews that ran for <30 min are recovered after a crash/restart.
    Falls back to a 5-minute age cutoff if _PROCESS_START is not set.
    """
    from app.database_layer.db_config import SessionLocal
    from app.database_layer.db_models import KbReview, KbReviewIssue

    db = SessionLocal()
    try:
        cutoff = _PROCESS_START or datetime.now() - timedelta(minutes=5)
        stale = db.query(KbReview).filter(
            (
                (KbReview.status == "RUNNING") &
                ((KbReview.started_at == None) | (KbReview.started_at < cutoff))
            ) |
            ((KbReview.status == "PENDING") & (KbReview.created_at < cutoff))
        ).all()
        for review in stale:
            # Exclude MINORITY so the count matches the default API response.
            # Split active vs resolved to match the issues_found/issues_resolved semantic.
            active_count = db.query(KbReviewIssue).filter(
                KbReviewIssue.review_id == review.id,
                KbReviewIssue.consensus != "MINORITY",
                KbReviewIssue.status.in_(["OPEN", "ACKNOWLEDGED"]),
            ).count()
            resolved_count = db.query(KbReviewIssue).filter(
                KbReviewIssue.review_id == review.id,
                KbReviewIssue.consensus != "MINORITY",
                KbReviewIssue.status.in_(["RESOLVED", "DISMISSED"]),
            ).count()
            total_count = active_count + resolved_count
            review.status = "PARTIAL" if total_count > 0 else "FAILED"
            review.issues_found = active_count
            review.issues_resolved = resolved_count
            review.progress = 100
            review.completed_at = datetime.now()
            review.error_message = "Service was interrupted during review"
            logger.warning(f"Recovered stale review {review.id}: {review.status} ({total_count} issues)")
        if stale:
            db.commit()
    except Exception:
        logger.exception("Failed to recover stale reviews")
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _PROCESS_START
    _PROCESS_START = datetime.now()
    logger.info("BRN_KB_REVIEW service starting")
    from app.cache_db.redis_client import is_redis_configured
    if not is_redis_configured():
        if settings.APP_ENV.lower() == "production":
            logger.error("REDIS_HOST not set in production — direct JWT auth will be rejected (503)")
        else:
            logger.warning("REDIS_HOST not set — direct JWT blacklist checking disabled")
    _recover_stale_reviews()
    yield
    from app.graph_db.neo4j_reader import neo4j_reader
    neo4j_reader.close()
    logger.info("BRN_KB_REVIEW service shutting down")


_is_production = settings.APP_ENV.lower() == "production"

app = FastAPI(
    title="Broadnet KB Review Service",
    description="Automated knowledge base quality analysis — contradictions, inconsistencies, and training issues",
    version="0.1.0",
    docs_url=None if _is_production else "/review-api/docs",
    openapi_url=None if _is_production else "/review-api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # not needed — auth is via headers, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/review-api/health", tags=["Health Check"])
async def health_check():
    from app.cache_db.redis_client import check_redis_health
    return {
        "status": "healthy",
        "service": "Broadnet KB Review Service",
        "redis": check_redis_health(),
    }


app.include_router(review_api.router, prefix="/review-api", tags=["KB Reviews"])
