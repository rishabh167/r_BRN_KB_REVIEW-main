"""Orchestrates the full review pipeline (Phases 1-5).

Designed to run as a background task. To upgrade to Celery:
swap background_tasks.add_task(run_review, ...) with run_review.delay(...)
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session
from langchain_core.rate_limiters import InMemoryRateLimiter
from app.database_layer.db_config import SessionLocal
from app.database_layer.db_models import (
    KbReview, KbReviewIssue, KbReviewJudgeResult, KbReviewJudgeStat,
    KbReviewDocHash, Agents,
)
from app.database_layer.db_schemas import JudgeConfig
from app.graph_db.neo4j_reader import neo4j_reader
from app.llm.judge_factory import create_judge
from .training_quality import run_training_quality_checks
from .pre_filter import build_candidate_pairs, deduplicate_url_chunks, _canonicalize_source
from .analyzers import analyze_pair_batch, analyze_ambiguity_batch
from .judge_aggregator import finding_key, update_consensus_for_review

logger = logging.getLogger("kb_review")

BATCH_SIZE = 5  # pairs per LLM call


class AllJudgesFailedError(Exception):
    """Raised when all judges fail (quorum not met). Contains first error for context."""
    pass


# ── Shared helpers ───────────────────────────────────────────────────────


def _lookup_agent(db: Session, agent_id: int):
    """Read-only lookup of an agent by ID."""
    return db.query(Agents).filter(Agents.id == agent_id).first()


def _update_progress(db: Session, review_id: int, progress: int, **kwargs):
    """Update review progress and optional fields."""
    review = db.query(KbReview).filter(KbReview.id == review_id).first()
    if review:
        review.progress = progress
        for k, v in kwargs.items():
            setattr(review, k, v)
        try:
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(f"Review {review_id}: failed to persist progress update to {progress}%")
            raise


def _persist_batch_findings(
    db: Session,
    review_id: int,
    findings: list[dict],
    num_judges: int,
    key_to_issue: dict[str, dict],
) -> int:
    """Persist findings from a single batch. Returns count of new issues created.

    key_to_issue maps finding_key → {"issue_id": int, "judges": set[int]}.
    judges_flagged is only bumped when a *new* judge encounters an existing key.

    Deduplicates within the batch (same judge can't flag the same thing twice).
    For each finding:
      - Key already seen by this judge → just add a KbReviewJudgeResult row.
      - Key already seen by a different judge → add result + bump judges_flagged.
      - Key is new → create KbReviewIssue + KbReviewJudgeResult, store in dict.
    """
    # Deduplicate within batch: same key → keep highest confidence
    deduped: dict[str, dict] = {}
    for f in findings:
        key = finding_key(f)
        if key not in deduped or f.get("confidence", 0) > deduped[key].get("confidence", 0):
            deduped[key] = f

    new_issues = 0
    for key, f in deduped.items():
        chunk_a = f.get("chunk_a", {})
        chunk_b = f.get("chunk_b")
        j_idx = f.get("judge_index", 0)

        if key in key_to_issue:
            issue_id = key_to_issue[key]["issue_id"]
            # Only bump judges_flagged if this is a new judge for this key
            if j_idx not in key_to_issue[key]["judges"]:
                issue = db.query(KbReviewIssue).filter(KbReviewIssue.id == issue_id).first()
                if issue:
                    issue.judges_flagged = (issue.judges_flagged or 0) + 1
                key_to_issue[key]["judges"].add(j_idx)
        else:
            # New issue
            entities_json = json.dumps(f.get("entities", []))
            issue = KbReviewIssue(
                review_id=review_id,
                issue_type=f["issue_type"],
                severity=f.get("severity", "MEDIUM"),
                confidence=f.get("confidence", 0.5),
                title=f["title"],
                description=f.get("description"),
                doc_a_name=chunk_a.get("source"),
                doc_a_page=chunk_a.get("page"),
                doc_a_excerpt=f.get("claim_a") or (chunk_a.get("text") or "")[:500],
                doc_b_name=chunk_b.get("source") if chunk_b else None,
                doc_b_page=chunk_b.get("page") if chunk_b else None,
                doc_b_excerpt=f.get("claim_b") or ((chunk_b.get("text") or "")[:500] if chunk_b else None),
                entities_involved=entities_json,
                consensus="PENDING",  # updated in Phase 4
                judges_flagged=1,
                judges_total=num_judges,
            )
            db.add(issue)
            db.flush()  # get issue.id
            key_to_issue[key] = {"issue_id": issue.id, "judges": {j_idx}}
            new_issues += 1
            issue_id = issue.id

        # Always add a judge result row
        jr = KbReviewJudgeResult(
            issue_id=issue_id,
            judge_index=j_idx,
            judge_provider=f.get("judge_provider"),
            judge_model=f.get("judge_model"),
            detected=True,
            severity=f.get("severity"),
            confidence=f.get("confidence"),
            reasoning=f.get("reasoning"),
            input_tokens=f.get("input_tokens", 0),
            output_tokens=f.get("output_tokens", 0),
        )
        db.add(jr)

    db.commit()
    return new_issues


# ── Change detection helpers ─────────────────────────────────────────────


def _compute_document_hashes(chunks: list[dict]) -> dict[str, str]:
    """Compute SHA-256 content hash per canonical source document.

    Groups chunks by canonical source, sorts by (page, split_id), and
    concatenates text with position markers for a deterministic hash.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        canon = _canonicalize_source(c.get("source", ""))
        by_source[canon].append(c)

    hashes = {}
    for source, source_chunks in by_source.items():
        source_chunks.sort(key=lambda c: (c.get("page", 0), c.get("split_id", 0)))
        content = "".join(
            f"p{c.get('page', 0)}s{c.get('split_id', 0)}:{c.get('text', '')}"
            for c in source_chunks
        )
        hashes[source] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return hashes


def _classify_documents(
    current_hashes: dict[str, str],
    previous_hashes: dict[str, str],
) -> tuple[set[str], set[str], set[str]]:
    """Classify documents as unchanged, changed, or new.

    Returns (unchanged, changed, new) sets of canonical source strings.
    """
    unchanged = set()
    changed = set()
    new = set()

    for source, hash_val in current_hashes.items():
        if source not in previous_hashes:
            new.add(source)
        elif previous_hashes[source] == hash_val:
            unchanged.add(source)
        else:
            changed.add(source)

    return unchanged, changed, new


def _configs_compatible_for_carryforward(
    current_config: dict, previous_config: dict,
) -> bool:
    """Check if two review configs are compatible for carry-forward.

    analysis_types, similarity_threshold, and max_candidate_pairs must
    match. Judge selection does NOT matter — findings are about content.
    """
    if sorted(current_config.get("analysis_types", [])) != sorted(previous_config.get("analysis_types", [])):
        return False
    if current_config.get("similarity_threshold") != previous_config.get("similarity_threshold"):
        return False
    if current_config.get("max_candidate_pairs") != previous_config.get("max_candidate_pairs"):
        return False
    return True


def _find_previous_review(
    db: Session, agent_id: int, current_config: dict,
) -> "KbReview | None":
    """Find the most recent COMPLETED review with compatible config."""
    recent = (
        db.query(KbReview)
        .filter(KbReview.agent_id == agent_id, KbReview.status == "COMPLETED")
        .order_by(KbReview.completed_at.desc())
        .limit(5)
        .all()
    )
    for r in recent:
        try:
            prev_config = json.loads(r.config_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if _configs_compatible_for_carryforward(current_config, prev_config):
            return r
    return None


def _split_candidate_pairs(
    candidates: list[dict], unchanged_sources: set[str],
) -> tuple[list[dict], list[dict]]:
    """Split candidate pairs into reusable and new.

    Reusable: both chunk sources (canonicalized) are in unchanged_sources.
    New: at least one source is changed or new.
    """
    reusable = []
    new = []
    for pair in candidates:
        src_a = _canonicalize_source(pair["chunk_a"].get("source", ""))
        src_b = _canonicalize_source(pair["chunk_b"].get("source", ""))
        if src_a in unchanged_sources and src_b in unchanged_sources:
            reusable.append(pair)
        else:
            new.append(pair)
    return reusable, new


def _carry_forward_findings(
    db: Session,
    review_id: int,
    previous_review_id: int,
    reusable_pairs: list[dict],
    unchanged_sources: set[str],
    analysis_types: list[str],
) -> int:
    """Copy findings from previous review for unchanged document pairs.

    Returns count of carried-forward issues.
    """
    # Build set of reusable (doc_a, doc_b) canonical pairs
    reusable_doc_pairs = set()
    for pair in reusable_pairs:
        a = _canonicalize_source(pair["chunk_a"].get("source", ""))
        b = _canonicalize_source(pair["chunk_b"].get("source", ""))
        reusable_doc_pairs.add((min(a, b), max(a, b)))

    # Query previous review's non-structural issues
    prev_issues = (
        db.query(KbReviewIssue)
        .filter(
            KbReviewIssue.review_id == previous_review_id,
            KbReviewIssue.judges_total != 0,  # skip structural
        )
        .all()
    )

    carried = 0
    for prev_issue in prev_issues:
        doc_a = _canonicalize_source(prev_issue.doc_a_name or "")
        doc_b = _canonicalize_source(prev_issue.doc_b_name or "") if prev_issue.doc_b_name else None

        should_carry = False
        if doc_b is not None:
            # Pair-based issue: carry forward if both docs are unchanged
            # and the canonical pair is in the reusable set
            pair_key = (min(doc_a, doc_b), max(doc_a, doc_b))
            if pair_key in reusable_doc_pairs and doc_a in unchanged_sources and doc_b in unchanged_sources:
                should_carry = True
        else:
            # Ambiguity issue (doc_b is NULL): carry forward if doc_a is
            # unchanged and AMBIGUITY is in analysis_types.
            # NOTE: Edge case — if disambiguating context was in a removed doc,
            # the ambiguity status could have changed. Not worth solving now.
            if "AMBIGUITY" in analysis_types and doc_a in unchanged_sources:
                should_carry = True

        if not should_carry:
            continue

        # Determine original_review_id (chain to first discovery)
        original_id = prev_issue.original_review_id or previous_review_id

        # Copy issue
        # Carry all statuses including DISMISSED. Rationale: if docs didn't change,
        # the finding is byte-for-byte identical — re-dismissing the same false
        # positive every review is pure busywork and alert fatigue. RESOLVED is
        # obvious (fix is still there). ACKNOWLEDGED carries too.
        new_issue = KbReviewIssue(
            review_id=review_id,
            issue_type=prev_issue.issue_type,
            severity=prev_issue.severity,
            confidence=prev_issue.confidence,
            title=prev_issue.title,
            description=prev_issue.description,
            doc_a_name=prev_issue.doc_a_name,
            doc_a_page=prev_issue.doc_a_page,
            doc_a_excerpt=prev_issue.doc_a_excerpt,
            doc_b_name=prev_issue.doc_b_name,
            doc_b_page=prev_issue.doc_b_page,
            doc_b_excerpt=prev_issue.doc_b_excerpt,
            entities_involved=prev_issue.entities_involved,
            consensus=prev_issue.consensus,
            judges_flagged=prev_issue.judges_flagged,
            judges_total=prev_issue.judges_total,
            carried_forward=True,
            original_review_id=original_id,
            status=getattr(prev_issue, "status", "OPEN"),
            status_updated_by=getattr(prev_issue, "status_updated_by", None),
            status_updated_at=getattr(prev_issue, "status_updated_at", None),
            status_note=getattr(prev_issue, "status_note", None),
        )
        db.add(new_issue)
        db.flush()  # get new_issue.id

        # Copy associated judge results
        prev_results = (
            db.query(KbReviewJudgeResult)
            .filter(KbReviewJudgeResult.issue_id == prev_issue.id)
            .all()
        )
        for pr in prev_results:
            jr = KbReviewJudgeResult(
                issue_id=new_issue.id,
                judge_index=pr.judge_index,
                judge_provider=pr.judge_provider,
                judge_model=pr.judge_model,
                detected=pr.detected,
                severity=pr.severity,
                confidence=pr.confidence,
                reasoning=pr.reasoning,
                input_tokens=0,
                output_tokens=0,
            )
            db.add(jr)

        carried += 1

    db.commit()
    return carried


# ── Parallel execution helpers ───────────────────────────────────────────


@dataclass
class ProgressTracker:
    """Tracks progress across concurrent judges within Phase 3 (30-85%)."""
    total_work: int       # num_judges * (len(batches) + 1 for ambiguity)
    completed: int = 0
    issues_persisted: int = 0

    @property
    def progress(self) -> int:
        if self.total_work == 0:
            return 85
        return 30 + int(self.completed / self.total_work * 55)


async def _run_single_judge(
    j_idx: int,
    judge,
    jc: JudgeConfig,
    batches: list[list[dict]],
    ambiguity_sample: list[dict],
    pair_types: list[str],
    db: Session,
    review_id: int,
    num_judges: int,
    key_to_issue: dict[str, dict],
    db_lock: asyncio.Lock,
    tracker: ProgressTracker,
) -> dict:
    """Run a single judge's full analysis (pairs + ambiguity). Returns judge stats."""
    extras = ""
    if jc.reasoning_effort:
        extras += f", reasoning={jc.reasoning_effort}"
    if jc.max_tokens:
        extras += f", max_tokens={jc.max_tokens}"
    logger.info(f"Review {review_id}: Judge {j_idx} starting — {jc.provider}/{jc.model}{extras}")

    j_started = datetime.now()
    j_start = time.monotonic()
    j_tokens_in = 0
    j_tokens_out = 0
    j_llm_calls = 0
    j_findings_count = 0

    # ── Pair analysis ────────────────────────────────────────────────
    if pair_types and batches:
        # Probe batch 0: detect config/auth errors early before burning credits
        try:
            t0 = time.monotonic()
            result = await analyze_pair_batch(
                judge, batches[0], pair_types,
                judge_index=j_idx, judge_provider=jc.provider, judge_model=jc.model,
            )
            elapsed = (time.monotonic() - t0) * 1000
            logger.info(f"Judge {j_idx} [{jc.provider}/{jc.model}] probe batch: {elapsed:.0f}ms")
        except Exception as e:
            logger.error(
                f"Review {review_id}: Judge {j_idx} ({jc.provider}/{jc.model}) "
                f"failed on probe batch — aborting judge: {e}"
            )
            raise

        # Persist probe batch findings
        batch_findings = result["findings"]
        j_tokens_in += result["input_tokens"]
        j_tokens_out += result["output_tokens"]
        j_llm_calls += 1
        j_findings_count += len(batch_findings)

        async with db_lock:
            new_count = _persist_batch_findings(db, review_id, batch_findings, num_judges, key_to_issue)
            tracker.completed += 1
            tracker.issues_persisted += new_count
            _update_progress(db, review_id, min(tracker.progress, 84),
                             issues_found=tracker.issues_persisted)

        # Remaining batches — concurrent, rate-limited by InMemoryRateLimiter.
        # Each batch analyzes and persists immediately on completion so that
        # findings are durable even if the process crashes mid-flight.
        if len(batches) > 1:
            async def _analyze_and_persist(batch):
                """Analyze a single batch and persist its findings immediately."""
                try:
                    result = await analyze_pair_batch(
                        judge, batch, pair_types,
                        judge_index=j_idx, judge_provider=jc.provider, judge_model=jc.model,
                    )
                except Exception:
                    # Still count failed batches toward progress
                    async with db_lock:
                        tracker.completed += 1
                        _update_progress(db, review_id, min(tracker.progress, 84),
                                         issues_found=tracker.issues_persisted)
                    raise
                async with db_lock:
                    new_count = _persist_batch_findings(
                        db, review_id, result["findings"], num_judges, key_to_issue,
                    )
                    tracker.completed += 1
                    tracker.issues_persisted += new_count
                    _update_progress(db, review_id, min(tracker.progress, 84),
                                     issues_found=tracker.issues_persisted)
                return result

            remaining_results = await asyncio.gather(
                *[_analyze_and_persist(batch) for batch in batches[1:]],
                return_exceptions=True,
            )

            for batch_idx_offset, batch_result in enumerate(remaining_results):
                batch_idx = batch_idx_offset + 1
                j_llm_calls += 1  # count the attempt regardless of outcome
                if isinstance(batch_result, Exception):
                    logger.warning(
                        f"Review {review_id}: Judge {j_idx} [{jc.provider}/{jc.model}] "
                        f"batch {batch_idx} failed, skipping: {batch_result}"
                    )
                    continue

                j_tokens_in += batch_result["input_tokens"]
                j_tokens_out += batch_result["output_tokens"]
                j_findings_count += len(batch_result["findings"])
    elif pair_types:
        # No batches (no candidate pairs) — nothing to do
        pass

    # ── Ambiguity analysis ───────────────────────────────────────────
    if ambiguity_sample:
        logger.info(f"Review {review_id}: Judge {j_idx} [{jc.provider}/{jc.model}] ambiguity scan")
        result = await analyze_ambiguity_batch(
            judge, ambiguity_sample,
            judge_index=j_idx, judge_provider=jc.provider, judge_model=jc.model,
        )
        ambiguity_findings = result["findings"]
        j_tokens_in += result["input_tokens"]
        j_tokens_out += result["output_tokens"]
        j_llm_calls += result["llm_calls"]
        j_findings_count += len(ambiguity_findings)

        async with db_lock:
            new_count = _persist_batch_findings(db, review_id, ambiguity_findings, num_judges, key_to_issue)
            tracker.completed += 1
            tracker.issues_persisted += new_count
            _update_progress(db, review_id, min(tracker.progress, 84),
                             issues_found=tracker.issues_persisted)

    # ── Persist judge stats ──────────────────────────────────────────
    j_completed = datetime.now()
    j_duration_ms = int((time.monotonic() - j_start) * 1000)

    async with db_lock:
        stat = KbReviewJudgeStat(
            review_id=review_id,
            judge_index=j_idx,
            judge_provider=jc.provider,
            judge_model=jc.model,
            total_input_tokens=j_tokens_in,
            total_output_tokens=j_tokens_out,
            total_llm_calls=j_llm_calls,
            total_findings=j_findings_count,
            duration_ms=j_duration_ms,
            started_at=j_started,
            completed_at=j_completed,
        )
        db.add(stat)
        db.commit()

    logger.info(
        f"Review {review_id}: Judge {j_idx} [{jc.provider}/{jc.model}] "
        f"completed in {j_duration_ms}ms — {j_findings_count} findings, {j_llm_calls} LLM calls"
    )

    return {
        "judge_index": j_idx,
        "findings_count": j_findings_count,
        "duration_ms": j_duration_ms,
    }


async def _run_judges_parallel(
    judges: list,
    judge_configs: list[JudgeConfig],
    batches: list[list[dict]],
    ambiguity_sample: list[dict],
    pair_types: list[str],
    db: Session,
    review_id: int,
    num_judges: int,
    key_to_issue: dict[str, dict],
    starting_issues: int = 0,
) -> int:
    """Run all judges concurrently. Returns total issues_persisted (including starting_issues)."""
    db_lock = asyncio.Lock()
    has_ambiguity = 1 if ambiguity_sample else 0
    has_pairs = len(batches) if pair_types else 0
    tracker = ProgressTracker(
        total_work=num_judges * (has_pairs + has_ambiguity),
        issues_persisted=starting_issues,
    )

    judge_coros = [
        _run_single_judge(
            j_idx, judges[j_idx], judge_configs[j_idx],
            batches, ambiguity_sample, pair_types,
            db, review_id, num_judges, key_to_issue,
            db_lock, tracker,
        )
        for j_idx in range(num_judges)
    ]

    results = await asyncio.gather(*judge_coros, return_exceptions=True)

    # ── Quorum check ─────────────────────────────────────────────────
    succeeded = []
    failed = []
    for j_idx, result in enumerate(results):
        if isinstance(result, Exception):
            failed.append((j_idx, result))
        else:
            succeeded.append(result)

    if failed:
        for j_idx, exc in failed:
            jc = judge_configs[j_idx]
            logger.warning(
                f"Review {review_id}: Judge {j_idx} ({jc.provider}/{jc.model}) failed: {exc}"
            )

    if not succeeded:
        # All judges failed — no quorum
        raise AllJudgesFailedError(
            f"All {num_judges} judge(s) failed — no results produced. "
            f"First error: {failed[0][1]}"
        )

    if failed:
        logger.warning(
            f"Review {review_id}: {len(failed)}/{num_judges} judges failed, "
            f"{len(succeeded)} succeeded (quorum met)"
        )

    return tracker.issues_persisted


# ── Main pipeline entry point ────────────────────────────────────────────


def run_review(review_id: int):
    """Main pipeline entry point. Runs synchronously (called from background task)."""
    db: Session = SessionLocal()
    review = None
    issues_persisted = 0
    try:
        review = db.query(KbReview).filter(KbReview.id == review_id).first()
        if not review:
            logger.error(f"Review {review_id} not found")
            return

        config = json.loads(review.config_json)
        judge_configs = [JudgeConfig(**j) for j in config["judges"]]
        analysis_types = config.get("analysis_types", ["CONTRADICTION", "ENTITY_INCONSISTENCY"])
        similarity_threshold = config.get("similarity_threshold", 0.85)
        max_candidate_pairs = config.get("max_candidate_pairs", 50)

        review.status = "RUNNING"
        review.started_at = datetime.now()
        db.commit()

        # ── Phase 1: Load data from Neo4j (0-10%) ──────────────────────
        _update_progress(db, review_id, 2)
        logger.info(f"Review {review_id}: Phase 1 — Loading data")

        agent = _lookup_agent(db, review.agent_id)
        if not agent or not agent.tenant_id:
            _update_progress(db, review_id, 0, status="FAILED",
                             error_message=f"Agent {review.agent_id} not found or has no tenant_id",
                             completed_at=datetime.now())
            return

        tenant_id = agent.tenant_id
        chunks = neo4j_reader.get_document_chunks(tenant_id)
        entity_mappings = neo4j_reader.get_entity_chunk_mappings(tenant_id)
        entity_relationships = neo4j_reader.get_entity_relationships(tenant_id)

        # Count distinct source documents
        source_docs = set(c["source"] for c in chunks)
        review.total_documents = len(source_docs)
        review.total_chunks = len(chunks)
        db.commit()

        _update_progress(db, review_id, 10)
        logger.info(f"Review {review_id}: Loaded {len(chunks)} chunks from {len(source_docs)} documents")

        if not chunks:
            _update_progress(db, review_id, 100, status="COMPLETED",
                             completed_at=datetime.now(),
                             issues_found=0)
            return

        # ── URL deduplication (between Phase 1 and Phase 1.5) ─────────
        deduped_chunks, url_groups = deduplicate_url_chunks(chunks)
        if url_groups:
            canonical_docs = set(_canonicalize_source(c["source"]) for c in chunks)
            review.total_documents = len(canonical_docs)
            review.url_duplicates_removed = len(chunks) - len(deduped_chunks)
            db.commit()
            logger.info(
                f"Review {review_id}: URL dedup — {len(chunks)} chunks → "
                f"{len(deduped_chunks)} after removing {len(chunks) - len(deduped_chunks)} "
                f"URL-variant duplicates ({len(url_groups)} groups)"
            )

        # ── Phase 1.5: Training quality checks (10-15%) ───────────────
        # Use original chunks for structural checks (report ALL issues),
        # but pass url_groups so URL_DUPLICATION issues are also reported.
        logger.info(f"Review {review_id}: Phase 1.5 — Training quality")
        structural_issues = run_training_quality_checks(chunks, url_groups=url_groups)
        _update_progress(db, review_id, 15,
                         chunks_with_issues=sum(1 for c in chunks
                                                if c.get("embedding") is None or not (c.get("text") or "").strip()))

        # Persist structural issues immediately
        for si in structural_issues:
            issue = KbReviewIssue(
                review_id=review_id,
                issue_type=si["issue_type"],
                severity=si["severity"],
                confidence=si["confidence"],
                title=si["title"],
                description=si["description"],
                doc_a_name=si.get("doc_a_name"),
                doc_a_page=si.get("doc_a_page"),
                doc_a_excerpt=si.get("doc_a_excerpt"),
                consensus=si["consensus"],
                judges_flagged=0,
                judges_total=0,
            )
            db.add(issue)
        if structural_issues:
            db.commit()
            issues_persisted += len(structural_issues)
            _update_progress(db, review_id, 15, issues_found=issues_persisted)

        # ── Change detection (after Phase 1, before Phase 2) ──────────
        # Always compute hashes so Phase 5 can store them for future reviews,
        # even when carryforward is disabled for this run.
        carryforward_enabled = config.get("carryforward", True)
        has_previous = False
        previous_review = None
        unchanged_sources: set[str] = set()
        current_hashes = _compute_document_hashes(deduped_chunks)

        if carryforward_enabled:
            previous_review = _find_previous_review(db, review.agent_id, config)
            if previous_review:
                # Load stored hashes for this agent
                stored_rows = (
                    db.query(KbReviewDocHash)
                    .filter(KbReviewDocHash.agent_id == review.agent_id)
                    .all()
                )
                previous_hashes = {r.source_canonical: r.content_hash for r in stored_rows}

                unchanged_sources, changed_sources, new_sources = _classify_documents(
                    current_hashes, previous_hashes,
                )

                has_previous = True
                review.previous_review_id = previous_review.id
                review.docs_changed = len(changed_sources) + len(new_sources)
                review.docs_unchanged = len(unchanged_sources)
                db.commit()

                logger.info(
                    f"Review {review_id}: Change detection — "
                    f"{len(unchanged_sources)} unchanged, {len(changed_sources)} changed, "
                    f"{len(new_sources)} new (previous review {previous_review.id})"
                )
            else:
                logger.info(f"Review {review_id}: No compatible previous review found — full analysis")
        else:
            logger.info(f"Review {review_id}: Carry-forward disabled — full analysis")

        # ── Phase 2: Pre-filter candidate pairs (15-25%) ──────────────
        # Use deduped_chunks so duplicate-URL pairs don't waste LLM calls.
        logger.info(f"Review {review_id}: Phase 2 — Pre-filtering")
        candidates = build_candidate_pairs(
            deduped_chunks, entity_mappings, entity_relationships,
            similarity_threshold=similarity_threshold,
            max_pairs=max_candidate_pairs,
        )
        review.candidate_pairs = len(candidates)
        db.commit()
        _update_progress(db, review_id, 25)
        logger.info(f"Review {review_id}: {len(candidates)} candidate pairs")

        # ── Phase 2.5: Carry-forward (25-30%) ────────────────────────
        reusable_pairs: list[dict] = []
        new_pairs = candidates  # default: all pairs are new
        carried_forward_count = 0

        if has_previous and previous_review:
            reusable_pairs, new_pairs = _split_candidate_pairs(candidates, unchanged_sources)

            prev_issue_count = (
                db.query(KbReviewIssue)
                .filter(
                    KbReviewIssue.review_id == previous_review.id,
                    KbReviewIssue.judges_total != 0,
                )
                .count()
            )
            logger.info(
                f"Review {review_id}: Carry-forward: {len(reusable_pairs)} reusable of "
                f"{len(candidates)} candidates ({prev_issue_count} issues in review "
                f"{previous_review.id})"
            )

            if reusable_pairs:
                carried_forward_count = _carry_forward_findings(
                    db, review_id, previous_review.id,
                    reusable_pairs, unchanged_sources,
                    analysis_types,
                )
                issues_persisted += carried_forward_count
                logger.info(
                    f"Review {review_id}: Carried forward {carried_forward_count} issues"
                )

            review.pairs_reused = len(reusable_pairs)
            review.pairs_analyzed = len(new_pairs)
            db.commit()
        else:
            review.pairs_reused = 0
            review.pairs_analyzed = len(candidates)
            db.commit()

        _update_progress(db, review_id, 30, issues_found=issues_persisted)

        # ── Phase 3: LLM judge analysis (30-85%) ──────────────────────
        logger.info(f"Review {review_id}: Phase 3 — LLM judges (parallel)")
        num_judges = len(judge_configs)

        # Build rate limiters — one per (provider, model), shared across judges
        # using the same underlying API endpoint + model combination.
        # LiteLLM/OpenRouter are proxies; rate limits apply at the model level.
        model_limiters: dict[tuple[str, str], InMemoryRateLimiter] = {}
        for jc in judge_configs:
            limiter_key = (jc.provider, jc.model)
            if jc.rate_limit_rpm and limiter_key not in model_limiters:
                model_limiters[limiter_key] = InMemoryRateLimiter(
                    requests_per_second=jc.rate_limit_rpm / 60,
                    check_every_n_seconds=0.1,
                    max_bucket_size=max(jc.rate_limit_rpm // 6, 2),
                )

        judges = [
            create_judge(jc, rate_limiter=model_limiters.get((jc.provider, jc.model)))
            for jc in judge_configs
        ]

        # Split new_pairs (not reusable) into batches
        batches = [new_pairs[i:i + BATCH_SIZE] for i in range(0, len(new_pairs), BATCH_SIZE)]

        # Pair analysis types (all except AMBIGUITY which is per-chunk)
        pair_types = [t for t in analysis_types if t != "AMBIGUITY"]

        # Prepare ambiguity sample once (used by all judges)
        ambiguity_sample = []
        if "AMBIGUITY" in analysis_types:
            by_doc: dict[str, list[dict]] = defaultdict(list)
            for c in chunks:
                if (c.get("text") or "").strip():
                    by_doc[c["source"]].append(c)
            for doc_chunks in by_doc.values():
                # If carry-forward is active, only include chunks from
                # changed/new docs — unchanged docs' ambiguity findings
                # are carried forward.
                if has_previous:
                    canon = _canonicalize_source(doc_chunks[0]["source"])
                    if canon in unchanged_sources:
                        continue
                ambiguity_sample.extend(doc_chunks[:3])
            ambiguity_sample = ambiguity_sample[:50]

        key_to_issue: dict[str, dict] = {}  # finding_key → {"issue_id": int, "judges": set}
        fallback_judge_defs = config.get("fallback_judges")

        # If all pairs were reused, skip LLM entirely
        if not new_pairs and not ambiguity_sample:
            if carried_forward_count > 0:
                logger.info(
                    f"Review {review_id}: All documents unchanged — carried forward "
                    f"{carried_forward_count} findings, 0 LLM calls"
                )
            else:
                logger.info(
                    f"Review {review_id}: No candidate pairs found and no ambiguity sample — "
                    f"skipping LLM analysis (0 LLM calls). "
                    f"Possible cause: missing embeddings in Neo4j or similarity_threshold too high."
                )
            _update_progress(db, review_id, 85, issues_found=issues_persisted)
        else:
            # Run judges in parallel
            # NOTE: asyncio.run() is safe here because background_tasks runs us in a
            # threadpool thread with no event loop. If run_review() is ever called from
            # an async context, this must change to `await _run_judges_parallel(...)`.
            try:
                llm_issues = asyncio.run(_run_judges_parallel(
                    judges, judge_configs, batches, ambiguity_sample, pair_types,
                    db, review_id, num_judges, key_to_issue,
                    starting_issues=issues_persisted,
                ))
            except AllJudgesFailedError:
                if not fallback_judge_defs:
                    raise
                # Primary judges all failed — retry with fallback judges.
                # AllJudgesFailedError only fires when every judge failed on probe
                # batch, so zero findings were persisted and key_to_issue is empty.
                logger.warning(
                    f"Review {review_id}: All primary judges failed, "
                    f"activating fallback judges"
                )
                key_to_issue.clear()  # safety measure

                # Clean up primary judge artifacts: stats, judge results, and
                # LLM-generated issues. Structural issues (Phase 1.5) have
                # consensus='STRUCTURAL' and must be preserved — they were
                # persisted before Phase 3 and are unrelated to judge failures.
                # Carried-forward issues must also be preserved.
                llm_issue_ids = (
                    db.query(KbReviewIssue.id)
                    .filter(KbReviewIssue.review_id == review_id)
                    .filter(KbReviewIssue.consensus != "STRUCTURAL")
                    .filter(KbReviewIssue.carried_forward == False)
                )
                db.query(KbReviewJudgeResult).filter(
                    KbReviewJudgeResult.issue_id.in_(llm_issue_ids)
                ).delete(synchronize_session=False)
                db.query(KbReviewIssue).filter(
                    KbReviewIssue.review_id == review_id,
                    KbReviewIssue.consensus != "STRUCTURAL",
                    KbReviewIssue.carried_forward == False,
                ).delete(synchronize_session=False)
                db.query(KbReviewJudgeStat).filter(
                    KbReviewJudgeStat.review_id == review_id
                ).delete(synchronize_session=False)
                db.commit()
                # Reset to structural + carried-forward count
                issues_persisted = (
                    db.query(KbReviewIssue)
                    .filter(KbReviewIssue.review_id == review_id)
                    .count()
                )

                # Rebuild with fallback config
                judge_configs = [JudgeConfig(**j) for j in fallback_judge_defs]
                num_judges = len(judge_configs)

                model_limiters = {}
                for jc in judge_configs:
                    limiter_key = (jc.provider, jc.model)
                    if jc.rate_limit_rpm and limiter_key not in model_limiters:
                        model_limiters[limiter_key] = InMemoryRateLimiter(
                            requests_per_second=jc.rate_limit_rpm / 60,
                            check_every_n_seconds=0.1,
                            max_bucket_size=max(jc.rate_limit_rpm // 6, 2),
                        )

                judges = [
                    create_judge(jc, rate_limiter=model_limiters.get((jc.provider, jc.model)))
                    for jc in judge_configs
                ]

                # Second asyncio.run() is safe: the first event loop was closed
                # when AllJudgesFailedError propagated out of asyncio.run().
                llm_issues = asyncio.run(_run_judges_parallel(
                    judges, judge_configs, batches, ambiguity_sample, pair_types,
                    db, review_id, num_judges, key_to_issue,
                    starting_issues=issues_persisted,
                ))
                # If fallback also fails, AllJudgesFailedError propagates to the
                # outer try/except which marks the review FAILED or PARTIAL.

            # llm_issues already includes starting_issues (structural + carried-forward)
            issues_persisted = llm_issues

            _update_progress(db, review_id, 85, issues_found=issues_persisted)

        # ── Phase 4: Update consensus from DB (85-95%) ─────────────────
        logger.info(f"Review {review_id}: Phase 4 — Consensus")
        update_consensus_for_review(
            db, review_id, num_judges,
            exclude_carried_forward=bool(carried_forward_count),
        )
        _update_progress(db, review_id, 95)

        # ── Phase 5: Finalize review + upsert doc hashes (95-100%) ─────
        logger.info(f"Review {review_id}: Phase 5 — Finalizing")

        # Upsert document hashes for this agent (always, even when
        # carryforward is disabled — keeps hashes fresh for future runs)
        if current_hashes:
            # Sort by source_canonical for consistent lock ordering
            for source_canonical, hash_val in sorted(current_hashes.items()):
                existing = (
                    db.query(KbReviewDocHash)
                    .filter(
                        KbReviewDocHash.agent_id == review.agent_id,
                        KbReviewDocHash.source_canonical == source_canonical,
                    )
                    .first()
                )
                if existing:
                    existing.content_hash = hash_val
                    existing.review_id = review_id
                    existing.updated_at = datetime.now()
                else:
                    db.add(KbReviewDocHash(
                        agent_id=review.agent_id,
                        source_canonical=source_canonical,
                        content_hash=hash_val,
                        review_id=review_id,
                    ))

            # Delete hashes for removed sources
            if has_previous:
                stored_rows = (
                    db.query(KbReviewDocHash)
                    .filter(KbReviewDocHash.agent_id == review.agent_id)
                    .all()
                )
                for row in stored_rows:
                    if row.source_canonical not in current_hashes:
                        db.delete(row)

            db.commit()

        review = db.query(KbReview).filter(KbReview.id == review_id).first()
        # Final issues_found excludes MINORITY and RESOLVED/DISMISSED so it
        # matches the default API response (list endpoint reads this column).
        non_minority_active = (
            db.query(KbReviewIssue)
            .filter(
                KbReviewIssue.review_id == review_id,
                KbReviewIssue.consensus != "MINORITY",
                KbReviewIssue.status.in_(["OPEN", "ACKNOWLEDGED"]),
            )
            .count()
        )
        non_minority_resolved = (
            db.query(KbReviewIssue)
            .filter(
                KbReviewIssue.review_id == review_id,
                KbReviewIssue.consensus != "MINORITY",
                KbReviewIssue.status.in_(["RESOLVED", "DISMISSED"]),
            )
            .count()
        )
        review.issues_found = non_minority_active
        review.issues_resolved = non_minority_resolved
        review.status = "COMPLETED"
        review.progress = 100
        review.completed_at = datetime.now()
        db.commit()

        logger.info(f"Review {review_id}: COMPLETED — {issues_persisted} issues found")

    except Exception as e:
        logger.exception(f"Review {review_id} FAILED: {e}")
        try:
            status = "PARTIAL" if issues_persisted > 0 else "FAILED"
            current_progress = review.progress if review and review.progress else 0
            # Recount excluding MINORITY for consistency with default API response
            active_count = (
                db.query(KbReviewIssue)
                .filter(
                    KbReviewIssue.review_id == review_id,
                    KbReviewIssue.consensus != "MINORITY",
                    KbReviewIssue.status.in_(["OPEN", "ACKNOWLEDGED"]),
                )
                .count()
            )
            resolved_count = (
                db.query(KbReviewIssue)
                .filter(
                    KbReviewIssue.review_id == review_id,
                    KbReviewIssue.consensus != "MINORITY",
                    KbReviewIssue.status.in_(["RESOLVED", "DISMISSED"]),
                )
                .count()
            )
            _update_progress(db, review_id, current_progress,
                             status=status, error_message=str(e)[:2000],
                             issues_found=active_count,
                             issues_resolved=resolved_count,
                             completed_at=datetime.now())
        except Exception:
            logger.exception(f"Review {review_id}: failed to update {status} status")
    finally:
        db.close()
