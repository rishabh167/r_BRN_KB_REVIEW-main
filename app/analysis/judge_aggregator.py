"""Phase 4 — N-judge majority voting consensus aggregation."""

import logging
from collections import defaultdict

from sqlalchemy.orm import Session

from app.analysis.pre_filter import _canonicalize_source

logger = logging.getLogger("kb_review")


def finding_key(f: dict) -> str:
    """Create a stable key for matching findings across judges.

    For pair-based issues: (sorted chunk identifiers, issue_type)
    For ambiguity: (chunk_source|page|split_id, issue_type, truncated_title)

    Sources are canonicalized so that URL variants (e.g. broadnet.ai/x
    vs www.broadnet.ai/x) produce the same key.
    """
    if f.get("chunk_b") is not None:
        chunk_a = f.get("chunk_a", {})
        chunk_b = f.get("chunk_b", {})
        a_id = f"{_canonicalize_source(chunk_a.get('source', ''))}|{chunk_a.get('page', '')}|{chunk_a.get('split_id', '')}"
        b_id = f"{_canonicalize_source(chunk_b.get('source', ''))}|{chunk_b.get('page', '')}|{chunk_b.get('split_id', '')}"
        pair_id = "||".join(sorted([a_id, b_id]))
        return f"{pair_id}|{f.get('issue_type', '')}"

    chunk_a = f.get("chunk_a", {})
    return f"{_canonicalize_source(chunk_a.get('source', ''))}|{chunk_a.get('page', '')}|{chunk_a.get('split_id', '')}|{f['issue_type']}|{f.get('title', '')[:60]}"


def aggregate_judge_findings(
    all_findings: list[list[dict]],
    num_judges: int,
) -> list[dict]:
    """Aggregate findings from N judges using majority voting.

    Args:
        all_findings: List of per-judge finding lists.
                      all_findings[j] = findings from judge j.
        num_judges: Total number of judges configured.

    Returns:
        List of aggregated issue dicts ready for DB persistence.
    """
    # Group findings by key
    grouped: dict[str, list[dict]] = defaultdict(list)
    for findings in all_findings:
        for f in findings:
            key = finding_key(f)
            grouped[key].append(f)

    aggregated = []
    for key, findings in grouped.items():
        judges_flagged = len(findings)

        # Determine consensus
        if num_judges == 1:
            consensus = "SINGLE_JUDGE"
        elif judges_flagged == num_judges:
            consensus = "UNANIMOUS"
        elif judges_flagged > num_judges / 2:
            consensus = "MAJORITY"
        else:
            consensus = "MINORITY"

        # Aggregate confidence
        confidences = [f["confidence"] for f in findings]
        if consensus == "UNANIMOUS":
            agg_confidence = max(confidences)
        elif consensus == "MAJORITY":
            agg_confidence = sum(confidences) / len(confidences) * 0.9
        elif consensus == "MINORITY":
            agg_confidence = sum(confidences) / len(confidences) * 0.6
        else:
            agg_confidence = confidences[0]

        # Use the highest-confidence finding as the representative
        best = max(findings, key=lambda f: f["confidence"])

        # Aggregate severity — take the most severe
        severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
        worst_severity = max(findings, key=lambda f: severity_order.get(f.get("severity", "LOW"), 0))

        chunk_a = best.get("chunk_a", {})
        chunk_b = best.get("chunk_b")

        issue = {
            "issue_type": best["issue_type"],
            "severity": worst_severity["severity"],
            "confidence": round(min(agg_confidence, 1.0), 3),
            "title": best["title"],
            "description": best["description"],
            "doc_a_name": chunk_a.get("source"),
            "doc_a_page": chunk_a.get("page"),
            "doc_a_excerpt": best.get("claim_a") or (chunk_a.get("text") or "")[:500],
            "doc_b_name": chunk_b.get("source") if chunk_b else None,
            "doc_b_page": chunk_b.get("page") if chunk_b else None,
            "doc_b_excerpt": best.get("claim_b") or ((chunk_b.get("text") or "")[:500] if chunk_b else None),
            "entities_involved": best.get("entities", []),
            "consensus": consensus,
            "judges_flagged": judges_flagged,
            "judges_total": num_judges,
            "judge_details": findings,
        }
        aggregated.append(issue)

    # Sort: UNANIMOUS first, then by severity, then confidence
    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    consensus_rank = {"UNANIMOUS": 0, "MAJORITY": 1, "SINGLE_JUDGE": 2, "MINORITY": 3}

    aggregated.sort(key=lambda x: (
        consensus_rank.get(x["consensus"], 9),
        severity_rank.get(x["severity"], 9),
        -x["confidence"],
    ))

    logger.info(
        f"Aggregated {sum(len(f) for f in all_findings)} raw findings "
        f"from {num_judges} judges into {len(aggregated)} issues"
    )
    return aggregated


def update_consensus_for_review(
    db: Session, review_id: int, num_judges: int,
    exclude_carried_forward: bool = False,
):
    """Update consensus, confidence, and severity on all issues for a review.

    Replaces in-memory aggregation — reads judge_results from the DB and
    computes final values directly on persisted KbReviewIssue rows.

    When exclude_carried_forward=True, carried-forward issues are skipped
    so they keep their original consensus from the previous review.
    """
    from app.database_layer.db_models import KbReviewIssue, KbReviewJudgeResult

    query = db.query(KbReviewIssue).filter(
        KbReviewIssue.review_id == review_id,
        KbReviewIssue.judges_total != 0,  # skip structural issues
    )
    if exclude_carried_forward:
        query = query.filter(KbReviewIssue.carried_forward == False)
    issues = query.all()

    severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

    for issue in issues:
        issue.judges_total = num_judges
        flagged = issue.judges_flagged

        # Consensus
        if num_judges == 1:
            consensus = "SINGLE_JUDGE"
        elif flagged == num_judges:
            consensus = "UNANIMOUS"
        elif flagged > num_judges / 2:
            consensus = "MAJORITY"
        else:
            consensus = "MINORITY"
        issue.consensus = consensus

        # Confidence — aggregate from judge results
        results = db.query(KbReviewJudgeResult).filter(
            KbReviewJudgeResult.issue_id == issue.id,
        ).all()
        confidences = [r.confidence for r in results if r.confidence is not None]
        if confidences:
            if consensus == "UNANIMOUS":
                agg_confidence = max(confidences)
            elif consensus == "MAJORITY":
                agg_confidence = sum(confidences) / len(confidences) * 0.9
            elif consensus == "MINORITY":
                agg_confidence = sum(confidences) / len(confidences) * 0.6
            else:
                agg_confidence = confidences[0]
            issue.confidence = round(min(agg_confidence, 1.0), 3)

        # Severity — max across judge results
        severities = [r.severity for r in results if r.severity]
        if severities:
            issue.severity = max(severities, key=lambda s: severity_order.get(s, 0))

    db.commit()
    logger.info(f"Review {review_id}: updated consensus on {len(issues)} issues")
