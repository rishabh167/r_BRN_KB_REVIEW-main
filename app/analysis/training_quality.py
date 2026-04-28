"""Phase 1.5 — Structural training quality checks (no LLM needed)."""

import logging
from collections import defaultdict

logger = logging.getLogger("kb_review")


def check_missing_embeddings(chunks: list[dict]) -> list[dict]:
    """Detect chunks that have no embedding vector.

    Groups by source document and returns one issue per document.
    """
    missing_by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c.get("embedding") is None:
            missing_by_doc[c.get("source") or "Unknown Document"].append(c)

    issues = []
    for source, doc_chunks in missing_by_doc.items():
        pages = sorted({c.get("page") for c in doc_chunks if c.get("page") is not None})
        page_str = ", ".join(str(p) for p in pages[:10])
        if len(pages) > 10:
            page_str += f" ... ({len(pages)} total)"

        issues.append({
            "issue_type": "MISSING_EMBEDDINGS",
            "severity": "HIGH",
            "confidence": 1.0,
            "title": f"{len(doc_chunks)} sections from {source} lack embeddings",
            "description": (
                f"{len(doc_chunks)} sections from **{source}** have no embedding vectors. "
                f"The agent cannot search this content via vector retrieval. "
                f"Affected pages: {page_str}. Re-train these documents to generate embeddings."
            ),
            "doc_a_name": source,
            "doc_a_page": pages[0] if pages else None,
            "doc_a_excerpt": (doc_chunks[0].get("text") or "")[:500] if doc_chunks else None,
            "consensus": "STRUCTURAL",
        })

    return issues


def check_empty_content(chunks: list[dict]) -> list[dict]:
    """Detect chunks with empty or whitespace-only text."""
    empty_by_doc: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            empty_by_doc[c.get("source") or "Unknown Document"].append(c)

    issues = []
    for source, doc_chunks in empty_by_doc.items():
        pages = sorted({c.get("page") for c in doc_chunks if c.get("page") is not None})
        page_str = ", ".join(str(p) for p in pages[:10])
        if len(pages) > 10:
            page_str += f" ... ({len(pages)} total)"

        issues.append({
            "issue_type": "EMPTY_CONTENT",
            "severity": "HIGH",
            "confidence": 1.0,
            "title": f"{len(doc_chunks)} sections from {source} contain no text",
            "description": (
                f"{len(doc_chunks)} sections from **{source}** contain no readable text. "
                f"This may indicate failed OCR or empty pages. "
                f"Affected pages: {page_str}. Check the original document and re-upload if needed."
            ),
            "doc_a_name": source,
            "doc_a_page": pages[0] if pages else None,
            "doc_a_excerpt": None,
            "consensus": "STRUCTURAL",
        })

    return issues


def check_url_duplicates(url_groups: dict[str, list[str]]) -> list[dict]:
    """Report groups of chunks that are URL variants of the same page.

    Args:
        url_groups: from pre_filter.deduplicate_url_chunks(), maps
                    canonical_source -> [url_variant_1, url_variant_2, ...]
    """
    issues = []
    for canonical, variants in url_groups.items():
        url_list = "\n".join(f"- {v}" for v in variants)
        issues.append({
            "issue_type": "URL_DUPLICATION",
            "severity": "MEDIUM",
            "confidence": 1.0,
            "title": f"{len(variants)} URL variants for {canonical}",
            "description": (
                f"The following URLs resolve to the same page but were ingested as "
                f"separate documents:\n{url_list}\n\n"
                f"This creates duplicate content in the knowledge base and inflates "
                f"pair comparisons. Consider canonicalizing URLs in the scraper or "
                f"deduplicating before training."
            ),
            "doc_a_name": variants[0],
            "doc_a_page": None,
            "doc_a_excerpt": None,
            "consensus": "STRUCTURAL",
        })
    return issues


def run_training_quality_checks(
    chunks: list[dict],
    url_groups: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Run all structural quality checks and return combined issue dicts."""
    issues = []
    issues.extend(check_missing_embeddings(chunks))
    issues.extend(check_empty_content(chunks))
    if url_groups:
        issues.extend(check_url_duplicates(url_groups))
    logger.info(f"Training quality checks found {len(issues)} structural issues")
    return issues
