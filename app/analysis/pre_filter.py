"""Phase 2 — Pre-filter candidate pairs for LLM analysis.

Uses embedding cosine similarity and entity graph overlap to find
chunk pairs that are likely to contain contradictions or inconsistencies.
"""

import logging
from collections import defaultdict
from urllib.parse import urlparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("kb_review")


def _canonicalize_source(source: str) -> str:
    """Normalize URL-based sources to a canonical form.

    Strips http/https scheme differences and www prefix so that
    'http://broadnet.ai/solutions' and 'https://www.broadnet.ai/solutions'
    map to the same canonical key.

    Non-URL sources (filenames like 'report.pdf') are returned unchanged.
    """
    if not source:
        return source or ""
    # Only canonicalize if it looks like a URL
    if "://" in source or source.startswith("www."):
        parsed = urlparse(source if "://" in source else f"https://{source}")
        host = (parsed.netloc or "").lower().removeprefix("www.")
        path = parsed.path.rstrip("/")
        return f"{host}{path}"
    return source


def deduplicate_url_chunks(chunks: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """Remove duplicate chunks from URL-variant sources.

    Returns:
        - deduped_chunks: list with only one chunk per canonical URL
        - url_groups: dict mapping canonical_source -> list of original source URLs
          (only entries with 2+ variants, i.e., actual duplicates)
    """
    # Group chunks by canonical source
    canonical_groups: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        canon = _canonicalize_source(c.get("source", ""))
        canonical_groups[canon].append(c)

    # Track which canonical keys map to multiple distinct original sources
    url_groups: dict[str, list[str]] = {}
    deduped: list[dict] = []

    for canon, group in canonical_groups.items():
        distinct_sources = list(dict.fromkeys(c["source"] for c in group))

        if len(distinct_sources) > 1:
            # Multiple URL variants — record the group and keep only chunks from one variant
            url_groups[canon] = distinct_sources

            # Pick the variant with the most total text as the keeper
            text_by_source: dict[str, int] = defaultdict(int)
            for c in group:
                text_by_source[c["source"]] += len(c.get("text") or "")
            best_source = max(text_by_source, key=text_by_source.get)

            deduped.extend(c for c in group if c["source"] == best_source)
        else:
            deduped.extend(group)

    if url_groups:
        total_removed = len(chunks) - len(deduped)
        logger.info(
            f"URL dedup: {len(url_groups)} canonical URLs had variants, "
            f"removed {total_removed} duplicate chunks"
        )

    return deduped, url_groups


def _chunk_key(chunk: dict) -> str:
    return f"{chunk['source']}|{chunk['page']}|{chunk['split_id']}"


def find_similar_pairs(
    chunks: list[dict],
    threshold: float = 0.85,
    max_pairs: int = 200,
) -> list[tuple[dict, dict, float]]:
    """Find chunk pairs with high embedding similarity across different documents.

    Returns list of (chunk_a, chunk_b, similarity_score).
    """
    # Filter to chunks that have embeddings and non-empty text
    embedded = [
        c for c in chunks
        if c.get("embedding") is not None and (c.get("text") or "").strip()
    ]

    if len(embedded) < 2:
        return []

    # Group by source document
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(embedded):
        by_source[c["source"]].append(i)

    # Build embedding matrix
    embeddings = np.array([c["embedding"] for c in embedded])
    sim_matrix = cosine_similarity(embeddings)

    pairs = []
    sources = list(by_source.keys())

    # Compare chunks across different documents
    for i_src in range(len(sources)):
        for j_src in range(i_src + 1, len(sources)):
            for idx_a in by_source[sources[i_src]]:
                for idx_b in by_source[sources[j_src]]:
                    score = float(sim_matrix[idx_a][idx_b])
                    if score >= threshold:
                        pairs.append((embedded[idx_a], embedded[idx_b], score))

    # Sort by similarity descending, cap at max_pairs
    pairs.sort(key=lambda x: x[2], reverse=True)
    logger.info(f"Embedding similarity: {len(pairs)} pairs above threshold {threshold}")
    return pairs[:max_pairs]


def find_shared_entity_pairs(
    chunks: list[dict],
    entity_mappings: list[dict],
    min_shared_entities: int = 2,
) -> list[tuple[dict, dict, list[str]]]:
    """Find chunk pairs across documents that share multiple entities.

    Returns list of (chunk_a, chunk_b, shared_entity_ids).
    """
    # Build chunk lookup
    chunk_lookup: dict[str, dict] = {}
    for c in chunks:
        key = _chunk_key(c)
        chunk_lookup[key] = c

    # Build entity -> chunk keys mapping
    entity_to_chunks: dict[str, set[str]] = defaultdict(set)
    for m in entity_mappings:
        key = f"{m['source']}|{m['page']}|{m['split_id']}"
        entity_to_chunks[m["entity_id"]].add(key)

    # Build chunk_key -> entity set
    chunk_entities: dict[str, set[str]] = defaultdict(set)
    for m in entity_mappings:
        key = f"{m['source']}|{m['page']}|{m['split_id']}"
        chunk_entities[key].add(m["entity_id"])

    # Find cross-document pairs sharing entities
    pairs: dict[tuple[str, str], list[str]] = {}
    all_keys = list(chunk_entities.keys())

    for i in range(len(all_keys)):
        for j in range(i + 1, len(all_keys)):
            key_a, key_b = all_keys[i], all_keys[j]
            chunk_a = chunk_lookup.get(key_a)
            chunk_b = chunk_lookup.get(key_b)
            if not chunk_a or not chunk_b:
                continue
            # Only cross-document
            if chunk_a["source"] == chunk_b["source"]:
                continue

            shared = chunk_entities[key_a] & chunk_entities[key_b]
            if len(shared) >= min_shared_entities:
                pair_key = tuple(sorted([key_a, key_b]))
                if pair_key not in pairs:
                    pairs[pair_key] = list(shared)

    result = []
    for (key_a, key_b), shared_entities in pairs.items():
        chunk_a = chunk_lookup.get(key_a)
        chunk_b = chunk_lookup.get(key_b)
        if chunk_a and chunk_b:
            result.append((chunk_a, chunk_b, shared_entities))

    logger.info(f"Shared entity pairs: {len(result)} pairs with >= {min_shared_entities} shared entities")
    return result


def find_relationship_conflict_pairs(
    chunks: list[dict],
    entity_mappings: list[dict],
    entity_relationships: list[dict],
) -> list[tuple[dict, dict, list[str]]]:
    """Find chunks where the same entity pair has different relationship types.

    This suggests contradictory information about how entities relate.
    """
    # Group relationships by (source, target) pair
    pair_rels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in entity_relationships:
        key = (r["source_entity"], r["target_entity"])
        pair_rels[key].add(r["relationship_type"])

    # Find conflicting pairs (same entities, different relationships)
    conflicting_entity_pairs = {
        k: v for k, v in pair_rels.items() if len(v) > 1
    }

    if not conflicting_entity_pairs:
        return []

    # Map entities to chunks
    chunk_lookup: dict[str, dict] = {}
    for c in chunks:
        chunk_lookup[_chunk_key(c)] = c

    entity_to_chunks: dict[str, set[str]] = defaultdict(set)
    for m in entity_mappings:
        key = f"{m['source']}|{m['page']}|{m['split_id']}"
        entity_to_chunks[m["entity_id"]].add(key)

    results = []
    seen = set()
    for (src_ent, tgt_ent), rel_types in conflicting_entity_pairs.items():
        src_chunks = entity_to_chunks.get(src_ent, set())
        tgt_chunks = entity_to_chunks.get(tgt_ent, set())
        # Find chunks that mention both entities
        relevant_chunks = src_chunks & tgt_chunks
        chunk_keys = list(relevant_chunks)

        for i in range(len(chunk_keys)):
            for j in range(i + 1, len(chunk_keys)):
                a = chunk_lookup.get(chunk_keys[i])
                b = chunk_lookup.get(chunk_keys[j])
                if not a or not b:
                    continue
                if a["source"] == b["source"]:
                    continue
                pair_key = tuple(sorted([chunk_keys[i], chunk_keys[j]]))
                if pair_key not in seen:
                    seen.add(pair_key)
                    results.append((a, b, [src_ent, tgt_ent] + list(rel_types)))

    logger.info(f"Relationship conflict pairs: {len(results)}")
    return results


def build_candidate_pairs(
    chunks: list[dict],
    entity_mappings: list[dict],
    entity_relationships: list[dict],
    similarity_threshold: float = 0.85,
    max_pairs: int = 200,
) -> list[dict]:
    """Combine all pre-filter strategies and return deduplicated candidate pairs.

    Each candidate is a dict with chunk_a, chunk_b, reason, entities, score.
    """
    candidates: dict[tuple[str, str], dict] = {}

    # Strategy 1: Embedding similarity
    for chunk_a, chunk_b, score in find_similar_pairs(chunks, similarity_threshold, max_pairs):
        pair_key = tuple(sorted([_chunk_key(chunk_a), _chunk_key(chunk_b)]))
        if pair_key not in candidates:
            candidates[pair_key] = {
                "chunk_a": chunk_a,
                "chunk_b": chunk_b,
                "reasons": [],
                "entities": [],
                "score": score,
            }
        candidates[pair_key]["reasons"].append("embedding_similarity")

    # Strategy 2: Shared entities
    for chunk_a, chunk_b, shared in find_shared_entity_pairs(chunks, entity_mappings):
        pair_key = tuple(sorted([_chunk_key(chunk_a), _chunk_key(chunk_b)]))
        if pair_key not in candidates:
            candidates[pair_key] = {
                "chunk_a": chunk_a,
                "chunk_b": chunk_b,
                "reasons": [],
                "entities": [],
                "score": 0.0,
            }
        candidates[pair_key]["reasons"].append("shared_entities")
        candidates[pair_key]["entities"].extend(shared)

    # Strategy 3: Relationship conflicts
    for chunk_a, chunk_b, entities in find_relationship_conflict_pairs(
        chunks, entity_mappings, entity_relationships
    ):
        pair_key = tuple(sorted([_chunk_key(chunk_a), _chunk_key(chunk_b)]))
        if pair_key not in candidates:
            candidates[pair_key] = {
                "chunk_a": chunk_a,
                "chunk_b": chunk_b,
                "reasons": [],
                "entities": [],
                "score": 0.0,
            }
        candidates[pair_key]["reasons"].append("relationship_conflict")
        candidates[pair_key]["entities"].extend(entities)

    # Deduplicate entities per candidate
    for v in candidates.values():
        v["entities"] = list(set(v["entities"]))
        v["reasons"] = list(set(v["reasons"]))

    # Sort: relationship conflicts first, then shared entities, then similarity
    def sort_key(item):
        reasons = item["reasons"]
        priority = 0
        if "relationship_conflict" in reasons:
            priority += 3
        if "shared_entities" in reasons:
            priority += 2
        if "embedding_similarity" in reasons:
            priority += 1
        return (-priority, -item["score"])

    sorted_candidates = sorted(candidates.values(), key=sort_key)
    total_before_cap = len(sorted_candidates)
    result = sorted_candidates[:max_pairs]
    if total_before_cap > max_pairs:
        logger.info(f"Total candidate pairs after dedup: {total_before_cap}, capped to {max_pairs}")
    else:
        logger.info(f"Total candidate pairs after dedup: {total_before_cap}")
    return result
