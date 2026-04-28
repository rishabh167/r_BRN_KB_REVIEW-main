"""Unit tests for Phase 2 — pre-filter candidate pair generation."""

import pytest
import numpy as np
from app.analysis.pre_filter import (
    find_similar_pairs,
    find_shared_entity_pairs,
    find_relationship_conflict_pairs,
    build_candidate_pairs,
    _chunk_key,
    _canonicalize_source,
    deduplicate_url_chunks,
)


class TestChunkKey:
    def test_key_format(self):
        chunk = {"source": "Doc.pdf", "page": 3, "split_id": 1}
        assert _chunk_key(chunk) == "Doc.pdf|3|1"


class TestFindSimilarPairs:
    def test_finds_similar_cross_document_pairs(self):
        # Two chunks from different docs with identical embeddings
        chunks = [
            {"text": "hello", "source": "A.pdf", "page": 1, "split_id": 0,
             "embedding": [1.0, 0.0, 0.0]},
            {"text": "world", "source": "B.pdf", "page": 1, "split_id": 0,
             "embedding": [1.0, 0.0, 0.0]},
        ]
        pairs = find_similar_pairs(chunks, threshold=0.9)
        assert len(pairs) == 1
        assert pairs[0][2] == pytest.approx(1.0)

    def test_skips_same_document_pairs(self):
        chunks = [
            {"text": "hello", "source": "A.pdf", "page": 1, "split_id": 0,
             "embedding": [1.0, 0.0, 0.0]},
            {"text": "world", "source": "A.pdf", "page": 2, "split_id": 0,
             "embedding": [1.0, 0.0, 0.0]},
        ]
        pairs = find_similar_pairs(chunks, threshold=0.9)
        assert len(pairs) == 0

    def test_filters_below_threshold(self):
        chunks = [
            {"text": "hello", "source": "A.pdf", "page": 1, "split_id": 0,
             "embedding": [1.0, 0.0, 0.0]},
            {"text": "world", "source": "B.pdf", "page": 1, "split_id": 0,
             "embedding": [0.0, 1.0, 0.0]},  # orthogonal = 0 similarity
        ]
        pairs = find_similar_pairs(chunks, threshold=0.5)
        assert len(pairs) == 0

    def test_skips_chunks_without_embeddings(self):
        chunks = [
            {"text": "hello", "source": "A.pdf", "page": 1, "split_id": 0, "embedding": None},
            {"text": "world", "source": "B.pdf", "page": 1, "split_id": 0, "embedding": None},
        ]
        pairs = find_similar_pairs(chunks, threshold=0.5)
        assert len(pairs) == 0

    def test_respects_max_pairs(self):
        # Create many similar chunks
        chunks = []
        for i in range(10):
            chunks.append({
                "text": f"chunk {i}",
                "source": f"Doc{i}.pdf",
                "page": 1,
                "split_id": 0,
                "embedding": [1.0, 0.0, 0.0],
            })
        pairs = find_similar_pairs(chunks, threshold=0.9, max_pairs=5)
        assert len(pairs) <= 5

    def test_empty_input(self):
        assert find_similar_pairs([], threshold=0.9) == []

    def test_single_chunk(self):
        chunks = [{"text": "a", "source": "A.pdf", "page": 1, "split_id": 0, "embedding": [1.0]}]
        assert find_similar_pairs(chunks, threshold=0.5) == []


class TestFindSharedEntityPairs:
    def test_finds_pairs_sharing_entities(self, sample_chunks, sample_entity_mappings):
        pairs = find_shared_entity_pairs(sample_chunks, sample_entity_mappings, min_shared_entities=2)
        assert len(pairs) >= 1
        # Should find Pricing doc + FAQ sharing "enterprise plan" and "pricing"
        found = False
        for chunk_a, chunk_b, shared in pairs:
            sources = {chunk_a["source"], chunk_b["source"]}
            if "Product_Pricing_2024.pdf" in sources and "Company_FAQ.docx" in sources:
                assert "enterprise plan" in shared
                assert "pricing" in shared
                found = True
        assert found

    def test_skips_same_document(self):
        chunks = [
            {"text": "a", "source": "Same.pdf", "page": 1, "split_id": 0},
            {"text": "b", "source": "Same.pdf", "page": 2, "split_id": 0},
        ]
        mappings = [
            {"entity_id": "X", "source": "Same.pdf", "page": 1, "split_id": 0},
            {"entity_id": "X", "source": "Same.pdf", "page": 2, "split_id": 0},
            {"entity_id": "Y", "source": "Same.pdf", "page": 1, "split_id": 0},
            {"entity_id": "Y", "source": "Same.pdf", "page": 2, "split_id": 0},
        ]
        pairs = find_shared_entity_pairs(chunks, mappings, min_shared_entities=2)
        assert len(pairs) == 0

    def test_requires_minimum_shared(self):
        chunks = [
            {"text": "a", "source": "A.pdf", "page": 1, "split_id": 0},
            {"text": "b", "source": "B.pdf", "page": 1, "split_id": 0},
        ]
        mappings = [
            {"entity_id": "X", "source": "A.pdf", "page": 1, "split_id": 0},
            {"entity_id": "X", "source": "B.pdf", "page": 1, "split_id": 0},
        ]
        # Only 1 shared entity, min is 2
        pairs = find_shared_entity_pairs(chunks, mappings, min_shared_entities=2)
        assert len(pairs) == 0

    def test_empty_input(self):
        assert find_shared_entity_pairs([], [], min_shared_entities=2) == []


class TestFindRelationshipConflictPairs:
    def test_finds_conflicting_relationships(self):
        chunks = [
            {"text": "CEO is John", "source": "A.pdf", "page": 1, "split_id": 0},
            {"text": "CEO is Jane", "source": "B.pdf", "page": 1, "split_id": 0},
        ]
        mappings = [
            {"entity_id": "company", "source": "A.pdf", "page": 1, "split_id": 0},
            {"entity_id": "company", "source": "B.pdf", "page": 1, "split_id": 0},
            {"entity_id": "john", "source": "A.pdf", "page": 1, "split_id": 0},
            {"entity_id": "john", "source": "B.pdf", "page": 1, "split_id": 0},
        ]
        relationships = [
            {"source_entity": "company", "target_entity": "john", "relationship_type": "HAS_CEO"},
            {"source_entity": "company", "target_entity": "john", "relationship_type": "HAS_FORMER_CEO"},
        ]
        pairs = find_relationship_conflict_pairs(chunks, mappings, relationships)
        assert len(pairs) >= 1

    def test_no_conflicts_with_single_rel_type(self):
        chunks = [
            {"text": "a", "source": "A.pdf", "page": 1, "split_id": 0},
        ]
        relationships = [
            {"source_entity": "X", "target_entity": "Y", "relationship_type": "SAME"},
        ]
        pairs = find_relationship_conflict_pairs(chunks, [], relationships)
        assert len(pairs) == 0


class TestBuildCandidatePairs:
    def test_deduplicates_across_strategies(self, sample_chunks, sample_entity_mappings, sample_entity_relationships):
        candidates = build_candidate_pairs(
            sample_chunks, sample_entity_mappings, sample_entity_relationships,
            similarity_threshold=0.5,  # low threshold to get more pairs
            max_pairs=100,
        )
        # Should have deduplicated pairs
        pair_keys = set()
        for c in candidates:
            key = tuple(sorted([
                f"{c['chunk_a']['source']}|{c['chunk_a']['page']}|{c['chunk_a']['split_id']}",
                f"{c['chunk_b']['source']}|{c['chunk_b']['page']}|{c['chunk_b']['split_id']}",
            ]))
            assert key not in pair_keys, f"Duplicate pair: {key}"
            pair_keys.add(key)

    def test_respects_max_pairs(self, sample_chunks, sample_entity_mappings, sample_entity_relationships):
        candidates = build_candidate_pairs(
            sample_chunks, sample_entity_mappings, sample_entity_relationships,
            similarity_threshold=0.0,  # everything matches
            max_pairs=2,
        )
        assert len(candidates) <= 2

    def test_empty_input(self):
        candidates = build_candidate_pairs([], [], [])
        assert candidates == []

    def test_candidate_has_required_keys(self, sample_chunks, sample_entity_mappings, sample_entity_relationships):
        candidates = build_candidate_pairs(
            sample_chunks, sample_entity_mappings, sample_entity_relationships,
            similarity_threshold=0.5,
        )
        for c in candidates:
            assert "chunk_a" in c
            assert "chunk_b" in c
            assert "reasons" in c
            assert "entities" in c
            assert "score" in c


class TestCanonicalizeSource:
    def test_strips_http_scheme(self):
        assert _canonicalize_source("http://broadnet.ai/solutions") == "broadnet.ai/solutions"

    def test_strips_https_scheme(self):
        assert _canonicalize_source("https://broadnet.ai/solutions") == "broadnet.ai/solutions"

    def test_strips_www_prefix(self):
        assert _canonicalize_source("https://www.broadnet.ai/solutions") == "broadnet.ai/solutions"

    def test_http_and_https_match(self):
        assert (_canonicalize_source("http://broadnet.ai/solutions")
                == _canonicalize_source("https://broadnet.ai/solutions"))

    def test_www_and_non_www_match(self):
        assert (_canonicalize_source("https://broadnet.ai/solutions")
                == _canonicalize_source("https://www.broadnet.ai/solutions"))

    def test_strips_trailing_slash(self):
        assert (_canonicalize_source("https://broadnet.ai/solutions/")
                == _canonicalize_source("https://broadnet.ai/solutions"))

    def test_preserves_path(self):
        result = _canonicalize_source("https://broadnet.ai/solutions/healthcare")
        assert result == "broadnet.ai/solutions/healthcare"

    def test_non_url_passthrough(self):
        assert _canonicalize_source("report.pdf") == "report.pdf"
        assert _canonicalize_source("Company_FAQ.docx") == "Company_FAQ.docx"

    def test_www_without_scheme(self):
        assert _canonicalize_source("www.broadnet.ai/solutions") == "broadnet.ai/solutions"

    def test_empty_string(self):
        assert _canonicalize_source("") == ""

    def test_none(self):
        assert _canonicalize_source(None) == ""

    def test_case_insensitive_host(self):
        assert (_canonicalize_source("https://Broadnet.AI/Solutions")
                == _canonicalize_source("https://broadnet.ai/Solutions"))


class TestDeduplicateUrlChunks:
    def test_groups_url_variants(self):
        chunks = [
            {"text": "Content A", "source": "https://broadnet.ai/solutions", "page": 1, "split_id": 0,
             "embedding": [0.1]},
            {"text": "Content A", "source": "https://www.broadnet.ai/solutions", "page": 1, "split_id": 0,
             "embedding": [0.1]},
        ]
        deduped, url_groups = deduplicate_url_chunks(chunks)
        assert len(deduped) == 1
        assert "broadnet.ai/solutions" in url_groups
        assert len(url_groups["broadnet.ai/solutions"]) == 2

    def test_keeps_best_chunk(self):
        """Should keep the variant with the most total text."""
        chunks = [
            {"text": "Short", "source": "https://broadnet.ai/page", "page": 1, "split_id": 0},
            {"text": "This is a much longer piece of text", "source": "http://www.broadnet.ai/page",
             "page": 1, "split_id": 0},
        ]
        deduped, url_groups = deduplicate_url_chunks(chunks)
        assert len(deduped) == 1
        assert deduped[0]["source"] == "http://www.broadnet.ai/page"

    def test_preserves_non_url_chunks(self):
        chunks = [
            {"text": "PDF content", "source": "report.pdf", "page": 1, "split_id": 0},
            {"text": "Doc content", "source": "guide.docx", "page": 1, "split_id": 0},
        ]
        deduped, url_groups = deduplicate_url_chunks(chunks)
        assert len(deduped) == 2
        assert url_groups == {}

    def test_mixed_urls_and_files(self):
        chunks = [
            {"text": "PDF content", "source": "report.pdf", "page": 1, "split_id": 0},
            {"text": "Web content", "source": "https://broadnet.ai/about", "page": 1, "split_id": 0},
            {"text": "Web content", "source": "https://www.broadnet.ai/about", "page": 1, "split_id": 0},
        ]
        deduped, url_groups = deduplicate_url_chunks(chunks)
        assert len(deduped) == 2
        sources = {c["source"] for c in deduped}
        assert "report.pdf" in sources

    def test_three_url_variants(self):
        chunks = [
            {"text": "Content", "source": "http://broadnet.ai/x", "page": 1, "split_id": 0},
            {"text": "Content", "source": "https://broadnet.ai/x", "page": 1, "split_id": 0},
            {"text": "Content", "source": "https://www.broadnet.ai/x", "page": 1, "split_id": 0},
        ]
        deduped, url_groups = deduplicate_url_chunks(chunks)
        assert len(deduped) == 1
        assert len(url_groups["broadnet.ai/x"]) == 3

    def test_empty_input(self):
        deduped, url_groups = deduplicate_url_chunks([])
        assert deduped == []
        assert url_groups == {}

    def test_multiple_chunks_per_source(self):
        """When a URL has multiple chunks (pages), all kept chunks should be from the same variant."""
        chunks = [
            {"text": "Page 1", "source": "https://broadnet.ai/docs", "page": 1, "split_id": 0},
            {"text": "Page 2", "source": "https://broadnet.ai/docs", "page": 2, "split_id": 0},
            {"text": "Page 1", "source": "https://www.broadnet.ai/docs", "page": 1, "split_id": 0},
            {"text": "Page 2", "source": "https://www.broadnet.ai/docs", "page": 2, "split_id": 0},
        ]
        deduped, url_groups = deduplicate_url_chunks(chunks)
        assert len(deduped) == 2
        # All kept chunks should have the same source
        assert len({c["source"] for c in deduped}) == 1


class TestBuildCandidatePairsWithDedup:
    def test_fewer_pairs_after_url_dedup(self):
        """URL dedup should happen before build_candidate_pairs is called,
        so passing pre-deduped chunks should produce fewer pairs."""
        # Create chunks with URL variants — identical embeddings
        emb = [1.0, 0.0, 0.0]
        all_chunks = [
            {"text": "Content", "source": "https://broadnet.ai/solutions", "page": 1, "split_id": 0,
             "embedding": emb},
            {"text": "Content", "source": "https://www.broadnet.ai/solutions", "page": 1, "split_id": 0,
             "embedding": emb},
            {"text": "Other", "source": "report.pdf", "page": 1, "split_id": 0,
             "embedding": emb},
        ]

        # Without dedup: 3 chunks, 3 cross-doc pairs possible
        pairs_all = build_candidate_pairs(all_chunks, [], [], similarity_threshold=0.9)

        # With dedup: 2 chunks, 1 cross-doc pair
        deduped, _ = deduplicate_url_chunks(all_chunks)
        pairs_deduped = build_candidate_pairs(deduped, [], [], similarity_threshold=0.9)

        assert len(pairs_deduped) < len(pairs_all)


