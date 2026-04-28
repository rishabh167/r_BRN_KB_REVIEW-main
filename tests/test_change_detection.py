"""Unit tests for change detection helpers in review_runner.

These are pure-function tests — no DB or LLM needed.
"""

import pytest

from app.analysis.review_runner import (
    _compute_document_hashes,
    _classify_documents,
    _configs_compatible_for_carryforward,
    _split_candidate_pairs,
)


# ── _compute_document_hashes ─────────────────────────────────────────────


class TestComputeDocumentHashes:
    def test_deterministic(self):
        """Same input produces the same hash."""
        chunks = [
            {"text": "Hello world", "source": "doc.pdf", "page": 1, "split_id": 0},
            {"text": "Second chunk", "source": "doc.pdf", "page": 2, "split_id": 0},
        ]
        h1 = _compute_document_hashes(chunks)
        h2 = _compute_document_hashes(chunks)
        assert h1 == h2

    def test_different_content_different_hash(self):
        """Different text produces different hashes."""
        chunks_a = [{"text": "Version A", "source": "doc.pdf", "page": 1, "split_id": 0}]
        chunks_b = [{"text": "Version B", "source": "doc.pdf", "page": 1, "split_id": 0}]
        h_a = _compute_document_hashes(chunks_a)
        h_b = _compute_document_hashes(chunks_b)
        assert h_a["doc.pdf"] != h_b["doc.pdf"]

    def test_url_canonicalization(self):
        """URL variants map to the same canonical source."""
        chunks = [
            {"text": "Content", "source": "https://www.example.com/page", "page": 1, "split_id": 0},
            {"text": "More", "source": "http://example.com/page", "page": 2, "split_id": 0},
        ]
        hashes = _compute_document_hashes(chunks)
        # Both URLs should canonicalize to the same key
        assert len(hashes) == 1
        assert "example.com/page" in hashes

    def test_multiple_sources(self):
        """Chunks from different sources produce separate hashes."""
        chunks = [
            {"text": "A content", "source": "A.pdf", "page": 1, "split_id": 0},
            {"text": "B content", "source": "B.pdf", "page": 1, "split_id": 0},
        ]
        hashes = _compute_document_hashes(chunks)
        assert len(hashes) == 2
        assert "A.pdf" in hashes
        assert "B.pdf" in hashes

    def test_sort_order_matters(self):
        """Chunks are sorted by (page, split_id) so order of input doesn't matter."""
        chunks_ordered = [
            {"text": "First", "source": "doc.pdf", "page": 1, "split_id": 0},
            {"text": "Second", "source": "doc.pdf", "page": 2, "split_id": 0},
        ]
        chunks_reversed = [
            {"text": "Second", "source": "doc.pdf", "page": 2, "split_id": 0},
            {"text": "First", "source": "doc.pdf", "page": 1, "split_id": 0},
        ]
        assert _compute_document_hashes(chunks_ordered) == _compute_document_hashes(chunks_reversed)

    def test_hash_format(self):
        """Hashes are 64-char lowercase hex strings (SHA-256)."""
        chunks = [{"text": "test", "source": "doc.pdf", "page": 1, "split_id": 0}]
        hashes = _compute_document_hashes(chunks)
        h = hashes["doc.pdf"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_source_content_changed(self):
        """Same source name but different content produces different hash."""
        chunks_v1 = [{"text": "Price is $99", "source": "Pricing.pdf", "page": 1, "split_id": 0}]
        chunks_v2 = [{"text": "Price is $149", "source": "Pricing.pdf", "page": 1, "split_id": 0}]
        h1 = _compute_document_hashes(chunks_v1)
        h2 = _compute_document_hashes(chunks_v2)
        assert h1["Pricing.pdf"] != h2["Pricing.pdf"]


# ── _classify_documents ──────────────────────────────────────────────────


class TestClassifyDocuments:
    def test_all_unchanged(self):
        """All docs with same hashes → all unchanged."""
        current = {"a.pdf": "abc123", "b.pdf": "def456"}
        previous = {"a.pdf": "abc123", "b.pdf": "def456"}
        unchanged, changed, new = _classify_documents(current, previous)
        assert unchanged == {"a.pdf", "b.pdf"}
        assert changed == set()
        assert new == set()

    def test_all_new(self):
        """No previous hashes → all new."""
        current = {"a.pdf": "abc123", "b.pdf": "def456"}
        previous = {}
        unchanged, changed, new = _classify_documents(current, previous)
        assert unchanged == set()
        assert changed == set()
        assert new == {"a.pdf", "b.pdf"}

    def test_mixed(self):
        """Mix of unchanged, changed, and new."""
        current = {"a.pdf": "abc123", "b.pdf": "changed", "c.pdf": "new_doc"}
        previous = {"a.pdf": "abc123", "b.pdf": "old_hash"}
        unchanged, changed, new = _classify_documents(current, previous)
        assert unchanged == {"a.pdf"}
        assert changed == {"b.pdf"}
        assert new == {"c.pdf"}

    def test_removed_not_in_current(self):
        """Removed docs are not in current_hashes, so they don't appear in any category."""
        current = {"a.pdf": "abc123"}
        previous = {"a.pdf": "abc123", "removed.pdf": "old"}
        unchanged, changed, new = _classify_documents(current, previous)
        assert unchanged == {"a.pdf"}
        assert changed == set()
        assert new == set()
        # removed = previous.keys() - current.keys() computed inline where needed


# ── _configs_compatible_for_carryforward ─────────────────────────────────


class TestConfigsCompatible:
    def test_same_config(self):
        config = {
            "analysis_types": ["CONTRADICTION", "ENTITY_INCONSISTENCY"],
            "similarity_threshold": 0.85,
            "max_candidate_pairs": 50,
        }
        assert _configs_compatible_for_carryforward(config, config) is True

    def test_different_analysis_types(self):
        a = {"analysis_types": ["CONTRADICTION"], "similarity_threshold": 0.85, "max_candidate_pairs": 50}
        b = {"analysis_types": ["CONTRADICTION", "AMBIGUITY"], "similarity_threshold": 0.85, "max_candidate_pairs": 50}
        assert _configs_compatible_for_carryforward(a, b) is False

    def test_analysis_types_order_independent(self):
        a = {"analysis_types": ["ENTITY_INCONSISTENCY", "CONTRADICTION"], "similarity_threshold": 0.85, "max_candidate_pairs": 50}
        b = {"analysis_types": ["CONTRADICTION", "ENTITY_INCONSISTENCY"], "similarity_threshold": 0.85, "max_candidate_pairs": 50}
        assert _configs_compatible_for_carryforward(a, b) is True

    def test_different_threshold(self):
        a = {"analysis_types": ["CONTRADICTION"], "similarity_threshold": 0.85, "max_candidate_pairs": 50}
        b = {"analysis_types": ["CONTRADICTION"], "similarity_threshold": 0.90, "max_candidate_pairs": 50}
        assert _configs_compatible_for_carryforward(a, b) is False

    def test_different_max_pairs(self):
        a = {"analysis_types": ["CONTRADICTION"], "similarity_threshold": 0.85, "max_candidate_pairs": 50}
        b = {"analysis_types": ["CONTRADICTION"], "similarity_threshold": 0.85, "max_candidate_pairs": 100}
        assert _configs_compatible_for_carryforward(a, b) is False

    def test_different_judges_ok(self):
        """Judge selection doesn't matter for compatibility."""
        a = {"analysis_types": ["CONTRADICTION"], "similarity_threshold": 0.85, "max_candidate_pairs": 50,
             "judges": [{"provider": "litellm", "model": "gemini-flash"}]}
        b = {"analysis_types": ["CONTRADICTION"], "similarity_threshold": 0.85, "max_candidate_pairs": 50,
             "judges": [{"provider": "litellm", "model": "claude-haiku"}]}
        assert _configs_compatible_for_carryforward(a, b) is True


# ── _split_candidate_pairs ───────────────────────────────────────────────


class TestSplitCandidatePairs:
    def test_all_reusable(self):
        """Both sources unchanged → pair is reusable."""
        candidates = [
            {"chunk_a": {"source": "A.pdf"}, "chunk_b": {"source": "B.pdf"}},
        ]
        reusable, new = _split_candidate_pairs(candidates, {"A.pdf", "B.pdf"})
        assert len(reusable) == 1
        assert len(new) == 0

    def test_one_changed(self):
        """One source changed → pair is new."""
        candidates = [
            {"chunk_a": {"source": "A.pdf"}, "chunk_b": {"source": "B.pdf"}},
        ]
        reusable, new = _split_candidate_pairs(candidates, {"A.pdf"})
        assert len(reusable) == 0
        assert len(new) == 1

    def test_both_changed(self):
        """Both sources changed → pair is new."""
        candidates = [
            {"chunk_a": {"source": "A.pdf"}, "chunk_b": {"source": "B.pdf"}},
        ]
        reusable, new = _split_candidate_pairs(candidates, set())
        assert len(reusable) == 0
        assert len(new) == 1

    def test_mixed(self):
        """Mix of reusable and new pairs."""
        candidates = [
            {"chunk_a": {"source": "A.pdf"}, "chunk_b": {"source": "B.pdf"}},
            {"chunk_a": {"source": "A.pdf"}, "chunk_b": {"source": "C.pdf"}},
            {"chunk_a": {"source": "C.pdf"}, "chunk_b": {"source": "D.pdf"}},
        ]
        unchanged = {"A.pdf", "B.pdf"}
        reusable, new = _split_candidate_pairs(candidates, unchanged)
        assert len(reusable) == 1  # A-B
        assert len(new) == 2  # A-C, C-D

    def test_url_canonicalization(self):
        """URL sources are canonicalized before comparison."""
        candidates = [
            {"chunk_a": {"source": "https://example.com/a"}, "chunk_b": {"source": "http://www.example.com/b"}},
        ]
        # Canonical forms: example.com/a, example.com/b
        reusable, new = _split_candidate_pairs(candidates, {"example.com/a", "example.com/b"})
        assert len(reusable) == 1
        assert len(new) == 0
