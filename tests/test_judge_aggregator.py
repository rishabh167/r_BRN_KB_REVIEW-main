"""Unit tests for Phase 4 — N-judge majority voting aggregation."""

from app.analysis.judge_aggregator import aggregate_judge_findings, finding_key


def _make_finding(pair_index=0, issue_type="CONTRADICTION", severity="HIGH",
                  confidence=0.9, judge_index=0, title="Test issue",
                  source_a="A.pdf", page_a=1, source_b="B.pdf", page_b=1):
    return {
        "pair_index": pair_index,
        "judge_index": judge_index,
        "judge_provider": "litellm",
        "judge_model": "test-model",
        "issue_type": issue_type,
        "severity": severity,
        "confidence": confidence,
        "title": title,
        "description": "Test description",
        "reasoning": "Test reasoning",
        "claim_a": "Claim A",
        "claim_b": "Claim B",
        "chunk_a": {"source": source_a, "page": page_a, "split_id": 0, "text": "Text A"},
        "chunk_b": {"source": source_b, "page": page_b, "split_id": 0, "text": "Text B"},
        "entities": ["entity1"],
        "input_tokens": 100,
        "output_tokens": 50,
    }


class TestFindingKey:
    def test_pair_based_key_uses_chunk_identifiers(self):
        f = {
            "pair_index": 3,
            "issue_type": "CONTRADICTION",
            "chunk_a": {"source": "A.pdf", "page": 1, "split_id": 0},
            "chunk_b": {"source": "B.pdf", "page": 2, "split_id": 1},
        }
        key = finding_key(f)
        assert "A.pdf|1|0" in key
        assert "B.pdf|2|1" in key
        assert key.endswith("|CONTRADICTION")

    def test_pair_key_is_order_independent(self):
        f1 = {
            "issue_type": "CONTRADICTION",
            "chunk_a": {"source": "A.pdf", "page": 1, "split_id": 0},
            "chunk_b": {"source": "B.pdf", "page": 2, "split_id": 1},
        }
        f2 = {
            "issue_type": "CONTRADICTION",
            "chunk_a": {"source": "B.pdf", "page": 2, "split_id": 1},
            "chunk_b": {"source": "A.pdf", "page": 1, "split_id": 0},
        }
        assert finding_key(f1) == finding_key(f2)

    def test_ambiguity_key(self):
        f = {
            "pair_index": None,
            "issue_type": "AMBIGUITY",
            "title": "Vague language",
            "chunk_a": {"source": "Doc.pdf", "page": 2, "split_id": 0},
            "chunk_b": None,
        }
        key = finding_key(f)
        assert "Doc.pdf" in key
        assert "AMBIGUITY" in key

    def test_url_variants_produce_same_pair_key(self):
        """broadnet.ai/x and www.broadnet.ai/x should produce the same finding key."""
        f1 = {
            "issue_type": "CONTRADICTION",
            "chunk_a": {"source": "https://broadnet.ai/solutions", "page": 1, "split_id": 0},
            "chunk_b": {"source": "report.pdf", "page": 1, "split_id": 0},
        }
        f2 = {
            "issue_type": "CONTRADICTION",
            "chunk_a": {"source": "https://www.broadnet.ai/solutions", "page": 1, "split_id": 0},
            "chunk_b": {"source": "report.pdf", "page": 1, "split_id": 0},
        }
        assert finding_key(f1) == finding_key(f2)

    def test_url_variants_produce_same_ambiguity_key(self):
        f1 = {
            "issue_type": "AMBIGUITY",
            "title": "Vague pricing info",
            "chunk_a": {"source": "https://broadnet.ai/pricing", "page": 1, "split_id": 0},
            "chunk_b": None,
        }
        f2 = {
            "issue_type": "AMBIGUITY",
            "title": "Vague pricing info",
            "chunk_a": {"source": "http://www.broadnet.ai/pricing", "page": 1, "split_id": 0},
            "chunk_b": None,
        }
        assert finding_key(f1) == finding_key(f2)


class TestAggregateUnanimous:
    def test_all_judges_agree(self):
        """3/3 judges flag the same issue -> UNANIMOUS."""
        findings = [
            [_make_finding(judge_index=0, confidence=0.95)],
            [_make_finding(judge_index=1, confidence=0.90)],
            [_make_finding(judge_index=2, confidence=0.85)],
        ]
        result = aggregate_judge_findings(findings, num_judges=3)
        assert len(result) == 1
        issue = result[0]
        assert issue["consensus"] == "UNANIMOUS"
        assert issue["judges_flagged"] == 3
        assert issue["judges_total"] == 3
        # UNANIMOUS confidence = max of individual confidences
        assert issue["confidence"] == 0.95

    def test_unanimous_takes_worst_severity(self):
        findings = [
            [_make_finding(judge_index=0, severity="MEDIUM")],
            [_make_finding(judge_index=1, severity="CRITICAL")],
            [_make_finding(judge_index=2, severity="HIGH")],
        ]
        result = aggregate_judge_findings(findings, num_judges=3)
        assert result[0]["severity"] == "CRITICAL"


class TestAggregateMajority:
    def test_majority_consensus(self):
        """2/3 judges flag -> MAJORITY."""
        findings = [
            [_make_finding(judge_index=0, confidence=0.8)],
            [_make_finding(judge_index=1, confidence=0.7)],
            [],  # judge 2 found nothing
        ]
        result = aggregate_judge_findings(findings, num_judges=3)
        assert len(result) == 1
        issue = result[0]
        assert issue["consensus"] == "MAJORITY"
        assert issue["judges_flagged"] == 2
        # MAJORITY confidence = avg * 0.9
        expected = (0.8 + 0.7) / 2 * 0.9
        assert abs(issue["confidence"] - expected) < 0.01


class TestAggregateMinority:
    def test_minority_consensus(self):
        """1/3 judges flag -> MINORITY."""
        findings = [
            [_make_finding(judge_index=0, confidence=0.6)],
            [],
            [],
        ]
        result = aggregate_judge_findings(findings, num_judges=3)
        assert len(result) == 1
        issue = result[0]
        assert issue["consensus"] == "MINORITY"
        assert issue["judges_flagged"] == 1
        # MINORITY confidence = avg * 0.6
        expected = 0.6 * 0.6
        assert abs(issue["confidence"] - expected) < 0.01


class TestAggregateSingleJudge:
    def test_single_judge(self):
        """1 judge configured -> SINGLE_JUDGE."""
        findings = [
            [_make_finding(judge_index=0, confidence=0.85)],
        ]
        result = aggregate_judge_findings(findings, num_judges=1)
        assert len(result) == 1
        assert result[0]["consensus"] == "SINGLE_JUDGE"
        assert result[0]["confidence"] == 0.85


class TestAggregateMultipleIssues:
    def test_different_pairs_kept_separate(self):
        """Issues from different document pairs should not be merged."""
        findings = [
            [
                _make_finding(pair_index=0, issue_type="CONTRADICTION", judge_index=0,
                              source_a="A.pdf", source_b="B.pdf"),
                _make_finding(pair_index=1, issue_type="CONTRADICTION", judge_index=0,
                              source_a="C.pdf", source_b="D.pdf"),
            ],
        ]
        result = aggregate_judge_findings(findings, num_judges=1)
        assert len(result) == 2

    def test_same_pair_different_types_kept_separate(self):
        """Same pair, different issue types -> separate issues."""
        findings = [
            [
                _make_finding(pair_index=0, issue_type="CONTRADICTION"),
                _make_finding(pair_index=0, issue_type="SEMANTIC_DUPLICATION"),
            ],
        ]
        result = aggregate_judge_findings(findings, num_judges=1)
        assert len(result) == 2


class TestAggregateSorting:
    def test_unanimous_before_majority(self):
        # Use different source_b so the two pairs get distinct finding keys
        findings_judge_0 = [
            _make_finding(pair_index=0, judge_index=0, source_b="B.pdf"),  # will be UNANIMOUS
            _make_finding(pair_index=1, judge_index=0, source_b="C.pdf"),  # will be MINORITY
        ]
        findings_judge_1 = [
            _make_finding(pair_index=0, judge_index=1, source_b="B.pdf"),  # same pair
            # judge 1 doesn't flag the C.pdf pair
        ]
        findings_judge_2 = [
            _make_finding(pair_index=0, judge_index=2, source_b="B.pdf"),  # same pair
            # judge 2 doesn't flag the C.pdf pair
        ]
        result = aggregate_judge_findings(
            [findings_judge_0, findings_judge_1, findings_judge_2], num_judges=3
        )
        assert len(result) == 2
        assert result[0]["consensus"] == "UNANIMOUS"
        assert result[1]["consensus"] == "MINORITY"


class TestAggregateEdgeCases:
    def test_empty_findings(self):
        result = aggregate_judge_findings([[], [], []], num_judges=3)
        assert result == []

    def test_no_judges(self):
        result = aggregate_judge_findings([], num_judges=0)
        assert result == []

    def test_preserves_judge_details(self):
        findings = [[_make_finding(judge_index=0)]]
        result = aggregate_judge_findings(findings, num_judges=1)
        assert len(result[0]["judge_details"]) == 1
        assert result[0]["judge_details"][0]["reasoning"] == "Test reasoning"

    def test_preserves_document_attribution(self):
        findings = [[_make_finding(source_a="Pricing.pdf", page_a=3,
                                   source_b="FAQ.docx", page_b=12)]]
        result = aggregate_judge_findings(findings, num_judges=1)
        issue = result[0]
        assert issue["doc_a_name"] == "Pricing.pdf"
        assert issue["doc_a_page"] == 3
        assert issue["doc_b_name"] == "FAQ.docx"
        assert issue["doc_b_page"] == 12
