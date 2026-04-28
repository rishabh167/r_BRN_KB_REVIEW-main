"""Unit tests for Phase 3 — LLM analyzer JSON parsing and batch construction."""

from app.analysis.analyzers import (
    _parse_json_response,
    _looks_truncated,
    _salvage_truncated_json,
    _process_pair_findings,
)


class TestParseJsonResponse:
    def test_plain_json_array(self):
        text = '[{"detected": true, "issue_type": "CONTRADICTION"}]'
        result = _parse_json_response(text)
        assert len(result) == 1
        assert result[0]["issue_type"] == "CONTRADICTION"

    def test_empty_array(self):
        assert _parse_json_response("[]") == []

    def test_markdown_fenced_json(self):
        text = '```json\n[{"detected": true}]\n```'
        result = _parse_json_response(text)
        assert len(result) == 1

    def test_markdown_fenced_no_lang(self):
        text = '```\n[{"detected": true}]\n```'
        result = _parse_json_response(text)
        assert len(result) == 1

    def test_json_with_surrounding_text(self):
        text = 'Here are the findings:\n[{"detected": true}]\nEnd of analysis.'
        result = _parse_json_response(text)
        assert len(result) == 1

    def test_single_dict_wrapped_in_list(self):
        text = '{"detected": true, "issue_type": "AMBIGUITY"}'
        result = _parse_json_response(text)
        assert len(result) == 1

    def test_garbage_input(self):
        result = _parse_json_response("this is not json at all")
        assert result == []

    def test_whitespace(self):
        text = '  \n  [{"detected": true}]  \n  '
        result = _parse_json_response(text)
        assert len(result) == 1

    def test_multiple_findings(self):
        text = '[{"detected": true, "pair_index": 0}, {"detected": true, "pair_index": 1}]'
        result = _parse_json_response(text)
        assert len(result) == 2
        assert result[0]["pair_index"] == 0
        assert result[1]["pair_index"] == 1

    def test_nested_json_in_description(self):
        text = '[{"detected": true, "description": "Price is \\"$99\\""}]'
        result = _parse_json_response(text)
        assert len(result) == 1

    def test_wrapper_dict_unwrapped(self):
        """Model returns {"findings": [...]} instead of bare [...]."""
        text = '{"findings": [{"detected": true, "pair_index": 0}, {"detected": true, "pair_index": 1}]}'
        result = _parse_json_response(text)
        assert len(result) == 2
        assert result[0]["pair_index"] == 0
        assert result[1]["pair_index"] == 1

    def test_wrapper_dict_with_issues_key(self):
        """Model returns {"issues": [...]}."""
        text = '{"issues": [{"detected": true}]}'
        result = _parse_json_response(text)
        assert len(result) == 1
        assert result[0]["detected"] is True

    def test_multi_key_dict_not_unwrapped(self):
        """Dict with multiple keys is treated as a single finding, not unwrapped."""
        text = '{"detected": true, "issue_type": "CONTRADICTION"}'
        result = _parse_json_response(text)
        assert len(result) == 1
        assert result[0]["detected"] is True

    def test_wrapper_dict_empty_array(self):
        """Model returns {"findings": []}."""
        text = '{"findings": []}'
        result = _parse_json_response(text)
        assert result == []


class TestLooksTruncated:
    def test_truncated_array(self):
        text = '[{"detected": true, "pair_index": 0}, {"detected": tr'
        assert _looks_truncated(text) is True

    def test_complete_array(self):
        text = '[{"detected": true}]'
        assert _looks_truncated(text) is False

    def test_no_findings(self):
        text = '[{"something": "else"'
        assert _looks_truncated(text) is False

    def test_not_array(self):
        text = '{"detected": true}'
        assert _looks_truncated(text) is False

    def test_empty(self):
        assert _looks_truncated("") is False


class TestSalvageTruncatedJson:
    def test_one_complete_one_truncated(self):
        text = '[{"detected": true, "pair_index": 0}, {"detected": tr'
        result = _salvage_truncated_json(text)
        assert len(result) == 1
        assert result[0]["pair_index"] == 0

    def test_two_complete_one_truncated(self):
        text = '[{"a": 1}, {"b": 2}, {"c":'
        result = _salvage_truncated_json(text)
        assert len(result) == 2

    def test_no_opening_bracket(self):
        assert _salvage_truncated_json("just text") == []

    def test_complete_array(self):
        text = '[{"a": 1}, {"b": 2}]'
        result = _salvage_truncated_json(text)
        assert len(result) == 2

    def test_escaped_braces_in_strings(self):
        text = '[{"desc": "price is \\"$99\\"", "x": 1}, {"broken'
        result = _salvage_truncated_json(text)
        assert len(result) == 1
        assert result[0]["x"] == 1


class TestProcessPairFindings:
    """Test _process_pair_findings with edge cases."""

    def _make_pairs(self, n):
        return [
            {
                "chunk_a": {"source": f"doc_a_{i}.pdf", "page": 1, "text": "a"},
                "chunk_b": {"source": f"doc_b_{i}.pdf", "page": 2, "text": "b"},
                "entities": [],
            }
            for i in range(n)
        ]

    def test_invalid_pair_index_skipped(self):
        pairs = self._make_pairs(2)
        raw = [
            {"detected": True, "pair_index": 0, "confidence": 0.9},
            {"detected": True, "pair_index": 999, "confidence": 0.9},  # out of range
            {"detected": True, "pair_index": -1, "confidence": 0.9},  # negative
        ]
        results = _process_pair_findings(raw, pairs, 0, "litellm", "test", 100, 50)
        assert len(results) == 1
        assert results[0]["pair_index"] == 0

    def test_low_confidence_filtered(self):
        pairs = self._make_pairs(1)
        raw = [{"detected": True, "pair_index": 0, "confidence": 0.3}]
        results = _process_pair_findings(raw, pairs, 0, "litellm", "test", 100, 50)
        assert len(results) == 0

    def test_not_detected_filtered(self):
        pairs = self._make_pairs(1)
        raw = [{"detected": False, "pair_index": 0, "confidence": 0.9}]
        results = _process_pair_findings(raw, pairs, 0, "litellm", "test", 100, 50)
        assert len(results) == 0
