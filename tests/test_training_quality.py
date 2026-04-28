"""Unit tests for Phase 1.5 — training quality structural checks."""

from app.analysis.training_quality import (
    check_missing_embeddings,
    check_empty_content,
    check_url_duplicates,
    run_training_quality_checks,
)


class TestMissingEmbeddings:
    def test_detects_missing_embeddings(self, chunks_with_issues):
        issues = check_missing_embeddings(chunks_with_issues)
        # Bad_Doc.pdf and Null_Doc.pdf have None embeddings
        sources = {i["doc_a_name"] for i in issues}
        assert "Bad_Doc.pdf" in sources
        assert "Null_Doc.pdf" in sources

    def test_no_issues_when_all_embedded(self, sample_chunks):
        issues = check_missing_embeddings(sample_chunks)
        assert issues == []

    def test_groups_by_document(self):
        chunks = [
            {"text": "a", "source": "Doc.pdf", "page": 1, "split_id": 0, "embedding": None},
            {"text": "b", "source": "Doc.pdf", "page": 2, "split_id": 0, "embedding": None},
            {"text": "c", "source": "Other.pdf", "page": 1, "split_id": 0, "embedding": None},
        ]
        issues = check_missing_embeddings(chunks)
        assert len(issues) == 2  # one per document
        doc_pdf_issue = next(i for i in issues if i["doc_a_name"] == "Doc.pdf")
        assert "2 sections" in doc_pdf_issue["title"]

    def test_issue_fields(self):
        chunks = [
            {"text": "some text", "source": "Test.pdf", "page": 5, "split_id": 0, "embedding": None},
        ]
        issues = check_missing_embeddings(chunks)
        assert len(issues) == 1
        issue = issues[0]
        assert issue["issue_type"] == "MISSING_EMBEDDINGS"
        assert issue["severity"] == "HIGH"
        assert issue["confidence"] == 1.0
        assert issue["consensus"] == "STRUCTURAL"
        assert issue["doc_a_name"] == "Test.pdf"
        assert issue["doc_a_page"] == 5

    def test_empty_input(self):
        assert check_missing_embeddings([]) == []


class TestEmptyContent:
    def test_detects_empty_text(self, chunks_with_issues):
        issues = check_empty_content(chunks_with_issues)
        sources = {i["doc_a_name"] for i in issues}
        assert "Empty_Doc.pdf" in sources
        assert "Null_Doc.pdf" in sources

    def test_no_issues_when_all_have_content(self, sample_chunks):
        issues = check_empty_content(sample_chunks)
        assert issues == []

    def test_groups_by_document(self):
        chunks = [
            {"text": "", "source": "Empty.pdf", "page": 1, "split_id": 0},
            {"text": "   ", "source": "Empty.pdf", "page": 2, "split_id": 0},
        ]
        issues = check_empty_content(chunks)
        assert len(issues) == 1
        assert "2 sections" in issues[0]["title"]

    def test_issue_fields(self):
        chunks = [
            {"text": None, "source": "Broken.pdf", "page": 3, "split_id": 0},
        ]
        issues = check_empty_content(chunks)
        assert len(issues) == 1
        issue = issues[0]
        assert issue["issue_type"] == "EMPTY_CONTENT"
        assert issue["severity"] == "HIGH"
        assert issue["confidence"] == 1.0
        assert issue["consensus"] == "STRUCTURAL"
        assert issue["doc_a_excerpt"] is None  # no text to excerpt

    def test_empty_input(self):
        assert check_empty_content([]) == []


class TestCheckUrlDuplicates:
    def test_groups_url_variants_into_one_issue(self):
        url_groups = {
            "broadnet.ai/solutions": [
                "https://broadnet.ai/solutions",
                "https://www.broadnet.ai/solutions",
                "http://broadnet.ai/solutions",
            ],
        }
        issues = check_url_duplicates(url_groups)
        assert len(issues) == 1
        issue = issues[0]
        assert issue["issue_type"] == "URL_DUPLICATION"
        assert issue["severity"] == "MEDIUM"
        assert issue["confidence"] == 1.0
        assert issue["consensus"] == "STRUCTURAL"
        assert "3 URL variants" in issue["title"]
        assert "broadnet.ai/solutions" in issue["title"]
        # All variants should be listed in description
        assert "https://broadnet.ai/solutions" in issue["description"]
        assert "https://www.broadnet.ai/solutions" in issue["description"]
        assert "http://broadnet.ai/solutions" in issue["description"]

    def test_empty_url_groups_produces_no_issues(self):
        issues = check_url_duplicates({})
        assert issues == []

    def test_multiple_groups(self):
        url_groups = {
            "broadnet.ai/solutions": [
                "https://broadnet.ai/solutions",
                "https://www.broadnet.ai/solutions",
            ],
            "broadnet.ai/about": [
                "http://broadnet.ai/about",
                "https://www.broadnet.ai/about",
            ],
        }
        issues = check_url_duplicates(url_groups)
        assert len(issues) == 2

    def test_issue_has_doc_a_name(self):
        url_groups = {
            "example.com/page": ["https://example.com/page", "https://www.example.com/page"],
        }
        issues = check_url_duplicates(url_groups)
        assert issues[0]["doc_a_name"] == "https://example.com/page"


class TestRunTrainingQualityChecks:
    def test_combines_all_checks(self, chunks_with_issues):
        issues = run_training_quality_checks(chunks_with_issues)
        types = {i["issue_type"] for i in issues}
        assert "MISSING_EMBEDDINGS" in types
        assert "EMPTY_CONTENT" in types

    def test_no_issues_on_clean_data(self, sample_chunks):
        issues = run_training_quality_checks(sample_chunks)
        assert issues == []

    def test_includes_url_duplicates_when_provided(self, sample_chunks):
        url_groups = {
            "broadnet.ai/page": [
                "https://broadnet.ai/page",
                "https://www.broadnet.ai/page",
            ],
        }
        issues = run_training_quality_checks(sample_chunks, url_groups=url_groups)
        types = {i["issue_type"] for i in issues}
        assert "URL_DUPLICATION" in types

    def test_no_url_issues_when_groups_none(self, sample_chunks):
        issues = run_training_quality_checks(sample_chunks, url_groups=None)
        types = {i["issue_type"] for i in issues}
        assert "URL_DUPLICATION" not in types
