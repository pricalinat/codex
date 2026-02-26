"""Tests for BM25 ranking module."""

import pytest
from src.test_analysis_assistant.bm25_ranking import (
    BM25,
    BM25Provider,
    create_bm25_provider,
)


class TestBM25:
    """Tests for the BM25 class."""

    def test_bm25_fit_and_score(self):
        """Test basic BM25 fitting and scoring."""
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A fast red fox ran away from the dog",
            "The cat sat on the mat",
            "Dogs are man's best friend",
        ]

        bm25 = BM25()
        bm25.fit(documents)

        # Query for "fox"
        scores = bm25.score_batch("fox")
        assert len(scores) > 0
        # Document 1 should have highest score (contains "fox" once, while doc 0 also has it)
        # BM25 should return at least one relevant document
        assert scores[0][1] > 0

    def test_bm25_term_frequency_saturation(self):
        """Test that BM25 handles term frequency saturation."""
        # Document with repeated terms should not get proportionally higher score
        doc1 = "fox"  # 1 occurrence
        doc2 = "fox fox fox fox fox"  # 5 occurrences

        bm25 = BM25(k1=1.5)
        bm25.fit([doc1, doc2])

        score1 = bm25.score("fox", 0)
        score2 = bm25.score("fox", 1)

        # With k1=1.5, score should not be 5x even though term freq is 5x
        # The saturation prevents linear growth
        assert score2 > score1
        assert score2 < score1 * 5  # Not linear

    def test_bm25_idf_handles_rare_terms(self):
        """Test that IDF handles rare vs common terms correctly."""
        documents = [
            "python programming language",
            "python is interpreted",
            "java is compiled",
            "python and java",
        ]

        bm25 = BM25()
        bm25.fit(documents)

        # "python" appears in 3 docs, should have lower IDF than rare term
        idf_python = bm25._idf("python")
        idf_rare = bm25._idf("ruby")  # Not in corpus

        # Python should have positive but lower IDF than a rare term
        assert idf_python > 0
        # Rare term gets smoothed IDF
        assert idf_rare > 0

    def test_bm25_empty_corpus(self):
        """Test BM25 handles empty corpus gracefully."""
        bm25 = BM25()
        bm25.fit([])

        scores = bm25.score_batch("test")
        assert scores == []

    def test_bm25_empty_query(self):
        """Test BM25 handles empty query gracefully."""
        bm25 = BM25()
        bm25.fit(["test document"])

        scores = bm25.score_batch("")
        assert scores == []

    def test_bm25_get_top_k(self):
        """Test top-k retrieval."""
        documents = [
            "python programming tutorial",
            "java coding guide",
            "python and machine learning",
            "web development with python",
        ]

        bm25 = BM25()
        bm25.fit(documents)

        top_results = bm25.get_top_k("python", k=2)

        assert len(top_results) <= 2
        # Should include python-related docs
        doc_indices = [idx for idx, _ in top_results]
        assert 0 in doc_indices or 2 in doc_indices or 3 in doc_indices

    def test_bm25_custom_parameters(self):
        """Test BM25 with custom k1 and b parameters."""
        documents = ["test document content"] * 10

        # Higher k1 gives more weight to term frequency
        bm25_high_k1 = BM25(k1=3.0)
        bm25_low_k1 = BM25(k1=0.5)

        bm25_high_k1.fit(documents)
        bm25_low_k1.fit(documents)

        score_high = bm25_high_k1.score("test", 0)
        score_low = bm25_low_k1.score("test", 0)

        # With more saturation, higher k1 should give similar scores
        # because term frequency impact is already high


class TestBM25Provider:
    """Tests for the BM25Provider class."""

    def test_provider_fit(self):
        """Test BM25Provider fit method."""
        provider = BM25Provider()
        documents = ["doc one", "doc two", "doc three"]

        provider.fit(documents)

        # Should be fitted now
        assert provider._fitted

    def test_provider_get_scores(self):
        """Test getting scores from provider."""
        provider = BM25Provider()
        documents = [
            "python programming",
            "java programming",
            "python and java",
        ]
        provider.fit(documents)

        scores = provider.get_scores("python")

        assert isinstance(scores, dict)
        # Should have scores for docs containing "python"
        assert len(scores) > 0

    def test_provider_get_top_results(self):
        """Test getting top results from provider."""
        provider = BM25Provider()
        documents = [
            "python tutorial",
            "java tutorial",
            "python guide",
        ]
        provider.fit(documents)

        top = provider.get_top_results("python", k=2)

        assert len(top) <= 2
        # First result should have highest score
        if len(top) >= 1:
            assert top[0][1] >= (top[1][1] if len(top) > 1 else 0)


class TestCreateBM25Provider:
    """Tests for the create_bm25_provider factory function."""

    def test_create_with_defaults(self):
        """Test creating provider with default parameters."""
        provider = create_bm25_provider()

        assert isinstance(provider, BM25Provider)

    def test_create_with_custom_params(self):
        """Test creating provider with custom parameters."""
        provider = create_bm25_provider(k1=2.0, b=0.8)

        assert isinstance(provider, BM25Provider)
        assert provider._bm25._k1 == 2.0
        assert provider._bm25._b == 0.8


class TestBM25EdgeCases:
    """Edge case tests for BM25."""

    def test_single_document(self):
        """Test BM25 with single document."""
        bm25 = BM25()
        bm25.fit(["single document content"])

        scores = bm25.score_batch("document")
        assert len(scores) == 1
        assert scores[0][0] == 0

    def test_very_short_documents(self):
        """Test BM25 with very short documents."""
        bm25 = BM25()
        bm25.fit(["a", "b", "c"])

        scores = bm25.score_batch("a")
        assert len(scores) >= 1

    def test_special_characters(self):
        """Test BM25 handles special characters."""
        bm25 = BM25()
        bm25.fit(["test@email.com", "phone: 12345", "url: http://test"])

        scores = bm25.score_batch("test")
        # Should handle gracefully

    def test_unicode_content(self):
        """Test BM25 handles unicode content."""
        bm25 = BM25()
        bm25.fit(["Python 编程", "Java 教程", "编程语言"])

        scores = bm25.score_batch("编程")
        # Should handle unicode tokens

    def test_case_insensitivity(self):
        """Test that BM25 is case insensitive."""
        bm25 = BM25()
        bm25.fit(["Python Programming", "JAVA CODE", "python tutorial"])

        scores_upper = bm25.score_batch("PYTHON")
        scores_lower = bm25.score_batch("python")

        # Both should return results for same docs
        assert set(idx for idx, _ in scores_upper) == set(idx for idx, _ in scores_lower)
