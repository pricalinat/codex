"""Tests for query expansion and fallback strategies."""

import unittest
from src.test_analysis_assistant.query_expansion import (
    AdaptiveQueryStrategy,
    ExpansionResult,
    FallbackQuery,
    QueryFallbackGenerator,
    SemanticExpander,
    create_adaptive_strategy,
)


class TestQueryFallbackGenerator(unittest.TestCase):
    """Tests for QueryFallbackGenerator."""

    def setUp(self):
        self.generator = QueryFallbackGenerator(
            min_results_threshold=3,
            confidence_threshold=0.3,
            max_fallbacks=5,
        )

    def test_should_generate_fallbacks_low_results(self):
        """Test that fallback is triggered with low result count."""
        result = self.generator.should_generate_fallbacks(2, 0.8)
        self.assertTrue(result)

    def test_should_generate_fallbacks_low_confidence(self):
        """Test that fallback is triggered with low confidence."""
        result = self.generator.should_generate_fallbacks(5, 0.2)
        self.assertTrue(result)

    def test_should_not_fallback_good_results(self):
        """Test that fallback is not triggered with good results."""
        result = self.generator.should_generate_fallbacks(5, 0.8)
        self.assertFalse(result)

    def test_generate_fallbacks_error_pattern(self):
        """Test fallback generation for error patterns."""
        fallbacks = self.generator.generate_fallbacks(
            "ModuleNotFoundError: No module named 'psycopg2'"
        )

        # Should generate error pattern fallbacks
        error_queries = [f for f in fallbacks if f.strategy == "error_pattern"]
        self.assertGreater(len(error_queries), 0)

        # Should have import-related fallbacks
        import_queries = [f for f in fallbacks if "import" in f.query_text.lower()]
        self.assertGreater(len(import_queries), 0)

    def test_generate_fallbacks_tech_expansion(self):
        """Test fallback generation for technology terms."""
        fallbacks = self.generator.generate_fallbacks(
            "pytest test failing with python"
        )

        # Should have tech expansion fallbacks
        tech_queries = [f for f in fallbacks if f.strategy == "tech_expansion"]
        self.assertGreater(len(tech_queries), 0)

    def test_generate_fallbacks_max_limit(self):
        """Test that fallback count is limited."""
        generator = QueryFallbackGenerator(max_fallbacks=3)
        fallbacks = generator.generate_fallbacks(
            "assertion error in test"
        )

        self.assertLessEqual(len(fallbacks), 3)

    def test_generate_fallbacks_sorted_by_confidence(self):
        """Test that fallbacks are sorted by confidence."""
        fallbacks = self.generator.generate_fallbacks(
            "timeout error in ci pipeline"
        )

        # Should be sorted by confidence descending
        if len(fallbacks) > 1:
            for i in range(len(fallbacks) - 1):
                self.assertGreaterEqual(
                    fallbacks[i].confidence,
                    fallbacks[i + 1].confidence,
                )


class TestSemanticExpander(unittest.TestCase):
    """Tests for SemanticExpander."""

    def setUp(self):
        self.expander = SemanticExpander(
            use_edit_distance=True,
            use_domain_terms=True,
            max_expansions=10,
        )

    def test_expand_basic(self):
        """Test basic query expansion."""
        result = self.expander.expand("test failure")

        self.assertEqual(result.original_query, "test failure")
        self.assertGreater(result.expansions_count, 0)
        self.assertIsNotNone(result.combined_query)

    def test_expand_with_known_terms(self):
        """Test expansion with known terms for edit distance."""
        result = self.expander.expand(
            "assertion erro",  # typo
            known_terms=["assertion", "error", "test", "failure"]
        )

        # Should expand with known terms
        self.assertGreater(result.expansions_count, 0)

    def test_expand_max_limit(self):
        """Test that expansion count is limited."""
        expander = SemanticExpander(max_expansions=3)
        result = expander.expand("test error failure")

        self.assertLessEqual(len(result.expanded_queries), 3)

    def test_expand_returns_expansion_result(self):
        """Test that expansion returns proper result type."""
        result = self.expander.expand("pytest failing")

        self.assertIsInstance(result, ExpansionResult)
        self.assertIsInstance(result.expanded_queries, list)
        self.assertIsInstance(result.combined_query, str)


class TestAdaptiveQueryStrategy(unittest.TestCase):
    """Tests for AdaptiveQueryStrategy."""

    def setUp(self):
        self.strategy = AdaptiveQueryStrategy()

    def test_should_fallback(self):
        """Test fallback decision."""
        should_fb = self.strategy.should_fallback(2, 0.2)
        self.assertTrue(should_fb)

    def test_should_not_fallback_good_results(self):
        """Test no fallback for good results."""
        should_fb = self.strategy.should_fallback(5, 0.8)
        self.assertFalse(should_fb)

    def test_get_fallback_queries(self):
        """Test getting fallback queries."""
        fallbacks = self.strategy.get_fallback_queries(
            "import error in test"
        )

        self.assertIsInstance(fallbacks, list)
        self.assertGreater(len(fallbacks), 0)

    def test_get_expanded_queries(self):
        """Test getting expanded queries."""
        result = self.strategy.get_expanded_queries("test failure")

        self.assertIsInstance(result, ExpansionResult)

    def test_disabled_expansion(self):
        """Test with expansion disabled."""
        strategy = AdaptiveQueryStrategy(use_expansion=False)
        result = strategy.get_expanded_queries("test failure")

        self.assertEqual(result.expansions_count, 0)
        self.assertEqual(result.combined_query, "test failure")

    def test_disabled_fallback(self):
        """Test with fallback disabled."""
        strategy = AdaptiveQueryStrategy(fallback_on_low_confidence=False)
        should_fb = strategy.should_fallback(2, 0.2)

        self.assertFalse(should_fb)


class TestCreateAdaptiveStrategy(unittest.TestCase):
    """Tests for create_adaptive_strategy convenience function."""

    def test_create_default(self):
        """Test creating with default settings."""
        strategy = create_adaptive_strategy()

        self.assertIsInstance(strategy, AdaptiveQueryStrategy)

    def test_create_with_expansion_disabled(self):
        """Test creating with expansion disabled."""
        strategy = create_adaptive_strategy(use_expansion=False)

        result = strategy.get_expanded_queries("test")
        self.assertEqual(result.expansions_count, 0)

    def test_create_with_fallbacks_disabled(self):
        """Test creating with fallbacks disabled."""
        strategy = create_adaptive_strategy(use_fallbacks=False)

        should_fb = strategy.should_fallback(2, 0.2)
        self.assertFalse(should_fb)


class TestFallbackQuery(unittest.TestCase):
    """Tests for FallbackQuery dataclass."""

    def test_fallback_query_creation(self):
        """Test creating a FallbackQuery."""
        query = FallbackQuery(
            query_text="test failure",
            strategy="error_pattern",
            confidence=0.8,
            source_terms=["test", "failure"],
            expansion_type="lexical",
        )

        self.assertEqual(query.query_text, "test failure")
        self.assertEqual(query.strategy, "error_pattern")
        self.assertEqual(query.confidence, 0.8)
        self.assertEqual(query.source_terms, ["test", "failure"])
        self.assertEqual(query.expansion_type, "lexical")

    def test_fallback_query_defaults(self):
        """Test FallbackQuery with default values."""
        query = FallbackQuery(
            query_text="test",
            strategy="test",
            confidence=0.5,
        )

        self.assertEqual(query.source_terms, [])
        self.assertEqual(query.expansion_type, "lexical")


class TestExpansionResult(unittest.TestCase):
    """Tests for ExpansionResult dataclass."""

    def test_expansion_result_creation(self):
        """Test creating an ExpansionResult."""
        expanded = [
            FallbackQuery("test", "test", 0.5),
            FallbackQuery("failure", "test", 0.5),
        ]
        result = ExpansionResult(
            original_query="test failure",
            expanded_queries=expanded,
            combined_query="test failure",
            expansions_count=1,
            estimated_improvement=0.2,
        )

        self.assertEqual(result.original_query, "test failure")
        self.assertEqual(len(result.expanded_queries), 2)
        self.assertEqual(result.expansions_count, 1)
        self.assertEqual(result.estimated_improvement, 0.2)


if __name__ == "__main__":
    unittest.main()
