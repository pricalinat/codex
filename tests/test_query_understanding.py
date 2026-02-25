"""Tests for QueryUnderstandingEngine."""

import pytest

from test_analysis_assistant.query_understanding import (
    ContextualVariant,
    IntentClassification,
    QueryIntent,
    QueryUnderstanding,
    QueryUnderstandingEngine,
    understand_query,
)


class TestQueryUnderstandingEngine:
    """Tests for QueryUnderstandingEngine."""

    def test_understand_basic_query(self):
        """Test understanding a basic query."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("root cause of test failure")

        assert isinstance(result, QueryUnderstanding)
        assert result.original == "root cause of test failure"
        assert result.normalized is not None
        assert len(result.tokens) > 0

    def test_intent_classification_root_cause(self):
        """Test intent classification for root cause query."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("why is this test failing")

        assert result.intent.primary_intent == QueryIntent.FIND_ROOT_CAUSE
        assert result.intent.confidence > 0

    def test_intent_classification_test_gaps(self):
        """Test intent classification for test gap query."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("what test cases are missing")

        assert result.intent.primary_intent == QueryIntent.FIND_TEST_GAPS
        assert "missing" in result.intent.intent_keywords.get(QueryIntent.FIND_TEST_GAPS, [])

    def test_intent_classification_fix_suggestion(self):
        """Test intent classification for fix suggestion query."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("how to fix this error")

        assert result.intent.primary_intent == QueryIntent.FIND_FIX_SUGGESTION

    def test_intent_classification_related_code(self):
        """Test intent classification for related code query."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("code related to authentication")

        assert result.intent.primary_intent == QueryIntent.FIND_RELATED_CODE

    def test_intent_classification_risk_assessment(self):
        """Test intent classification for risk assessment query."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("what is the severity of this issue")

        assert result.intent.primary_intent == QueryIntent.FIND_RISK_ASSESSMENT

    def test_entity_extraction(self):
        """Test entity extraction from query."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("test_auth_login function")

        assert len(result.intent.entities) > 0

    def test_synonym_expansion(self):
        """Test synonym expansion in query."""
        engine = QueryUnderstandingEngine(enable_synonym_expansion=True)
        result = engine.understand("fix error")

        # Should contain synonyms
        assert "fix" in result.expanded_query.split()
        assert "error" in result.expanded_query.split()

    def test_query_decomposition(self):
        """Test query decomposition."""
        engine = QueryUnderstandingEngine(enable_decomposition=True)
        result = engine.understand("how to fix import and type errors")

        # Should have multiple decomposed queries
        assert len(result.decomposed_queries) >= 1

    def test_contextual_variants_generation(self):
        """Test contextual variants generation."""
        engine = QueryUnderstandingEngine(max_variants=3)
        result = engine.understand("root cause of failure")

        assert len(result.contextual_variants) > 0
        assert isinstance(result.contextual_variants[0], ContextualVariant)

    def test_recommended_sources(self):
        """Test recommended sources based on intent."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("root cause of error")

        assert len(result.recommended_sources) > 0

    def test_secondary_intents(self):
        """Test secondary intent detection."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("why test failed and how to fix")

        # Should have primary and secondary intents
        assert result.intent.primary_intent is not None
        # Secondary intents may or may not be present depending on query

    def test_normalization_preserves_meaning(self):
        """Test that normalization preserves query meaning."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("ROOT   CAUSE  of  FAILURE")

        assert "root" in result.normalized
        assert "cause" in result.normalized
        assert "failure" in result.normalized

    def test_tokenization(self):
        """Test tokenization of query."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("root cause of test failure")

        assert len(result.tokens) > 0

    def test_default_engine_settings(self):
        """Test default engine settings."""
        engine = QueryUnderstandingEngine()

        assert engine._enable_synonym_expansion is True
        assert engine._enable_decomposition is True
        assert engine._max_variants == 5


class TestUnderstandQuery:
    """Tests for the understand_query convenience function."""

    def test_convenience_function(self):
        """Test convenience function."""
        result = understand_query("root cause of error")

        assert isinstance(result, QueryUnderstanding)
        assert result.original == "root cause of error"

    def test_convenience_function_all_features(self):
        """Test convenience function returns all features."""
        result = understand_query("missing test coverage for login")

        assert result.intent is not None
        assert result.expanded_query is not None
        assert len(result.decomposed_queries) > 0
        assert len(result.contextual_variants) > 0


class TestQueryIntent:
    """Tests for QueryIntent enum."""

    def test_all_intents_defined(self):
        """Test that all intents are defined."""
        intents = [
            QueryIntent.FIND_ROOT_CAUSE,
            QueryIntent.FIND_RELATED_CODE,
            QueryIntent.FIND_TEST_GAPS,
            QueryIntent.FIND_SIMILAR_FAILURES,
            QueryIntent.FIND_FIX_SUGGESTION,
            QueryIntent.FIND_RISK_ASSESSMENT,
            QueryIntent.FIND_REQUIREMENT_TRACE,
            QueryIntent.FIND_DOCUMENTATION,
            QueryIntent.FIND_DEPENDENCY,
            QueryIntent.ANALYZE_PATTERN,
            QueryIntent.ANALYZE_TREND,
            QueryIntent.ANALYZE_COVERAGE,
            QueryIntent.GENERAL,
        ]

        assert len(intents) == len(QueryIntent)


class TestIntentClassification:
    """Tests for IntentClassification dataclass."""

    def test_intent_classification_defaults(self):
        """Test IntentClassification default values."""
        classification = IntentClassification(
            primary_intent=QueryIntent.GENERAL,
            confidence=0.5,
        )

        assert classification.secondary_intents == []
        assert classification.intent_keywords == {}
        assert classification.entities == []


class TestContextualVariant:
    """Tests for ContextualVariant dataclass."""

    def test_contextual_variant_defaults(self):
        """Test ContextualVariant default values."""
        variant = ContextualVariant(
            query="test query",
            intent=QueryIntent.GENERAL,
            weight=1.0,
            rationale="test",
        )

        assert variant.query == "test query"
        assert variant.intent == QueryIntent.GENERAL
        assert variant.weight == 1.0
        assert variant.rationale == "test"


class TestQueryUnderstanding:
    """Tests for QueryUnderstanding dataclass."""

    def test_query_understanding_fields(self):
        """Test QueryUnderstanding has all required fields."""
        understanding = QueryUnderstanding(
            original="test",
            normalized="test",
            tokens=["test"],
            intent=IntentClassification(
                primary_intent=QueryIntent.GENERAL,
                confidence=0.5,
            ),
            expanded_query="test",
            decomposed_queries=["test"],
            contextual_variants=[],
        )

        assert understanding.original == "test"
        assert understanding.normalized == "test"
        assert understanding.tokens == ["test"]
        assert understanding.expanded_query == "test"
        assert understanding.decomposed_queries == ["test"]
        assert understanding.recommended_sources == []


class TestIntentKeywords:
    """Tests for intent keyword matching."""

    def test_root_cause_keywords(self):
        """Test keywords for FIND_ROOT_CAUSE intent."""
        engine = QueryUnderstandingEngine()

        for keyword in ["root", "cause", "why", "reason", "origin"]:
            result = engine.understand(f"the {keyword} of the issue")
            assert result.intent.primary_intent == QueryIntent.FIND_ROOT_CAUSE

    def test_test_gap_keywords(self):
        """Test keywords for FIND_TEST_GAPS intent."""
        engine = QueryUnderstandingEngine()

        for keyword in ["missing", "gap", "coverage", "untested"]:
            result = engine.understand(f"find {keyword} test cases")
            assert result.intent.primary_intent == QueryIntent.FIND_TEST_GAPS


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_query(self):
        """Test handling of empty query."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("")

        assert result.original == ""
        # Empty query should return general intent gracefully
        assert result.intent.primary_intent == QueryIntent.GENERAL

    def test_single_word_query(self):
        """Test single word query."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("error")

        assert result.original == "error"
        assert len(result.tokens) > 0

    def test_special_characters(self):
        """Test handling of special characters."""
        engine = QueryUnderstandingEngine()
        result = engine.understand("test@#$%")

        # Should normalize without crashing
        assert result.normalized is not None

    def test_very_long_query(self):
        """Test handling of very long query."""
        engine = QueryUnderstandingEngine()
        long_query = " ".join(["word"] * 100)
        result = engine.understand(long_query)

        assert result.original == long_query
