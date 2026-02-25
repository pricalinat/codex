"""Tests for query reformulator module."""

import pytest

from src.test_analysis_assistant.models import FailureCluster, FailureRecord
from src.test_analysis_assistant.query_reformulator import (
    ErrorCategory,
    FailureTrace,
    QueryReformulator,
    QueryVariant,
    ReformulatedQueries,
    _ERROR_CATEGORY_MAP,
    reformulate_queries,
)


class TestQueryReformulator:
    """Tests for QueryReformulator class."""

    def test_reformulator_initialization(self):
        """Test that QueryReformulator initializes correctly."""
        reformulator = QueryReformulator(max_variants_per_failure=5)
        assert reformulator._max_variants_per_failure == 5

    def test_error_category_map(self):
        """Test error category mapping."""
        assert _ERROR_CATEGORY_MAP["AssertionError"] == ErrorCategory.ASSERTION
        assert _ERROR_CATEGORY_MAP["ImportError"] == ErrorCategory.IMPORT
        assert _ERROR_CATEGORY_MAP["ModuleNotFoundError"] == ErrorCategory.IMPORT
        assert _ERROR_CATEGORY_MAP["TypeError"] == ErrorCategory.TYPE
        assert _ERROR_CATEGORY_MAP["TimeoutError"] == ErrorCategory.TIMEOUT
        assert _ERROR_CATEGORY_MAP["RuntimeError"] == ErrorCategory.RUNTIME

    def test_extract_symbols_from_test_name(self):
        """Test symbol extraction from test names."""
        reformulator = QueryReformulator()

        # Test module_class_method pattern
        symbols = reformulator._extract_symbols("test_auth_user_login")
        assert "auth" in symbols
        assert "user" in symbols

        # Test simple test name
        symbols = reformulator._extract_symbols("test_simple")
        assert "simple" in symbols

        # Test with multiple underscores
        symbols = reformulator._extract_symbols("test_api_v1_users_create")
        assert len(symbols) > 0

    def test_reformulate_from_empty_failures(self):
        """Test reformulation with empty failures."""
        reformulator = QueryReformulator()
        result = reformulator.reformulate_from_failures([], "test query")
        assert result.original_query == "test query"
        assert len(result.variants) == 1  # Only original query

    def test_reformulate_from_single_failure(self):
        """Test reformulation from single failure."""
        reformulator = QueryReformulator()
        failures = [
            FailureRecord(
                suite="test_auth",
                test_name="test_login_failure",
                file_path="tests/test_auth.py",
                error_type="AssertionError",
                message="Expected success but got failure",
                traceback_excerpt='File "test_auth.py", line 42, in test_login_failure',
            )
        ]
        result = reformulator.reformulate_from_failures(failures)

        assert len(result.variants) > 0
        assert len(result.traces) == 1
        # Check that we have test_context intent variants
        assert any(v.intent == "test_context" for v in result.variants)

    def test_reformulate_from_multiple_failures(self):
        """Test reformulation from multiple failures."""
        reformulator = QueryReformulator()
        failures = [
            FailureRecord(
                suite="test_auth",
                test_name="test_login",
                file_path="tests/test_auth.py",
                error_type="AssertionError",
                message="Expected True but got False",
                traceback_excerpt="",
            ),
            FailureRecord(
                suite="test_api",
                test_name="test_user_fetch",
                file_path="tests/test_api.py",
                error_type="TypeError",
                message="expected str got int",
                traceback_excerpt="",
            ),
        ]
        result = reformulator.reformulate_from_failures(failures)

        assert len(result.variants) > 1
        assert len(result.traces) == 2

    def test_reformulate_with_original_query(self):
        """Test reformulation with original query."""
        reformulator = QueryReformulator()
        failures = [
            FailureRecord(
                suite="test_auth",
                test_name="test_login",
                file_path="tests/test_auth.py",
                error_type="RuntimeError",
                message="Something went wrong",
                traceback_excerpt="",
            )
        ]
        result = reformulator.reformulate_from_failures(failures, "my original query")

        assert result.original_query == "my original query"
        # Original query should be included as first variant
        assert result.variants[0].query_text == "my original query"

    def test_generate_error_queries_assertion(self):
        """Test error-specific query generation for AssertionError."""
        reformulator = QueryReformulator()
        failure = FailureRecord(
            suite="test",
            test_name="test",
            file_path="test.py",
            error_type="AssertionError",
            message="Expected 200 but got 404",
            traceback_excerpt="",
        )

        variants = reformulator._generate_error_queries(
            failure, ErrorCategory.ASSERTION
        )

        assert len(variants) > 0
        assert all(v.error_type == "AssertionError" for v in variants)

    def test_generate_error_queries_import(self):
        """Test error-specific query generation for ImportError."""
        reformulator = QueryReformulator()
        failure = FailureRecord(
            suite="test",
            test_name="test",
            file_path="test.py",
            error_type="ModuleNotFoundError",
            message="No module named 'requests'",
            traceback_excerpt="",
        )

        variants = reformulator._generate_error_queries(
            failure, ErrorCategory.IMPORT
        )

        assert len(variants) > 0

    def test_extract_error_detail_assertion(self):
        """Test error detail extraction from assertion messages."""
        reformulator = QueryReformulator()

        # Test expected/actual extraction
        detail = reformulator._extract_error_detail(
            "Expected 'success' but got 'failure'", ErrorCategory.ASSERTION
        )
        # Should extract something (actual or expected)
        assert isinstance(detail, str)

    def test_extract_source_hints(self):
        """Test source hint extraction from traceback."""
        reformulator = QueryReformulator()

        traceback = '''File "/path/to/test.py", line 42, in test_something
    assert result == expected
File "/path/to/app.py", line 100, in process'''

        hints = reformulator._extract_source_hints(traceback)

        assert len(hints) > 0
        # Should contain file paths
        assert any("test.py" in h or "app.py" in h for h in hints)

    def test_reformulate_from_clusters(self):
        """Test reformulation from failure clusters."""
        reformulator = QueryReformulator()
        clusters = [
            FailureCluster(
                cluster_id="C01",
                reason="Test failure",
                error_type="AssertionError",
                count=3,
                tests=["test_auth::test_login", "test_auth::test_logout"],
            )
        ]

        result = reformulator.reformulate_from_clusters(clusters)

        assert len(result.variants) > 0
        assert result.traces == []

    def test_deduplication(self):
        """Test query variant deduplication."""
        reformulator = QueryReformulator()

        variants = [
            QueryVariant("same query", 0.5, "intent1"),
            QueryVariant("SAME QUERY", 0.8, "intent2"),  # Should dedupe with higher weight
            QueryVariant("different query", 0.6, "intent3"),
        ]

        deduped = reformulator._deduplicate_variants(variants)

        assert len(deduped) == 2
        # The "same query" should have weight 0.8 (the higher one)
        same_query = next(v for v in deduped if v.query_text.lower() == "same query")
        assert same_query.weight == 0.8


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_reformulate_queries_basic(self):
        """Test basic reformulate_queries function."""
        failures = [
            FailureRecord(
                suite="test",
                test_name="test_example",
                file_path="test.py",
                error_type="RuntimeError",
                message="Error",
                traceback_excerpt="",
            )
        ]

        result = reformulate_queries(failures, "original")

        assert isinstance(result, ReformulatedQueries)
        assert result.original_query == "original"

    def test_reformulate_queries_max_variants(self):
        """Test max_variants limit."""
        failures = [
            FailureRecord(
                suite="test",
                test_name=f"test_{i}",
                file_path="test.py",
                error_type="AssertionError",
                message="Error",
                traceback_excerpt="",
            )
            for i in range(10)
        ]

        result = reformulate_queries(failures, "", max_variants=3)

        assert len(result.variants) <= 3


class TestQueryVariant:
    """Tests for QueryVariant dataclass."""

    def test_query_variant_creation(self):
        """Test QueryVariant creation."""
        variant = QueryVariant(
            query_text="test query",
            weight=0.8,
            intent="test_intent",
            error_type="AssertionError",
            focus_area="test_coverage",
        )

        assert variant.query_text == "test query"
        assert variant.weight == 0.8
        assert variant.intent == "test_intent"
        assert variant.error_type == "AssertionError"
        assert variant.focus_area == "test_coverage"


class TestFailureTrace:
    """Tests for FailureTrace dataclass."""

    def test_failure_trace_creation(self):
        """Test FailureTrace creation."""
        failure = FailureRecord(
            suite="test",
            test_name="test_example",
            file_path="test.py",
            error_type="TypeError",
            message="Error",
            traceback_excerpt="",
        )

        trace = FailureTrace(
            failure=failure,
            related_queries=[],
            source_hints=["hint1"],
            extracted_symbols=["symbol1"],
        )

        assert trace.failure == failure
        assert trace.source_hints == ["hint1"]
        assert trace.extracted_symbols == ["symbol1"]
