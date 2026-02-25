"""Query reformulation and test failure tracing for improved retrieval."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .models import FailureCluster, FailureRecord


class ErrorCategory(str, Enum):
    """Categorization of error types for targeted query generation."""

    ASSERTION = "assertion"  # AssertionError, assert failures
    IMPORT = "import"  # ImportError, ModuleNotFoundError
    TYPE = "type"  # TypeError, Type mismatch
    TIMEOUT = "timeout"  # TimeoutError, slow operations
    RUNTIME = "runtime"  # RuntimeError, unhandled exceptions
    SYNTAX = "syntax"  # SyntaxError
    ASSERTION_DETAIL = "assertion_detail"  # expected vs actual value extraction


# Error type to category mapping
_ERROR_CATEGORY_MAP: Dict[str, ErrorCategory] = {
    "AssertionError": ErrorCategory.ASSERTION,
    "ImportError": ErrorCategory.IMPORT,
    "ModuleNotFoundError": ErrorCategory.IMPORT,
    "TypeError": ErrorCategory.TYPE,
    "TimeoutError": ErrorCategory.TIMEOUT,
    "RuntimeError": ErrorCategory.RUNTIME,
    "SyntaxError": ErrorCategory.SYNTAX,
}


@dataclass
class QueryVariant:
    """A reformulated query variant with metadata."""

    query_text: str
    weight: float
    intent: str
    error_type: Optional[str] = None
    focus_area: Optional[str] = None


@dataclass
class FailureTrace:
    """Trace from a failing test to related code/context."""

    failure: FailureRecord
    related_queries: List[QueryVariant] = field(default_factory=list)
    source_hints: List[str] = field(default_factory=list)
    extracted_symbols: List[str] = field(default_factory=list)


@dataclass
class ReformulatedQueries:
    """Collection of reformulated queries from test failures."""

    original_query: str
    variants: List[QueryVariant] = field(default_factory=list)
    traces: List[FailureTrace] = field(default_factory=list)


class QueryReformulator:
    """Reformulates queries based on test failure patterns for improved retrieval.

    This class generates targeted queries that:
    1. Extract symbols (function names, class names) from test names
    2. Generate error-specific query variants based on error types
    3. Create focused queries for different analysis goals
    """

    # Query templates for different error categories
    _ERROR_QUERY_TEMPLATES: Dict[ErrorCategory, List[str]] = {
        ErrorCategory.ASSERTION: [
            "assertion failure expected actual value mismatch",
            "assert test expected got difference",
        ],
        ErrorCategory.IMPORT: [
            "import module dependency missing import path",
            "module not found import resolution",
        ],
        ErrorCategory.TYPE: [
            "type error type mismatch invalid argument",
            "function parameter type signature",
        ],
        ErrorCategory.TIMEOUT: [
            "timeout slow operation async waiting",
            "timeout handling retry logic",
        ],
        ErrorCategory.RUNTIME: [
            "runtime exception error handling edge case",
            "unhandled exception error path",
        ],
        ErrorCategory.SYNTAX: [
            "syntax error parsing code structure",
        ],
    }

    # Focus area query templates
    _FOCUS_QUERIES: Dict[str, List[str]] = {
        "test_gap": [
            "missing test coverage edge case negative test",
            "test case gap boundary condition",
        ],
        "root_cause": [
            "root cause failure origin traceback",
            "failure reason underlying cause",
        ],
        "risk": [
            "release risk blocking issue severity",
            "critical bug security issue",
        ],
    }

    def __init__(self, max_variants_per_failure: int = 3) -> None:
        self._max_variants_per_failure = max_variants_per_failure

    def reformulate_from_failures(
        self,
        failures: Sequence[FailureRecord],
        original_query: str = "",
    ) -> ReformulatedQueries:
        """Generate reformulated queries from test failures.

        Args:
            failures: List of failing test records
            original_query: Optional original user query

        Returns:
            ReformulatedQueries with variants and traces
        """
        all_variants: List[QueryVariant] = []
        traces: List[FailureTrace] = []

        # Add original query if provided
        if original_query:
            all_variants.append(
                QueryVariant(
                    query_text=original_query,
                    weight=1.0,
                    intent="original",
                )
            )

        for failure in failures:
            trace = self._trace_failure(failure)
            traces.append(trace)
            all_variants.extend(trace.related_queries)

        # Deduplicate variants while preserving order
        deduped = self._deduplicate_variants(all_variants)

        return ReformulatedQueries(
            original_query=original_query,
            variants=deduped,
            traces=traces,
        )

    def reformulate_from_clusters(
        self,
        clusters: Sequence[FailureCluster],
        original_query: str = "",
    ) -> ReformulatedQueries:
        """Generate reformulated queries from failure clusters.

        Args:
            clusters: List of failure clusters
            original_query: Optional original user query

        Returns:
            ReformulatedQueries with variants
        """
        all_variants: List[QueryVariant] = []

        if original_query:
            all_variants.append(
                QueryVariant(
                    query_text=original_query,
                    weight=1.0,
                    intent="original",
                )
            )

        for cluster in clusters:
            variants = self._generate_cluster_variants(cluster)
            all_variants.extend(variants)

        deduped = self._deduplicate_variants(all_variants)

        return ReformulatedQueries(
            original_query=original_query,
            variants=deduped,
            traces=[],
        )

    def _trace_failure(self, failure: FailureRecord) -> FailureTrace:
        """Generate trace for a single failure.

        Args:
            failure: The failing test record

        Returns:
            FailureTrace with related queries
        """
        related_queries: List[QueryVariant] = []
        source_hints: List[str] = []
        symbols: List[str] = []

        # Extract symbols from test name
        extracted = self._extract_symbols(failure.test_name)
        symbols.extend(extracted)

        # Generate base query from test name
        test_query = self._generate_test_query(failure)
        related_queries.append(
            QueryVariant(
                query_text=test_query,
                weight=0.9,
                intent="test_context",
                error_type=failure.error_type,
                focus_area="test_implementation",
            )
        )

        # Generate error-specific queries
        category = _ERROR_CATEGORY_MAP.get(failure.error_type, ErrorCategory.RUNTIME)
        error_queries = self._generate_error_queries(failure, category)
        related_queries.extend(error_queries)

        # Extract source hints from traceback
        if failure.traceback_excerpt:
            hints = self._extract_source_hints(failure.traceback_excerpt)
            source_hints.extend(hints)

        # Add focus area queries
        focus_queries = self._generate_focus_queries(failure)
        related_queries.extend(focus_queries)

        # Limit variants per failure
        return FailureTrace(
            failure=failure,
            related_queries=related_queries[: self._max_variants_per_failure],
            source_hints=source_hints,
            extracted_symbols=symbols,
        )

    def _extract_symbols(self, test_name: str) -> List[str]:
        """Extract function/class symbols from test name.

        Args:
            test_name: Name of the test

        Returns:
            List of extracted symbols
        """
        symbols: List[str] = []

        # Strip "test_" prefix if present
        name_without_prefix = test_name
        if test_name.startswith("test_"):
            name_without_prefix = test_name[5:]
        elif test_name.startswith("test"):
            name_without_prefix = test_name[4:]

        # Split by underscore to get components
        if "_" in name_without_prefix:
            groups = name_without_prefix.split("_")
            # Add individual components
            symbols.extend([g for g in groups if g])
            # Add combined
            if len(groups) > 1:
                symbols.append(".".join(groups))
        elif name_without_prefix:
            # Single component
            symbols.append(name_without_prefix)

        # Extract camelCase and PascalCase
        camel_matches = re.findall(r"[a-z]+[A-Z][a-zA-Z]*", test_name)
        symbols.extend(camel_matches)

        return list(set(symbols))[:5]  # Limit to 5 symbols

    def _generate_test_query(self, failure: FailureRecord) -> str:
        """Generate query from test context.

        Args:
            failure: The failing test

        Returns:
            Query string
        """
        parts = [failure.suite]

        # Add extracted symbols
        symbols = self._extract_symbols(failure.test_name)
        if symbols:
            parts.extend(symbols[:2])

        # Add error type context
        parts.append(failure.error_type.lower())

        return " ".join(parts)

    def _generate_error_queries(
        self,
        failure: FailureRecord,
        category: ErrorCategory,
    ) -> List[QueryVariant]:
        """Generate queries specific to error category.

        Args:
            failure: The failing test
            category: Error category

        Returns:
            List of QueryVariants
        """
        variants: List[QueryVariant] = []

        templates = self._ERROR_QUERY_TEMPLATES.get(category, [])

        # Extract specific details from message
        detail = self._extract_error_detail(failure.message, category)

        for template in templates[:2]:
            if detail:
                query = f"{template} {detail}"
            else:
                query = template

            variants.append(
                QueryVariant(
                    query_text=query,
                    weight=0.75,
                    intent=f"error_{category.value}",
                    error_type=failure.error_type,
                    focus_area=f"{category.value}_analysis",
                )
            )

        return variants

    def _extract_error_detail(self, message: str, category: ErrorCategory) -> str:
        """Extract specific details from error message.

        Args:
            message: Error message
            category: Error category

        Returns:
            Extracted detail string
        """
        if category == ErrorCategory.ASSERTION:
            # Extract expected/actual from assertion
            expected_match = re.search(r"expected[:\s]+(.+?)(?:\s+but|\s+got|\s+actual)", message, re.IGNORECASE)
            if expected_match:
                return expected_match.group(1).strip()[:50]
            actual_match = re.search(r"(?:got|actual)[:\s]+(.+?)(?:\s+expected|$)", message, re.IGNORECASE)
            if actual_match:
                return actual_match.group(1).strip()[:50]

        elif category == ErrorCategory.IMPORT:
            # Extract module name
            module_match = re.search(r"module ['\"]([^'\"]+)['\"]", message)
            if module_match:
                return module_match.group(1)

        elif category == ErrorCategory.TYPE:
            # Extract type names
            types = re.findall(r"'([^']+)'", message)
            if types:
                return " ".join(types[:2])

        return ""

    def _extract_source_hints(self, traceback: str) -> List[str]:
        """Extract source file hints from traceback.

        Args:
            traceback: Traceback excerpt

        Returns:
            List of source hints
        """
        hints: List[str] = []

        # Extract file paths
        file_matches = re.findall(r"File ['\"]([^'\"]+)['\"]", traceback)
        hints.extend(file_matches)

        # Extract line numbers
        line_matches = re.findall(r", line (\d+)", traceback)
        hints.extend([f"line {lm}" for lm in line_matches[:3]])

        # Extract function names from traceback
        func_matches = re.findall(r"in (\w+)", traceback)
        hints.extend(func_matches[:3])

        return list(set(hints))[:5]

    def _generate_focus_queries(self, failure: FailureRecord) -> List[QueryVariant]:
        """Generate focus area queries.

        Args:
            failure: The failing test

        Returns:
            List of QueryVariants
        """
        variants: List[QueryVariant] = []

        # Determine applicable focus areas
        focus_areas = []
        if failure.error_type in ("AssertionError", "TypeError"):
            focus_areas.append("test_gap")
        focus_areas.append("root_cause")
        if failure.error_type in ("RuntimeError", "TimeoutError"):
            focus_areas.append("risk")

        for area in focus_areas:
            templates = self._FOCUS_QUERIES.get(area, [])
            for template in templates[:1]:
                variants.append(
                    QueryVariant(
                        query_text=template,
                        weight=0.65,
                        intent=f"focus_{area}",
                        error_type=failure.error_type,
                        focus_area=area,
                    )
                )

        return variants

    def _generate_cluster_variants(self, cluster: FailureCluster) -> List[QueryVariant]:
        """Generate query variants for a failure cluster.

        Args:
            cluster: Failure cluster

        Returns:
            List of QueryVariants
        """
        variants: List[QueryVariant] = []

        # Generate queries based on cluster error type
        category = _ERROR_CATEGORY_MAP.get(cluster.error_type, ErrorCategory.RUNTIME)
        templates = self._ERROR_QUERY_TEMPLATES.get(category, [])

        for i, template in enumerate(templates[:2]):
            variants.append(
                QueryVariant(
                    query_text=f"{template} {cluster.count} occurrence",
                    weight=0.8 - (i * 0.1),
                    intent=f"cluster_{category.value}",
                    error_type=cluster.error_type,
                    focus_area=f"{category.value}_cluster",
                )
            )

        # Add tests from cluster as hints
        if cluster.tests:
            sample_tests = cluster.tests[:2]
            for test in sample_tests:
                symbols = self._extract_symbols(test)
                if symbols:
                    variants.append(
                        QueryVariant(
                            query_text=" ".join(symbols[:2]),
                            weight=0.7,
                            intent="cluster_symbol",
                            error_type=cluster.error_type,
                            focus_area="cluster_context",
                        )
                    )

        return variants

    def _deduplicate_variants(
        self,
        variants: List[QueryVariant],
    ) -> List[QueryVariant]:
        """Deduplicate query variants while preserving order and best weights.

        Args:
            variants: List of query variants

        Returns:
            Deduplicated list
        """
        seen: Dict[str, QueryVariant] = {}

        for variant in variants:
            key = variant.query_text.lower()
            if key not in seen or variant.weight > seen[key].weight:
                seen[key] = variant

        # Return in order of first appearance weight
        return list(seen.values())


def reformulate_queries(
    failures: Sequence[FailureRecord],
    original_query: str = "",
    max_variants: int = 6,
) -> ReformulatedQueries:
    """Convenience function to reformulate queries from failures.

    Args:
        failures: List of failing test records
        original_query: Optional original query
        max_variants: Maximum number of variants to return

    Returns:
        ReformulatedQueries with variants
    """
    reformulator = QueryReformulator(max_variants_per_failure=max_variants)
    result = reformulator.reformulate_from_failures(failures, original_query)
    result.variants = result.variants[:max_variants]
    return result
