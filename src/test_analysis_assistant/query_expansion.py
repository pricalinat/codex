"""Query fallback and expansion strategies for improved retrieval.

This module provides intelligent query recovery mechanisms:
- QueryFallbackGenerator: Generates alternative queries when initial retrieval fails
- SemanticExpander: Expands queries with related terms using similarity
- AdaptiveQueryStrategy: Combines multiple strategies for robust retrieval

These components help when:
- Initial query returns too few results
- Results have low confidence scores
- Relevant content uses different terminology
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .retrieval import QueryPlan, SourceType


# Common error/failure patterns for query generation
_ERROR_SYNONYMS = {
    "assertion": ["assert", "assertionerror", "assertion failed", "check failed"],
    "import": ["import error", "modulenotfound", "importerror", "missing module"],
    "timeout": ["timeout", "timed out", "deadline", "took too long"],
    "connection": ["connection refused", "connection error", "network error", "connect failed"],
    "permission": ["permission denied", "access denied", "unauthorized", "forbidden"],
    "null": ["none", "null", "undefined", "nan", "missing value"],
    "index": ["indexerror", "index out of range", "out of bounds"],
    "type": ["typeerror", "type mismatch", "wrong type", "cannot convert"],
    "value": ["valueerror", "invalid value", "bad value"],
    "memory": ["out of memory", "memory error", "oom", "heap"],
}

# Technology/framework related term expansions
_TECH_EXPANSIONS = {
    "python": ["pytest", "unittest", "pytest", "pip", "virtualenv", "venv"],
    "javascript": ["node", "npm", "yarn", "jest", "mocha", "typescript"],
    "java": ["maven", "gradle", "junit", "spring", "hibernate"],
    "database": ["sql", "postgresql", "mysql", "mongodb", "redis", "query"],
    "api": ["rest", "graphql", "http", "endpoint", "request", "response"],
    "web": ["http", "html", "css", "frontend", "backend", "fullstack"],
    "test": ["unit", "integration", "e2e", "functional", "acceptance"],
    "ci": ["github actions", "gitlab ci", "jenkins", "circleci", "travis"],
    "docker": ["container", "kubernetes", "k8s", "pod", "deployment"],
}

# Test failure related terms
_TEST_FAILURE_TERMS = {
    "flaky": ["intermittent", "sporadic", "unstable", "random"],
    "failure": ["fail", "error", "exception", "crash", "broken"],
    "passing": ["success", "pass", "green", "working"],
    "debug": ["debugging", "troubleshoot", "investigate", "diagnose"],
    "fix": ["repair", "resolve", "correct", "patch"],
}


@dataclass
class FallbackQuery:
    """A generated fallback query with metadata."""
    query_text: str
    strategy: str  # error_pattern, synonym, tech_expansion, generalization, specialization
    confidence: float  # 0-1, how likely this query will help
    source_terms: List[str] = field(default_factory=list)  # Terms that triggered this query
    expansion_type: str = "lexical"  # lexical, semantic, hybrid


@dataclass
class ExpansionResult:
    """Result of query expansion."""
    original_query: str
    expanded_queries: List[FallbackQuery]
    combined_query: str  # All terms combined
    expansions_count: int
    estimated_improvement: float  # Expected improvement in recall


class QueryFallbackGenerator:
    """Generates fallback queries when initial retrieval fails.

    This helps improve recall when the initial query doesn't return good results
    by generating alternative queries based on:
    - Error pattern matching
    - Synonym expansion
    - Technology/framework context
    - Term generalization/specialization
    """

    def __init__(
        self,
        min_results_threshold: int = 3,
        confidence_threshold: float = 0.3,
        max_fallbacks: int = 5,
    ) -> None:
        """Initialize the fallback generator.

        Args:
            min_results_threshold: Minimum results to consider retrieval successful
            confidence_threshold: Minimum confidence for fallback queries
            max_fallbacks: Maximum number of fallback queries to generate
        """
        self._min_results_threshold = min_results_threshold
        self._confidence_threshold = confidence_threshold
        self._max_fallbacks = max_fallbacks

    def should_generate_fallbacks(
        self,
        result_count: int,
        top_confidence: float,
    ) -> bool:
        """Determine if fallback queries should be generated.

        Args:
            result_count: Number of results from initial query
            top_confidence: Confidence of top result

        Returns:
            True if fallbacks should be generated
        """
        return (
            result_count < self._min_results_threshold
            or top_confidence < self._confidence_threshold
        )

    def generate_fallbacks(
        self,
        query_text: str,
        query_plan: Optional[QueryPlan] = None,
    ) -> List[FallbackQuery]:
        """Generate fallback queries for a failed/weak initial retrieval.

        Args:
            query_text: Original query text
            query_plan: Optional query plan with parsed intent

        Returns:
            List of fallback queries sorted by confidence
        """
        fallback_queries: List[FallbackQuery] = []
        query_lower = query_text.lower()
        tokens = set(re.findall(r'\b\w+\b', query_lower))

        # 1. Error pattern based fallbacks
        error_fallbacks = self._generate_error_pattern_fallbacks(query_text, tokens)
        fallback_queries.extend(error_fallbacks)

        # 2. Synonym based fallbacks
        synonym_fallbacks = self._generate_synonym_fallbacks(query_text, tokens)
        fallback_queries.extend(synonym_fallbacks)

        # 3. Technology expansion fallbacks
        tech_fallbacks = self._generate_tech_expansions(query_text, tokens)
        fallback_queries.extend(tech_fallbacks)

        # 4. Test failure term expansions
        test_fallbacks = self._generate_test_failure_fallbacks(query_text, tokens)
        fallback_queries.extend(test_fallbacks)

        # 5. Generalization fallbacks (remove specific terms)
        generalization_fallbacks = self._generate_generalization_fallbacks(query_text, tokens)
        fallback_queries.extend(generalization_fallbacks)

        # 6. Specialization fallbacks (add context-specific terms)
        if query_plan:
            specialization_fallbacks = self._generate_specialization_fallbacks(query_text, query_plan)
            fallback_queries.extend(specialization_fallbacks)

        # Sort by confidence and limit
        fallback_queries.sort(key=lambda x: x.confidence, reverse=True)
        return fallback_queries[:self._max_fallbacks]

    def _generate_error_pattern_fallbacks(
        self,
        query_text: str,
        tokens: Set[str],
    ) -> List[FallbackQuery]:
        """Generate fallbacks based on error patterns."""
        fallbacks = []

        for error_type, synonyms in _ERROR_SYNONYMS.items():
            # Check if any synonym is in the query
            if any(syn in query_text.lower() for syn in synonyms):
                # Add generic version
                fallbacks.append(FallbackQuery(
                    query_text=f"{error_type} test failure",
                    strategy="error_pattern",
                    confidence=0.7,
                    source_terms=[error_type],
                ))
                # Add specific test context
                fallbacks.append(FallbackQuery(
                    query_text=f"fix {error_type} in test",
                    strategy="error_pattern",
                    confidence=0.6,
                    source_terms=[error_type],
                ))

        return fallbacks

    def _generate_synonym_fallbacks(
        self,
        query_text: str,
        tokens: Set[str],
    ) -> List[FallbackQuery]:
        """Generate fallbacks based on synonyms."""
        fallbacks = []

        # Build synonym expansion
        for token in tokens:
            for error_type, synonyms in _ERROR_SYNONYMS.items():
                if token in synonyms:
                    # Add other synonyms from same category
                    for syn in synonyms:
                        if syn != token:
                            new_query = query_text.lower().replace(token, syn)
                            if new_query != query_text.lower():
                                fallbacks.append(FallbackQuery(
                                    query_text=new_query,
                                    strategy="synonym",
                                    confidence=0.5,
                                    source_terms=[token, syn],
                                ))

        return fallbacks

    def _generate_tech_expansions(
        self,
        query_text: str,
        tokens: Set[str],
    ) -> List[FallbackQuery]:
        """Generate fallbacks based on technology context."""
        fallbacks = []

        for tech, expansions in _TECH_EXPANSIONS.items():
            if tech in tokens:
                for exp in expansions:
                    # Add technology context
                    fallbacks.append(FallbackQuery(
                        query_text=f"{query_text} {exp}",
                        strategy="tech_expansion",
                        confidence=0.4,
                        source_terms=[tech, exp],
                    ))

        return fallbacks

    def _generate_test_failure_fallbacks(
        self,
        query_text: str,
        tokens: Set[str],
    ) -> List[FallbackQuery]:
        """Generate fallbacks based on test failure terminology."""
        fallbacks = []

        for term, expansions in _TEST_FAILURE_TERMS.items():
            if term in tokens:
                for exp in expansions:
                    new_query = query_text.lower().replace(term, exp)
                    if new_query != query_text.lower():
                        fallbacks.append(FallbackQuery(
                            query_text=new_query,
                            strategy="test_term",
                            confidence=0.45,
                            source_terms=[term, exp],
                        ))

        return fallbacks

    def _generate_generalization_fallbacks(
        self,
        query_text: str,
        tokens: Set[str],
    ) -> List[FallbackQuery]:
        """Generate more general versions of the query."""
        fallbacks = []

        # Remove specific terms to get more general results
        specific_terms = {"test", "pytest", "unittest", "junit", "jest", "mocha"}

        # If query has specific test framework, add general version
        if any(term in tokens for term in specific_terms):
            general_query = re.sub(r'\b(test|pytest|unittest|junit|jest|mocha)\b', '', query_text, flags=re.IGNORECASE)
            general_query = ' '.join(general_query.split()).strip()
            if general_query and general_query != query_text.lower():
                fallbacks.append(FallbackQuery(
                    query_text=general_query,
                    strategy="generalization",
                    confidence=0.35,
                    source_terms=list(specific_terms.intersection(tokens)),
                ))

        return fallbacks

    def _generate_specialization_fallbacks(
        self,
        query_text: str,
        query_plan: QueryPlan,
    ) -> List[FallbackQuery]:
        """Generate more specific versions based on query plan."""
        fallbacks = []

        # Add source type context if available
        if query_plan.preferred_source_types:
            for source_type in query_plan.preferred_source_types:
                # Add context based on source type
                if source_type == SourceType.CODE_SNIPPET:
                    fallbacks.append(FallbackQuery(
                        query_text=f"{query_text} code implementation",
                        strategy="specialization",
                        confidence=0.4,
                        source_terms=[source_type.value],
                    ))
                elif source_type == SourceType.REQUIREMENTS:
                    fallbacks.append(FallbackQuery(
                        query_text=f"{query_text} requirement specification",
                        strategy="specialization",
                        confidence=0.4,
                        source_terms=[source_type.value],
                    ))

        # Add modality context
        if query_plan.preferred_modalities:
            for modality in query_plan.preferred_modalities:
                if modality == "table":
                    fallbacks.append(FallbackQuery(
                        query_text=f"{query_text} table data",
                        strategy="specialization",
                        confidence=0.35,
                        source_terms=[modality],
                    ))

        return fallbacks


class SemanticExpander:
    """Expands queries using semantic similarity.

    This provides query expansion based on:
    - Word embeddings similarity (when available)
    - Edit distance for typo correction
    - Domain-specific term relationships
    """

    def __init__(
        self,
        use_edit_distance: bool = True,
        use_domain_terms: bool = True,
        max_expansions: int = 10,
    ) -> None:
        """Initialize the semantic expander.

        Args:
            use_edit_distance: Whether to use edit distance for typo correction
            use_domain_terms: Whether to use domain-specific term expansions
            max_expansions: Maximum number of term expansions
        """
        self._use_edit_distance = use_edit_distance
        self._use_domain_terms = use_domain_terms
        self._max_expansions = max_expansions

    def expand(
        self,
        query_text: str,
        known_terms: Optional[Sequence[str]] = None,
    ) -> ExpansionResult:
        """Expand query with semantically related terms.

        Args:
            query_text: Original query text
            known_terms: Optional list of known terms in the corpus

        Returns:
            ExpansionResult with expanded queries
        """
        tokens = re.findall(r'\b\w+\b', query_text.lower())
        expanded_terms: Set[str] = set(tokens)
        source_terms: List[Tuple[str, str]] = []  # (term, expansion_type)

        # 1. Domain-specific expansions
        if self._use_domain_terms:
            domain_expansions = self._expand_with_domain_terms(tokens)
            expanded_terms.update(domain_expansions)
            source_terms.extend((t, "domain") for t in domain_expansions)

        # 2. Edit distance based expansions (for typo correction)
        if self._use_edit_distance and known_terms:
            edit_expansions = self._expand_with_edit_distance(tokens, known_terms)
            expanded_terms.update(edit_expansions)
            source_terms.extend((t, "edit_distance") for t in edit_expansions)

        # Build combined query
        combined = ' '.join(sorted(expanded_terms))

        # Build individual expansions
        expanded_queries = [
            FallbackQuery(
                query_text=term,
                strategy="semantic_expansion",
                confidence=0.5,
                source_terms=[s for s, _ in source_terms if s in term],
                expansion_type="semantic",
            )
            for term in list(expanded_terms)[:self._max_expansions]
        ]

        # Estimate improvement
        estimated_improvement = min(len(expanded_terms) / max(len(tokens), 1) - 1, 0.5)

        return ExpansionResult(
            original_query=query_text,
            expanded_queries=expanded_queries,
            combined_query=combined,
            expansions_count=len(expanded_terms) - len(tokens),
            estimated_improvement=estimated_improvement,
        )

    def _expand_with_domain_terms(self, tokens: List[str]) -> Set[str]:
        """Expand using domain-specific knowledge."""
        expansions = set()

        for token in tokens:
            # Check against all expansion dictionaries
            for expansion_dict in [_ERROR_SYNONYMS, _TECH_EXPANSIONS, _TEST_FAILURE_TERMS]:
                for key, values in expansion_dict.items():
                    if token in values or token == key:
                        expansions.update(values)

        return expansions

    def _expand_with_edit_distance(
        self,
        tokens: List[str],
        known_terms: Sequence[str],
    ) -> Set[str]:
        """Find similar terms using edit distance."""
        expansions = set()
        known_set = set(k.lower() for k in known_terms)

        for token in tokens:
            if token in known_set:
                continue

            # Find closest matches
            for known in known_set:
                if self._edit_distance(token, known) <= 2:  # Max 2 edits
                    expansions.add(known)

        return expansions

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


class AdaptiveQueryStrategy:
    """Adaptive query strategy that combines multiple approaches.

    This provides a unified interface for query optimization:
    - Initial query attempt
    - Fallback generation when needed
    - Semantic expansion
    - Result fusion
    """

    def __init__(
        self,
        fallback_generator: Optional[QueryFallbackGenerator] = None,
        semantic_expander: Optional[SemanticExpander] = None,
        use_expansion: bool = True,
        fallback_on_low_confidence: bool = True,
    ) -> None:
        """Initialize the adaptive query strategy.

        Args:
            fallback_generator: Query fallback generator
            semantic_expander: Semantic query expander
            use_expansion: Whether to use semantic expansion
            fallback_on_low_confidence: Whether to generate fallbacks on low confidence
        """
        self._fallback_generator = fallback_generator or QueryFallbackGenerator()
        self._semantic_expander = semantic_expander or SemanticExpander()
        self._use_expansion = use_expansion
        self._fallback_on_low_confidence = fallback_on_low_confidence

    def should_fallback(
        self,
        result_count: int,
        top_confidence: float,
    ) -> bool:
        """Check if fallback should be triggered.

        Args:
            result_count: Number of results from initial query
            top_confidence: Confidence of top result

        Returns:
            True if fallback should be used
        """
        return (
            self._fallback_on_low_confidence
            and self._fallback_generator.should_generate_fallbacks(result_count, top_confidence)
        )

    def get_fallback_queries(
        self,
        query_text: str,
        query_plan: Optional[QueryPlan] = None,
    ) -> List[FallbackQuery]:
        """Get fallback queries for the given query.

        Args:
            query_text: Original query text
            query_plan: Optional query plan

        Returns:
            List of fallback queries
        """
        return self._fallback_generator.generate_fallbacks(query_text, query_plan)

    def get_expanded_queries(
        self,
        query_text: str,
        known_terms: Optional[Sequence[str]] = None,
    ) -> ExpansionResult:
        """Get semantically expanded queries.

        Args:
            query_text: Original query text
            known_terms: Optional known terms for edit distance matching

        Returns:
            Expansion result
        """
        if not self._use_expansion:
            return ExpansionResult(
                original_query=query_text,
                expanded_queries=[],
                combined_query=query_text,
                expansions_count=0,
                estimated_improvement=0.0,
            )
        return self._semantic_expander.expand(query_text, known_terms)


# Convenience function for creating adaptive query strategy
def create_adaptive_strategy(
    use_expansion: bool = True,
    use_fallbacks: bool = True,
) -> AdaptiveQueryStrategy:
    """Create an adaptive query strategy with default settings.

    Args:
        use_expansion: Whether to use semantic expansion
        use_fallbacks: Whether to use fallback queries

    Returns:
        Configured AdaptiveQueryStrategy
    """
    return AdaptiveQueryStrategy(
        fallback_generator=QueryFallbackGenerator() if use_fallbacks else None,
        semantic_expander=SemanticExpander() if use_expansion else None,
        use_expansion=use_expansion,
        fallback_on_low_confidence=use_fallbacks,
    )
