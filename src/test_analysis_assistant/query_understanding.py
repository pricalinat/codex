"""Query understanding and intelligent reformulation for improved retrieval.

This module provides:
- Intent classification: Understanding what the user is looking for
- Synonym expansion: Adding related terms for better retrieval
- Query decomposition: Breaking complex queries into simpler components
- Contextual reformulation: Generating context-aware variants

It builds on the QueryReformulator to provide smarter RAG interfaces.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class QueryIntent(str, Enum):
    """Known query intents for test analysis."""

    # Primary intents
    FIND_ROOT_CAUSE = "find_root_cause"
    FIND_RELATED_CODE = "find_related_code"
    FIND_TEST_GAPS = "find_test_gaps"
    FIND_SIMILAR_FAILURES = "find_similar_failures"
    FIND_FIX_SUGGESTION = "find_fix_suggestion"
    FIND_RISK_ASSESSMENT = "find_risk_assessment"
    FIND_REQUIREMENT_TRACE = "find_requirement_trace"
    FIND_DOCUMENTATION = "find_documentation"
    FIND_DEPENDENCY = "find_dependency"

    # Analysis intents
    ANALYZE_PATTERN = "analyze_pattern"
    ANALYZE_TREND = "analyze_trend"
    ANALYZE_COVERAGE = "analyze_coverage"

    # General
    GENERAL = "general"


@dataclass
class IntentClassification:
    """Classification result for a query."""

    primary_intent: QueryIntent
    confidence: float
    secondary_intents: List[QueryIntent] = field(default_factory=list)
    intent_keywords: Dict[QueryIntent, List[str]] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)


@dataclass
class QueryUnderstanding:
    """Complete understanding of a query."""

    original: str
    normalized: str
    tokens: List[str]
    intent: IntentClassification
    expanded_query: str
    decomposed_queries: List[str]
    contextual_variants: List["ContextualVariant"]
    recommended_sources: List[str] = field(default_factory=list)


@dataclass
class ContextualVariant:
    """A contextually reformulated query variant."""

    query: str
    intent: QueryIntent
    weight: float
    rationale: str


# Synonym dictionaries for different domains
_ERROR_SYNONYMS: Dict[str, Set[str]] = {
    "error": {"failure", "exception", "bug", "issue", "problem", "fault"},
    "failure": {"fail", "error", "broken", "defect", "crash"},
    "fix": {"repair", "resolve", "solution", "patch", "correct", "remedy"},
    "test": {"testing", "checks", "validation", "verification"},
    "import": {"include", "require", "load", "module"},
    "assertion": {"assert", "check", "verify", "validate"},
}

_CODE_SYNONYMS: Dict[str, Set[str]] = {
    "function": {"method", "procedure", "routine", "callable"},
    "class": {"object", "type", "entity", "model"},
    "variable": {"var", "parameter", "argument", "property", "attribute"},
    "return": {"output", "result", "value"},
    "parameter": {"arg", "argument", "input", "prop"},
    "file": {"module", "source", "script", "document"},
}

_DOMAIN_SYNONYMS: Dict[str, Set[str]] = {
    "root cause": {"origin", "source", "origin", "reason", "why", "because"},
    "test gap": {"missing test", "untested", "edge case", "boundary", "coverage"},
    "similar failure": {"related", "duplicate", "pattern", "recurring"},
    "risk": {"danger", "threat", "concern", "severity", "critical"},
    "dependency": {"requirement", "prerequisite", "needed", "uses"},
}

# Intent keyword mappings
_INTENT_KEYWORDS: Dict[QueryIntent, Set[str]] = {
    QueryIntent.FIND_ROOT_CAUSE: {
        "root", "cause", "why", "reason", "origin", "source",
        "because", "originated", "triggered", "started",
    },
    QueryIntent.FIND_RELATED_CODE: {
        "related", "similar", "near", "around", "around", "neighbor",
        "connected", "associated", "linked", "coupled",
    },
    QueryIntent.FIND_TEST_GAPS: {
        "gap", "missing", "coverage", "untested", "edge", "boundary",
        "negative", "corner", "case", "not tested", "lacking",
    },
    QueryIntent.FIND_SIMILAR_FAILURES: {
        "similar", "like", "duplicate", "same", "pattern", "recurring",
        "repeated", "again", "another", "also", "related failure",
    },
    QueryIntent.FIND_FIX_SUGGESTION: {
        "fix", "solution", "resolve", "how to", "help", "suggestion",
        "repair", "correct", "patch", "workaround", "answer",
    },
    QueryIntent.FIND_RISK_ASSESSMENT: {
        "risk", "severity", "critical", "important", "serious",
        "danger", "threat", "blocking", "urgent", "priority",
    },
    QueryIntent.FIND_REQUIREMENT_TRACE: {
        "requirement", "spec", "specification", "trace", "mapped",
        "linked to", "derived from", "corresponds",
    },
    QueryIntent.FIND_DOCUMENTATION: {
        "doc", "document", "readme", "guide", "manual", "help",
        "description", "explain", "reference", "api",
    },
    QueryIntent.FIND_DEPENDENCY: {
        "dependency", "depends", "import", "require", "needed",
        "uses", "used by", "requires", "prerequisite",
    },
    QueryIntent.ANALYZE_PATTERN: {
        "pattern", "trend", "common", "frequent", "recurring",
        "repeated", "usual", "typically", "often",
    },
    QueryIntent.ANALYZE_TREND: {
        "trend", "over time", "increasing", "decreasing", "growing",
        "changing", "evolving", "history", "past",
    },
    QueryIntent.ANALYZE_COVERAGE: {
        "coverage", "covered", "tested", "untested", "percent",
        "percentage", "how much", "scope",
    },
}


class QueryUnderstandingEngine:
    """Engine for understanding and reformulating queries.

    This engine provides:
    1. Intent classification - understanding what the user wants
    2. Synonym expansion - adding related terms
    3. Query decomposition - breaking complex queries
    4. Contextual variants - generating context-aware variants
    """

    def __init__(
        self,
        enable_synonym_expansion: bool = True,
        enable_decomposition: bool = True,
        max_variants: int = 5,
    ) -> None:
        """Initialize the query understanding engine.

        Args:
            enable_synonym_expansion: Whether to expand with synonyms
            enable_decomposition: Whether to decompose complex queries
            max_variants: Maximum number of contextual variants
        """
        self._enable_synonym_expansion = enable_synonym_expansion
        self._enable_decomposition = enable_decomposition
        self._max_variants = max_variants

    def understand(self, query: str) -> QueryUnderstanding:
        """Understand a query and generate reformulations.

        Args:
            query: The user's query string

        Returns:
            QueryUnderstanding with full analysis
        """
        # Normalize query
        normalized = self._normalize_query(query)
        tokens = self._tokenize(normalized)

        # Classify intent
        intent = self._classify_intent(query, tokens)

        # Expand with synonyms
        expanded = self._expand_with_synonyms(tokens, intent)

        # Decompose if complex
        decomposed = self._decompose_query(query, tokens) if self._enable_decomposition else [query]

        # Generate contextual variants
        variants = self._generate_contextual_variants(query, intent, tokens)

        # Determine recommended sources
        recommended = self._get_recommended_sources(intent)

        return QueryUnderstanding(
            original=query,
            normalized=normalized,
            tokens=tokens,
            intent=intent,
            expanded_query=expanded,
            decomposed_queries=decomposed,
            contextual_variants=variants,
            recommended_sources=recommended,
        )

    def _normalize_query(self, query: str) -> str:
        """Normalize query text.

        Args:
            query: Raw query

        Returns:
            Normalized query
        """
        # Lowercase
        normalized = query.lower()
        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()
        # Remove special chars except spaces and alphanum
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        return normalized

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        return text.split()

    def _classify_intent(self, query: str, tokens: List[str]) -> IntentClassification:
        """Classify the intent of a query.

        Args:
            query: Original query
            tokens: Query tokens

        Returns:
            IntentClassification result
        """
        query_lower = query.lower()
        intent_scores: Dict[QueryIntent, float] = {}

        # Score each intent based on keyword matches
        for intent, keywords in _INTENT_KEYWORDS.items():
            score = 0.0
            matched_keywords: List[str] = []

            for keyword in keywords:
                if keyword in query_lower:
                    # Count occurrences
                    count = query_lower.count(keyword)
                    score += count
                    matched_keywords.append(keyword)

            if matched_keywords:
                intent_scores[intent] = score

        # Sort by score
        if intent_scores:
            sorted_intents = sorted(
                intent_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            primary = sorted_intents[0][0]
            confidence = min(1.0, sorted_intents[0][1] / 3.0)  # Normalize
            secondary = [i for i, _ in sorted_intents[1:4]]
            keyword_map = {i: matched_keywords for i, matched_keywords in [
                (i, [k for k in _INTENT_KEYWORDS[i] if k in query_lower])
                for i in [primary] + secondary
            ] if matched_keywords}
        else:
            primary = QueryIntent.GENERAL
            confidence = 0.5
            secondary = []
            keyword_map = {}

        # Extract entities (potential code artifacts)
        entities = self._extract_entities(query)

        return IntentClassification(
            primary_intent=primary,
            confidence=confidence,
            secondary_intents=secondary[:3],
            intent_keywords=keyword_map,
            entities=entities,
        )

    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query.

        Args:
            query: Query text

        Returns:
            List of extracted entities
        """
        entities: List[str] = []

        # Extract camelCase and PascalCase
        entities.extend(re.findall(r"[a-z]+[A-Z][a-zA-Z]*", query))

        # Extract snake_case
        entities.extend(re.findall(r"[a-z]+_[a-z]+", query.lower()))

        # Extract potential file paths
        entities.extend(re.findall(r"[\w/]+\.py", query))

        # Extract potential function names (followed by parenthesis)
        entities.extend(re.findall(r"\b(\w+)\s*\(", query))

        return list(set(entities))[:5]

    def _expand_with_synonyms(
        self,
        tokens: List[str],
        intent: IntentClassification,
    ) -> str:
        """Expand query with synonyms.

        Args:
            tokens: Query tokens
            intent: Classified intent

        Returns:
            Expanded query string
        """
        if not self._enable_synonym_expansion:
            return " ".join(tokens)

        expanded: Set[str] = set(tokens)

        # Add synonyms for each token
        for token in tokens:
            # Check error synonyms
            if token in _ERROR_SYNONYMS:
                expanded.update(_ERROR_SYNONYMS[token])
            # Check code synonyms
            if token in _CODE_SYNONYMS:
                expanded.update(_CODE_SYNONYMS[token])
            # Check domain synonyms
            for domain_syns in _DOMAIN_SYNONYMS.values():
                if token in domain_syns:
                    expanded.update(domain_syns)

        # Add intent-specific terms
        intent_terms = _INTENT_KEYWORDS.get(intent.primary_intent, set())
        expanded.update(intent_terms)

        return " ".join(sorted(expanded))

    def _decompose_query(self, query: str, tokens: List[str]) -> List[str]:
        """Decompose complex query into sub-queries.

        Args:
            query: Original query
            tokens: Query tokens

        Returns:
            List of sub-queries
        """
        sub_queries = [query]

        # Look for conjunctions that suggest decomposition
        if " and " in query.lower():
            parts = re.split(r"\s+and\s+", query.lower(), maxsplit=2)
            sub_queries.extend(parts[1:3])  # Add up to 2 sub-parts

        if " or " in query.lower():
            parts = re.split(r"\s+or\s+", query.lower(), maxsplit=2)
            sub_queries.extend(parts[1:3])

        # Look for question patterns
        question_patterns = [
            (r"how\s+(to|do|does|did)\s+(\w+)", r"how to \2"),
            (r"why\s+(is|are|does|do)\s+(\w+)", r"why \2"),
            (r"what\s+(is|are|does|do)\s+(\w+)", r"what \2"),
        ]

        for pattern, replacement in question_patterns:
            match = re.search(pattern, query.lower())
            if match:
                sub_queries.append(replacement)

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in sub_queries:
            q_normalized = q.lower().strip()
            if q_normalized and q_normalized not in seen:
                seen.add(q_normalized)
                unique_queries.append(q)

        return unique_queries[:4]  # Limit to 4 sub-queries

    def _generate_contextual_variants(
        self,
        query: str,
        intent: IntentClassification,
        tokens: List[str],
    ) -> List[ContextualVariant]:
        """Generate contextual query variants.

        Args:
            query: Original query
            intent: Classified intent
            tokens: Query tokens

        Returns:
            List of contextual variants
        """
        variants: List[ContextualVariant] = []

        # Always include original
        variants.append(ContextualVariant(
            query=query,
            intent=intent.primary_intent,
            weight=1.0,
            rationale="Original query",
        ))

        # Generate intent-specific variants
        variant_templates = self._get_variant_templates(intent.primary_intent)

        for i, template in enumerate(variant_templates[:self._max_variants - 1]):
            # Fill in tokens if template has placeholders
            if "{0}" in template and tokens:
                filled = template.format(tokens[0])
            else:
                filled = template

            variants.append(ContextualVariant(
                query=filled,
                intent=intent.primary_intent,
                weight=0.9 - (i * 0.1),
                rationale=f"Intent-specific variant for {intent.primary_intent.value}",
            ))

        # Add some secondary intent variants
        for secondary in intent.secondary_intents[:2]:
            if len(variants) >= self._max_variants:
                break

            templates = self._get_variant_templates(secondary)
            if templates:
                template = templates[0]
                if "{0}" in template and tokens:
                    filled = template.format(tokens[0])
                else:
                    filled = template

                variants.append(ContextualVariant(
                    query=filled,
                    intent=secondary,
                    weight=0.6,
                    rationale=f"Secondary intent variant for {secondary.value}",
                ))

        return variants[:self._max_variants]

    def _get_variant_templates(self, intent: QueryIntent) -> List[str]:
        """Get query variant templates for an intent.

        Args:
            intent: Query intent

        Returns:
            List of template strings
        """
        templates: Dict[QueryIntent, List[str]] = {
            QueryIntent.FIND_ROOT_CAUSE: [
                "root cause of {0} failure",
                "why {0} is failing",
                "origin of {0} error",
            ],
            QueryIntent.FIND_RELATED_CODE: [
                "code related to {0}",
                "functions similar to {0}",
                "implementation of {0}",
            ],
            QueryIntent.FIND_TEST_GAPS: [
                "missing tests for {0}",
                "untested edge cases for {0}",
                "test coverage gaps in {0}",
            ],
            QueryIntent.FIND_SIMILAR_FAILURES: [
                "similar failures to {0}",
                "duplicate issues with {0}",
                "pattern of {0} failures",
            ],
            QueryIntent.FIND_FIX_SUGGESTION: [
                "how to fix {0}",
                "solution for {0}",
                "workaround for {0}",
            ],
            QueryIntent.FIND_RISK_ASSESSMENT: [
                "risk level of {0}",
                "severity of {0} issue",
                "impact of {0} failure",
            ],
            QueryIntent.FIND_REQUIREMENT_TRACE: [
                "requirements for {0}",
                "specification of {0}",
                "tests for {0} requirement",
            ],
            QueryIntent.FIND_DOCUMENTATION: [
                "documentation for {0}",
                "docs about {0}",
                "how to use {0}",
            ],
            QueryIntent.FIND_DEPENDENCY: [
                "dependencies of {0}",
                "requires {0}",
                "imports needed for {0}",
            ],
            QueryIntent.ANALYZE_PATTERN: [
                "pattern in {0} failures",
                "common issues with {0}",
            ],
            QueryIntent.ANALYZE_TREND: [
                "trend of {0} over time",
                "history of {0}",
            ],
            QueryIntent.ANALYZE_COVERAGE: [
                "test coverage for {0}",
                "what is tested in {0}",
            ],
            QueryIntent.GENERAL: [
                "information about {0}",
                "details of {0}",
            ],
        }

        return templates.get(intent, [])

    def _get_recommended_sources(self, intent: IntentClassification) -> List[str]:
        """Get recommended source types for an intent.

        Args:
            intent: Classified intent

        Returns:
            List of recommended source type names
        """
        source_mapping: Dict[QueryIntent, List[str]] = {
            QueryIntent.FIND_ROOT_CAUSE: ["repository", "code_snippet"],
            QueryIntent.FIND_RELATED_CODE: ["repository", "code_snippet"],
            QueryIntent.FIND_TEST_GAPS: ["requirements", "system_analysis"],
            QueryIntent.FIND_SIMILAR_FAILURES: ["knowledge"],
            QueryIntent.FIND_FIX_SUGGESTION: ["knowledge", "code_snippet"],
            QueryIntent.FIND_RISK_ASSESSMENT: ["requirements", "system_analysis"],
            QueryIntent.FIND_REQUIREMENT_TRACE: ["requirements"],
            QueryIntent.FIND_DOCUMENTATION: ["knowledge"],
            QueryIntent.FIND_DEPENDENCY: ["repository", "code_snippet"],
            QueryIntent.ANALYZE_PATTERN: ["knowledge", "system_analysis"],
            QueryIntent.ANALYZE_TREND: ["knowledge"],
            QueryIntent.ANALYZE_COVERAGE: ["requirements", "system_analysis"],
            QueryIntent.GENERAL: ["knowledge", "code_snippet", "requirements"],
        }

        return source_mapping.get(intent.primary_intent, ["knowledge"])


def understand_query(query: str) -> QueryUnderstanding:
    """Convenience function to understand a query.

    Args:
        query: The user's query string

    Returns:
        QueryUnderstanding with full analysis
    """
    engine = QueryUnderstandingEngine()
    return engine.understand(query)
