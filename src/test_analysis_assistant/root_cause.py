"""Root cause hypothesis generation with confidence scoring.

This module provides sophisticated failure pattern analysis and hypothesis
generation for test failures, with confidence scoring based on evidence
and retrieval context.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .models import AnalysisResult, FailureCluster, FailureRecord


class RootCauseCategory(str, Enum):
    """Categories of root causes for test failures."""

    CODE_DEFECT = "code_defect"  # Actual bug in implementation
    TEST_DEFECT = "test_defect"  # Bug in the test itself
    ENVIRONMENT = "environment"  # Environment/configuration issue
    DEPENDENCY = "dependency"  # Missing or incompatible dependency
    DATA_ISSUE = "data_issue"  # Test data problem
    TIMING_ISSUE = "timing_issue"  # Race conditions, timeouts
    RESOURCE_ISSUE = "resource_issue"  # Memory, CPU, disk issues
    FLAKY_TEST = "flaky_test"  # Non-deterministic test behavior
    UNKNOWN = "unknown"  # Unable to determine


class HypothesisConfidence(str, Enum):
    """Confidence levels for root cause hypotheses."""

    HIGH = "high"  # Strong evidence from multiple sources
    MEDIUM = "medium"  # Moderate evidence
    LOW = "low"  # Weak or circumstantial evidence


@dataclass
class RootCauseHypothesis:
    """A hypothesis about the root cause of test failures."""

    hypothesis_id: str
    category: RootCauseCategory
    description: str
    confidence: float  # 0-1
    confidence_level: HypothesisConfidence
    evidence: List[str] = field(default_factory=list)
    supporting_failures: List[str] = field(default_factory=list)
    related_code_locations: List[str] = field(default_factory=list)
    remediation_suggestions: List[str] = field(default_factory=list)
    priority: int = 0  # 0 = highest priority


@dataclass
class RootCauseAnalysis:
    """Complete root cause analysis result."""

    analysis_id: str
    total_failures_analyzed: int
    hypotheses: List[RootCauseHypothesis] = field(default_factory=list)
    primary_hypothesis: Optional[RootCauseHypothesis] = None
    overall_confidence: float = 0.0
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    analyzed_at: str = ""


class RootCauseAnalyzer:
    """Analyzer for generating root cause hypotheses from test failures."""

    # Error type to category mappings with weights
    ERROR_CATEGORY_PATTERNS = {
        "ModuleNotFoundError": (RootCauseCategory.DEPENDENCY, 0.9),
        "ImportError": (RootCauseCategory.DEPENDENCY, 0.85),
        "SyntaxError": (RootCauseCategory.CODE_DEFECT, 0.95),
        "IndentationError": (RootCauseCategory.CODE_DEFECT, 0.9),
        "TypeError": (RootCauseCategory.CODE_DEFECT, 0.7),
        "AttributeError": (RootCauseCategory.CODE_DEFECT, 0.75),
        "ValueError": (RootCauseCategory.CODE_DEFECT, 0.7),
        "KeyError": (RootCauseCategory.CODE_DEFECT, 0.7),
        "IndexError": (RootCauseCategory.CODE_DEFECT, 0.7),
        "AssertionError": (RootCauseCategory.CODE_DEFECT, 0.6),
        "RuntimeError": (RootCauseCategory.UNKNOWN, 0.4),
        "TimeoutError": (RootCauseCategory.TIMING_ISSUE, 0.8),
        "ConnectionError": (RootCauseCategory.ENVIRONMENT, 0.7),
        "PermissionError": (RootCauseCategory.ENVIRONMENT, 0.8),
        "FileNotFoundError": (RootCauseCategory.ENVIRONMENT, 0.7),
        "MemoryError": (RootCauseCategory.RESOURCE_ISSUE, 0.9),
        "OSError": (RootCauseCategory.ENVIRONMENT, 0.5),
    }

    # Code patterns suggesting test defects
    TEST_DEFECT_PATTERNS = [
        (r"assert.*True\s*==\s*False", "Possibly inverted assertion"),
        (r"assert.*None\s*is\s*not", "Potentially incorrect assertion"),
        (r"assert.*\[.*\]\s*==", "List comparison may miss ordering"),
        (r"mock.*\.return_value\s*=.*None", "Mock may be returning None incorrectly"),
    ]

    # Flaky test indicators
    FLAKY_PATTERNS = [
        (r"time\.sleep", "Timing-dependent test"),
        (r"random", "Non-deterministic behavior"),
        (r"thread|Thread|asyncio", "Concurrent execution possible race"),
        (r"network|http|request", "Network-dependent test"),
    ]

    def __init__(self) -> None:
        self._hypothesis_counter = 0

    def analyze(
        self,
        analysis_result: AnalysisResult,
        context_chunks: Optional[Sequence[str]] = None,
    ) -> RootCauseAnalysis:
        """Analyze test failures and generate root cause hypotheses.

        Args:
            analysis_result: The base analysis result from test report
            context_chunks: Optional context from retrieval for evidence

        Returns:
            RootCauseAnalysis with hypotheses and confidence scores
        """
        self._hypothesis_counter = 0

        # Analyze each failure cluster
        hypotheses: List[RootCauseHypothesis] = []
        all_evidence: Dict[str, Any] = {"by_category": {}, "total_evidence": 0}

        for cluster in analysis_result.clusters:
            hypothesis = self._analyze_cluster(
                cluster,
                analysis_result.failures,
                context_chunks,
            )
            if hypothesis:
                hypotheses.append(hypothesis)

                # Track evidence
                cat = hypothesis.category.value
                all_evidence["by_category"][cat] = all_evidence["by_category"].get(cat, 0) + 1

        # Add hypotheses for individual high-frequency errors
        error_counts = self._count_error_types(analysis_result.failures)
        for error_type, count in error_counts.items():
            if count >= 3 and not any(h.category != RootCauseCategory.UNKNOWN for h in hypotheses):
                hypothesis = self._create_error_type_hypothesis(error_type, count)
                hypotheses.append(hypothesis)

        # Sort by priority and confidence
        hypotheses.sort(key=lambda h: (h.priority, -h.confidence))

        # Determine primary hypothesis
        primary = hypotheses[0] if hypotheses else None

        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(hypotheses)

        return RootCauseAnalysis(
            analysis_id=f"rca-{hash(str(analysis_result.failures)) % 100000:05d}",
            total_failures_analyzed=analysis_result.total_failures,
            hypotheses=hypotheses,
            primary_hypothesis=primary,
            overall_confidence=overall_confidence,
            evidence_summary=all_evidence,
        )

    def _analyze_cluster(
        self,
        cluster: FailureCluster,
        all_failures: List[FailureRecord],
        context_chunks: Optional[Sequence[str]],
    ) -> Optional[RootCauseHypothesis]:
        """Analyze a single failure cluster to generate hypothesis."""
        error_type = cluster.error_type
        category, base_confidence = self.ERROR_CATEGORY_PATTERNS.get(
            error_type,
            (RootCauseCategory.UNKNOWN, 0.3)
        )

        # Adjust confidence based on cluster size
        cluster_factor = min(1.0, cluster.count / 5.0)
        confidence = base_confidence * 0.6 + cluster_factor * 0.4

        # Check for test defect patterns in test names/messages
        for failure in all_failures:
            if failure.test_name in cluster.tests:
                for pattern, description in self.TEST_DEFECT_PATTERNS:
                    if re.search(pattern, failure.message, re.IGNORECASE):
                        category = RootCauseCategory.TEST_DEFECT
                        confidence = min(0.9, confidence + 0.2)
                        break

        # Build evidence list
        evidence = []
        evidence.append(f"Found {cluster.count} failures of type {error_type}")
        evidence.append(f"Error category: {category.value}")

        if context_chunks:
            for chunk in context_chunks[:3]:
                if error_type.lower() in chunk.lower() or category.value in chunk.lower():
                    evidence.append(f"Context evidence: {chunk[:80]}...")

        # Get related failures
        related_failures = cluster.tests[:3]

        # Generate remediation suggestions
        suggestions = self._generate_remediation(category, error_type, cluster.count)

        return RootCauseHypothesis(
            hypothesis_id=f"H{self._hypothesis_counter:03d}",
            category=category,
            description=self._build_description(category, error_type, cluster),
            confidence=round(confidence, 3),
            confidence_level=self._get_confidence_level(confidence),
            evidence=evidence,
            supporting_failures=related_failures,
            remediation_suggestions=suggestions,
            priority=self._get_priority(category, cluster.count),
        )

    def _create_error_type_hypothesis(
        self,
        error_type: str,
        count: int,
    ) -> RootCauseHypothesis:
        """Create a hypothesis for a frequent error type."""
        self._hypothesis_counter += 1
        category, base_confidence = self.ERROR_CATEGORY_PATTERNS.get(
            error_type,
            (RootCauseCategory.UNKNOWN, 0.3)
        )

        return RootCauseHypothesis(
            hypothesis_id=f"H{self._hypothesis_counter:03d}",
            category=category,
            description=f"Frequent {error_type} occurrences suggest systemic issue",
            confidence=round(base_confidence * 0.8, 3),
            confidence_level=self._get_confidence_level(base_confidence * 0.8),
            evidence=[f"Error type {error_type} occurred {count} times"],
            supporting_failures=[],
            remediation_suggestions=self._generate_remediation(category, error_type, count),
            priority=self._get_priority(category, count),
        )

    def _count_error_types(
        self,
        failures: List[FailureRecord],
    ) -> Dict[str, int]:
        """Count occurrences of each error type."""
        counts: Dict[str, int] = {}
        for failure in failures:
            counts[failure.error_type] = counts.get(failure.error_type, 0) + 1
        return counts

    def _build_description(
        self,
        category: RootCauseCategory,
        error_type: str,
        cluster: FailureCluster,
    ) -> str:
        """Build a human-readable description for the hypothesis."""
        if category == RootCauseCategory.CODE_DEFECT:
            return (
                f"Code defect causing {error_type}: {cluster.reason}. "
                f"Affects {cluster.count} test(s)."
            )
        elif category == RootCauseCategory.DEPENDENCY:
            return (
                f"Missing or incompatible dependency causing {error_type}. "
                f"Review import statements and requirements."
            )
        elif category == RootCauseCategory.ENVIRONMENT:
            return (
                f"Environment configuration issue causing {error_type}. "
                f"Check test environment setup."
            )
        elif category == RootCauseCategory.TEST_DEFECT:
            return (
                f"Test itself appears defective ({error_type}). "
                f"Review test assertion logic."
            )
        elif category == RootCauseCategory.TIMING_ISSUE:
            return (
                f"Timing or race condition causing {error_type}. "
                f"Consider async handling or timeouts."
            )
        elif category == RootCauseCategory.RESOURCE_ISSUE:
            return (
                f"Resource constraint causing {error_type}. "
                f"Check memory/disk/CPU availability."
            )
        else:
            return f"Unknown root cause for {error_type}: {cluster.reason}"

    def _generate_remediation(
        self,
        category: RootCauseCategory,
        error_type: str,
        count: int,
    ) -> List[str]:
        """Generate remediation suggestions based on category."""
        suggestions = []

        if category == RootCauseCategory.CODE_DEFECT:
            suggestions.append(f"Fix the underlying code causing {error_type}")
            suggestions.append("Review recent code changes in affected modules")
            if count > 1:
                suggestions.append("Check for shared utilities or base classes")

        elif category == RootCauseCategory.DEPENDENCY:
            suggestions.append("Verify all dependencies are installed")
            suggestions.append("Check requirements.txt or pyproject.toml")
            suggestions.append("Ensure compatible versions")

        elif category == RootCauseCategory.ENVIRONMENT:
            suggestions.append("Verify environment configuration")
            suggestions.append("Check file paths and permissions")
            suggestions.append("Review environment variables")

        elif category == RootCauseCategory.TEST_DEFECT:
            suggestions.append("Review test assertion logic")
            suggestions.append("Check for hardcoded values")
            suggestions.append("Verify test setup/teardown")

        elif category == RootCauseCategory.TIMING_ISSUE:
            suggestions.append("Add or increase timeouts")
            suggestions.append("Use synchronization primitives")
            suggestions.append("Consider retry mechanisms")

        elif category == RootCauseCategory.RESOURCE_ISSUE:
            suggestions.append("Increase resource limits")
            suggestions.append("Optimize resource usage")
            suggestions.append("Run resource monitoring")

        return suggestions[:3]  # Limit to top 3

    def _get_confidence_level(self, confidence: float) -> HypothesisConfidence:
        """Convert numeric confidence to level."""
        if confidence >= 0.7:
            return HypothesisConfidence.HIGH
        elif confidence >= 0.4:
            return HypothesisConfidence.MEDIUM
        else:
            return HypothesisConfidence.LOW

    def _get_priority(self, category: RootCauseCategory, count: int) -> int:
        """Determine priority based on category and count."""
        category_priority = {
            RootCauseCategory.CODE_DEFECT: 1,
            RootCauseCategory.DEPENDENCY: 2,
            RootCauseCategory.ENVIRONMENT: 3,
            RootCauseCategory.TEST_DEFECT: 4,
            RootCauseCategory.TIMING_ISSUE: 5,
            RootCauseCategory.RESOURCE_ISSUE: 6,
            RootCauseCategory.FLAKY_TEST: 7,
            RootCauseCategory.DATA_ISSUE: 8,
            RootCauseCategory.UNKNOWN: 9,
        }
        base = category_priority.get(category, 9)
        # Increase priority for more frequent failures
        return base - min(2, count // 3)

    def _calculate_overall_confidence(
        self,
        hypotheses: List[RootCauseHypothesis],
    ) -> float:
        """Calculate overall analysis confidence."""
        if not hypotheses:
            return 0.0

        # Weight by confidence and priority
        total_weight = 0.0
        weighted_sum = 0.0

        for h in hypotheses:
            weight = 1.0 / (h.priority + 1)
            weighted_sum += h.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return round(weighted_sum / total_weight, 3)


def generate_root_cause_hypotheses(
    analysis_result: AnalysisResult,
    context_chunks: Optional[Sequence[str]] = None,
) -> RootCauseAnalysis:
    """Convenience function to generate root cause hypotheses.

    Args:
        analysis_result: The base analysis result from test report
        context_chunks: Optional context from retrieval for evidence

    Returns:
        RootCauseAnalysis with hypotheses and confidence scores
    """
    analyzer = RootCauseAnalyzer()
    return analyzer.analyze(analysis_result, context_chunks)
