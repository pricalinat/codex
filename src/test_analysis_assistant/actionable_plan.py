"""Actionable plan generation with confidence scoring for test analysis."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from .models import AnalysisResult
from .retrieval import RankedChunk

if TYPE_CHECKING:
    from .rag_analyzer import (
        RAGAnalysisResult,
        RetrievalInsight,
        RequirementTrace,
        TestGapAnalysis,
    )


@dataclass
class ActionableStep:
    """A single step in an actionable plan."""

    step_id: str
    description: str
    priority: str  # P0, P1, P2, P3
    estimated_effort: str  # low, medium, high
    confidence: float
    related_failures: List[str] = field(default_factory=list)
    related_gaps: List[str] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)


@dataclass
class ActionablePlan:
    """Complete actionable plan with confidence scoring."""

    title: str
    summary: str
    steps: List[ActionableStep] = field(default_factory=list)
    overall_confidence: float = 0.0
    risk_level: str = "low"  # low, medium, high, critical
    missing_evidence: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


def generate_actionable_plan(
    analysis_result: RAGAnalysisResult,
    ranked_context: Optional[Sequence[RankedChunk]] = None,
) -> ActionablePlan:
    """Generate an actionable plan from RAG analysis results.

    Args:
        analysis_result: The RAG analysis result
        ranked_context: Optional retrieved context for evidence

    Returns:
        ActionablePlan with structured steps and confidence scoring
    """
    steps: List[ActionableStep] = []
    step_id_counter = 1

    # Generate steps from clusters
    for cluster in analysis_result.base_result.clusters:
        step = _cluster_to_step(cluster, step_id_counter)
        steps.append(step)
        step_id_counter += 1

    # Generate steps from test gaps
    for gap in analysis_result.structured_gaps[:5]:  # Limit to top 5 gaps
        step = _gap_to_step(gap, step_id_counter)
        steps.append(step)
        step_id_counter += 1

    # Generate steps from high-confidence insights
    for insight in analysis_result.retrieval_insights:
        if insight.confidence >= 0.5 and insight.insight_type in ("risk_factor", "root_cause_evidence"):
            step = _insight_to_step(insight, step_id_counter)
            steps.append(step)
            step_id_counter += 1
            if step_id_counter > 15:  # Limit total steps
                break

    # Sort by priority
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    steps.sort(key=lambda s: (priority_order.get(s.priority, 99), s.step_id))

    # Calculate overall confidence
    confidence = _calculate_plan_confidence(analysis_result, steps)

    # Determine risk level
    risk_level = _determine_risk_level(analysis_result.risk_assessment, steps)

    # Collect missing evidence
    missing_evidence = _identify_missing_evidence(analysis_result, ranked_context)

    # Build prerequisites
    prerequisites = _extract_prerequisites(steps)

    return ActionablePlan(
        title=f"Test Remediation Plan ({analysis_result.base_result.total_failures} failures)",
        summary=_build_summary(analysis_result, steps),
        steps=steps,
        overall_confidence=confidence,
        risk_level=risk_level,
        missing_evidence=missing_evidence,
        prerequisites=prerequisites,
    )


def _cluster_to_step(cluster, step_num: int) -> ActionableStep:
    """Convert a failure cluster to an actionable step."""
    priority_map = {
        "ModuleNotFoundError": "P0",
        "ImportError": "P0",
        "SyntaxError": "P0",
        "TypeError": "P1",
        "TimeoutError": "P1",
        "RuntimeError": "P1",
        "AssertionError": "P2",
    }

    effort_map = {
        "ModuleNotFoundError": "low",
        "ImportError": "low",
        "SyntaxError": "low",
        "TypeError": "medium",
        "TimeoutError": "high",
        "RuntimeError": "medium",
        "AssertionError": "medium",
    }

    return ActionableStep(
        step_id=f"S{step_num:02d}",
        description=f"Fix {cluster.error_type} in {cluster.count} test(s): {cluster.reason}",
        priority=priority_map.get(cluster.error_type, "P2"),
        estimated_effort=effort_map.get(cluster.error_type, "medium"),
        confidence=0.85,
        related_failures=cluster.tests[:3],
    )


def _gap_to_step(gap: TestGapAnalysis, step_num: int) -> ActionableStep:
    """Convert a test gap to an actionable step."""
    return ActionableStep(
        step_id=f"S{step_num:02d}",
        description=f"Address test gap ({gap.gap_type}): {gap.description[:80]}",
        priority=_gap_priority_to_step_priority(gap.severity),
        estimated_effort="medium",
        confidence=gap.confidence,
        related_gaps=[gap.gap_id],
        evidence_sources=gap.related_requirements,
    )


def _insight_to_step(insight: RetrievalInsight, step_num: int) -> ActionableStep:
    """Convert a retrieval insight to an actionable step."""
    return ActionableStep(
        step_id=f"S{step_num:02d}",
        description=f"Investigate: {insight.title} - {insight.description[:60]}",
        priority=_insight_priority_to_step_priority(insight.severity),
        estimated_effort="medium",
        confidence=insight.confidence,
        evidence_sources=insight.evidence_chunks,
    )


def _gap_priority_to_step_priority(severity: str) -> str:
    mapping = {"critical": "P0", "high": "P1", "medium": "P2", "low": "P3"}
    return mapping.get(severity, "P2")


def _insight_priority_to_step_priority(severity: str) -> str:
    mapping = {"critical": "P0", "high": "P1", "medium": "P2", "low": "P3"}
    return mapping.get(severity, "P2")


def _calculate_plan_confidence(result: RAGAnalysisResult, steps: List[ActionableStep]) -> float:
    """Calculate overall confidence for the plan."""
    if not steps:
        return 0.0

    # Base confidence from retrieval insights
    insight_confidences = [i.confidence for i in result.retrieval_insights]
    avg_insight_conf = sum(insight_confidences) / len(insight_confidences) if insight_confidences else 0.3

    # Step confidence
    step_confidences = [s.confidence for s in steps]
    avg_step_conf = sum(step_confidences) / len(step_confidences)

    # Coverage factor (how many gaps addressed)
    coverage_factor = min(1.0, len(steps) / 10.0)

    # Evidence factor
    evidence_count = len(result.evidence_sources)
    evidence_factor = min(1.0, evidence_count / 5.0)

    overall = (avg_insight_conf * 0.25) + (avg_step_conf * 0.35) + (coverage_factor * 0.20) + (evidence_factor * 0.20)
    return round(max(0.0, min(1.0, overall)), 2)


def _determine_risk_level(risk_assessment: Dict[str, Any], steps: List[ActionableStep]) -> str:
    """Determine overall risk level for the plan."""
    # Check existing risk assessment
    if "overall_risk" in risk_assessment:
        return risk_assessment["overall_risk"]

    # Check for P0 steps
    p0_count = sum(1 for s in steps if s.priority == "P0")
    if p0_count >= 3:
        return "critical"
    if p0_count >= 1:
        return "high"

    # Check for high-effort steps
    high_effort = sum(1 for s in steps if s.estimated_effort == "high")
    if high_effort >= 3:
        return "high"

    return "medium"


def _identify_missing_evidence(
    result: RAGAnalysisResult,
    ranked_context: Optional[Sequence[RankedChunk]],
) -> List[str]:
    """Identify what evidence is missing for higher confidence."""
    missing: List[str] = []

    if not result.requirement_traces:
        missing.append("Requirements documentation for traceability")

    if len(result.structured_gaps) == 0:
        missing.append("Structured test gap analysis")

    if not result.retrieval_insights:
        missing.append("Retrieval-augmented insights (needs corpus)")

    if ranked_context and len(ranked_context) < 3:
        missing.append("More contextual documents for retrieval")

    if result.base_result.total_failures > 10 and len(result.base_result.clusters) < 2:
        missing.append("Failure clustering details")

    return missing


def _extract_prerequisites(steps: List[ActionableStep]) -> List[str]:
    """Extract prerequisites from steps."""
    prereqs: List[str] = []

    # Check for dependency-related failures
    for step in steps:
        if "ModuleNotFoundError" in step.description or "ImportError" in step.description:
            if "Install dependencies" not in prereqs:
                prereqs.append("Install missing dependencies")
            break

    # Check for environment-related issues
    for step in steps:
        if "TimeoutError" in step.description:
            if "Check environment/configuration" not in prereqs:
                prereqs.append("Check environment/configuration")
            break

    return prereqs


def _build_summary(result: RAGAnalysisResult, steps: List[ActionableStep]) -> str:
    """Build a summary for the plan."""
    total = result.base_result.total_failures
    clusters = len(result.base_result.clusters)
    gaps = len(result.structured_gaps)

    summary_parts = [
        f"Found {total} test failures across {clusters} clusters.",
    ]

    if gaps > 0:
        summary_parts.append(f"Identified {gaps} test gaps.")

    p0_count = sum(1 for s in steps if s.priority == "P0")
    if p0_count > 0:
        summary_parts.append(f"{p0_count} P0 (critical) steps required.")

    return " ".join(summary_parts)


def build_plan_prompt(plan: ActionablePlan, question: str = "") -> str:
    """Build an LLM prompt from an actionable plan.

    Args:
        plan: The actionable plan
        question: Optional specific question for the LLM

    Returns:
        Formatted prompt string
    """
    lines = [
        "You are a test analysis assistant. Use the actionable plan below to respond to the user's question.",
        f"Question: {question or 'Generate a detailed remediation plan for the test failures.'}",
        "",
        f"Plan: {plan.title}",
        f"Summary: {plan.summary}",
        f"Overall Confidence: {plan.overall_confidence:.0%}",
        f"Risk Level: {plan.risk_level}",
        "",
        "Actionable Steps:",
    ]

    for step in plan.steps:
        lines.append(f"  [{step.step_id}] {step.description}")
        lines.append(f"       Priority: {step.priority}, Effort: {step.estimated_effort}, Confidence: {step.confidence:.0%}")
        if step.related_failures:
            lines.append(f"       Related failures: {', '.join(step.related_failures[:2])}")

    if plan.prerequisites:
        lines.append("")
        lines.append("Prerequisites:")
        for prereq in plan.prerequisites:
            lines.append(f"  - {prereq}")

    if plan.missing_evidence:
        lines.append("")
        lines.append("Missing Evidence (would improve confidence):")
        for evidence in plan.missing_evidence:
            lines.append(f"  - {evidence}")

    lines.extend([
        "",
        "Respond with:",
        "1. A brief interpretation of the plan",
        "2. Specific recommendations for each P0/P1 step",
        "3. Any additional context that would help",
    ])

    return "\n".join(lines)
