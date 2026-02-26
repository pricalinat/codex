"""Hybrid analyzer combining RAG retrieval with LLM analysis.

This module provides a unified analyzer that leverages both retrieval-augmented
analysis and LLM-based deep analysis. It automatically determines when to use
each approach and combines results for comprehensive test analysis.

Features:
- Seamless RAG + LLM integration
- Automatic fallback when LLM is unavailable
- Structured output from LLM analysis
- Confidence-based routing between RAG and LLM
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .llm_integration import (
    LLMAnalyzer,
    LLMConfig,
    LLMProvider,
    create_analyzer,
)
from .models import AnalysisResult
from .prompt_strategy import AnalysisPromptConfig
from .rag_analyzer import RAGAnalysisResult, RAGAnalyzer, RetrievalInsight
from .retrieval import ArtifactBundle, IngestionRecord, SourceType


@dataclass
class HybridInsight:
    """Insight from hybrid (RAG + LLM) analysis."""

    insight_type: str  # Same as RetrievalInsight types
    title: str
    description: str
    confidence: float
    source: str  # "rag" or "llm"
    evidence_chunks: List[str] = field(default_factory=list)
    severity: str = "medium"
    llm_reasoning: Optional[str] = None  # LLM-generated reasoning if applicable


@dataclass
class HybridAnalysisResult:
    """Result of hybrid (RAG + LLM) test analysis."""

    # RAG-based results
    rag_result: RAGAnalysisResult

    # LLM-based results
    llm_insights: List[HybridInsight] = field(default_factory=list)
    llm_root_cause: Optional[Dict[str, Any]] = field(default_factory=dict)
    llm_test_gaps: List[Dict[str, Any]] = field(default_factory=list)

    # Combined metadata
    hybrid_confidence: float = 0.0
    analysis_mode: str = "hybrid"  # "rag_only", "llm_only", "hybrid"
    llm_used: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rag_result": self.rag_result.to_dict(),
            "llm_insights": [
                {
                    "insight_type": i.insight_type,
                    "title": i.title,
                    "description": i.description,
                    "confidence": i.confidence,
                    "source": i.source,
                    "evidence_chunks": i.evidence_chunks,
                    "severity": i.severity,
                    "llm_reasoning": i.llm_reasoning,
                }
                for i in self.llm_insights
            ],
            "llm_root_cause": self.llm_root_cause,
            "llm_test_gaps": self.llm_test_gaps,
            "hybrid_confidence": self.hybrid_confidence,
            "analysis_mode": self.analysis_mode,
            "llm_used": self.llm_used,
            "errors": self.errors,
        }


class HybridAnalyzer:
    """Hybrid analyzer combining RAG and LLM approaches.

    This analyzer first performs RAG-based retrieval to gather context,
    then uses LLM for deeper analysis when beneficial. It provides
    automatic fallback to RAG-only mode when LLM is unavailable.

    Usage:
        analyzer = HybridAnalyzer()
        result = analyzer.analyze(test_report, repo_path="/path/to/repo")
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_config: Optional[LLMConfig] = None,
        prompt_config: Optional[AnalysisPromptConfig] = None,
        use_hybrid: bool = True,
        lexical_weight: float = 0.5,
    ):
        """Initialize the hybrid analyzer.

        Args:
            use_llm: Whether to use LLM for analysis
            llm_config: LLM configuration (defaults to mock for testing)
            prompt_config: Prompt configuration for LLM
            use_hybrid: Whether to use hybrid retrieval (RAG)
            lexical_weight: Weight for lexical search in hybrid mode
        """
        self._rag_analyzer = RAGAnalyzer(
            use_hybrid=use_hybrid,
            lexical_weight=lexical_weight,
        )
        self._use_llm = use_llm
        self._llm_analyzer: Optional[LLMAnalyzer] = None

        if use_llm:
            try:
                config = llm_config or LLMConfig()
                self._llm_analyzer = LLMAnalyzer(
                    config=config,
                    prompt_config=prompt_config,
                )
            except Exception as e:
                # Fall back to RAG-only mode
                self._llm_analyzer = None
                self._use_llm = False

    def analyze(
        self,
        test_report_content: str,
        repo_path: Optional[str] = None,
        requirements_docs: Optional[Sequence[tuple]] = None,
        system_analysis_docs: Optional[Sequence[tuple]] = None,
        knowledge_docs: Optional[Sequence[tuple]] = None,
        artifact_bundles: Optional[Sequence[ArtifactBundle]] = None,
        ingestion_records: Optional[Sequence[IngestionRecord]] = None,
        query_for_context: Optional[str] = None,
    ) -> HybridAnalysisResult:
        """Perform hybrid RAG + LLM analysis.

        Args:
            test_report_content: Raw test report content
            repo_path: Optional path to code repository
            requirements_docs: Optional requirements documents
            system_analysis_docs: Optional system analysis docs
            knowledge_docs: Optional knowledge base documents
            artifact_bundles: Optional artifact bundles (multimodal)
            ingestion_records: Optional ingestion records
            query_for_context: Optional additional query context

        Returns:
            HybridAnalysisResult with combined RAG + LLM insights
        """
        # Placeholder that will be replaced after RAG analysis
        placeholder_rag_result = RAGAnalysisResult(
            base_result=AnalysisResult(
                input_format="placeholder",
                total_failures=0,
            ),
        )
        result = HybridAnalysisResult(rag_result=placeholder_rag_result)

        # First, run RAG analysis
        try:
            # Initialize corpus if needed
            if (
                repo_path
                or requirements_docs
                or system_analysis_docs
                or knowledge_docs
                or artifact_bundles
                or ingestion_records
            ):
                self._rag_analyzer.initialize_corpus(
                    repo_path=repo_path,
                    requirements_docs=requirements_docs,
                    system_analysis_docs=system_analysis_docs,
                    knowledge_docs=knowledge_docs,
                    artifact_bundles=artifact_bundles,
                    ingestion_records=ingestion_records,
                )

            # Perform RAG analysis
            rag_result = self._rag_analyzer.analyze(
                test_report_content=test_report_content,
                query_for_context=query_for_context,
            )
            result.rag_result = rag_result

        except Exception as e:
            result.errors.append(f"RAG analysis error: {str(e)}")

        # Convert RAG insights to hybrid insights
        for insight in rag_result.retrieval_insights:
            result.llm_insights.append(HybridInsight(
                insight_type=insight.insight_type,
                title=insight.title,
                description=insight.description,
                confidence=insight.confidence,
                source="rag",
                evidence_chunks=insight.evidence_chunks,
                severity=insight.severity,
            ))

        # If LLM is available, enhance with LLM analysis
        if self._use_llm and self._llm_analyzer:
            result.llm_used = True
            result.analysis_mode = "hybrid"
            self._enhance_with_llm(result, test_report_content, query_for_context)
        else:
            result.analysis_mode = "rag_only"

        # Calculate hybrid confidence
        result.hybrid_confidence = self._calculate_hybrid_confidence(result)

        return result

    def _enhance_with_llm(
        self,
        result: HybridAnalysisResult,
        test_report_content: str,
        query_for_context: Optional[str],
    ) -> None:
        """Enhance results with LLM analysis."""
        if not self._llm_analyzer:
            return

        try:
            # Gather context from RAG results - use evidence sources as text context
            if result.rag_result.evidence_sources:
                context_chunks = result.rag_result.evidence_sources[:10]
            else:
                context_chunks = []

            # Analyze root cause with LLM
            root_cause_response = self._llm_analyzer.analyze_root_cause(
                failure_description=f"Test failures: {test_report_content[:500]}",
                context=context_chunks,
            )

            # Extract structured root cause analysis
            root_cause_json = root_cause_response.extract_json({
                "root_cause_category": "string",
                "confidence": "float",
                "reasoning": "string",
                "recommended_actions": "list",
            })

            if root_cause_json:
                result.llm_root_cause = root_cause_json

                # Add as insight
                result.llm_insights.append(HybridInsight(
                    insight_type="root_cause_evidence",
                    title="LLM Root Cause Analysis",
                    description=root_cause_json.get("reasoning", "")[:200],
                    confidence=root_cause_json.get("confidence", 0.5),
                    source="llm",
                    severity="high",
                    llm_reasoning=root_cause_json.get("reasoning"),
                ))

            # Analyze test gaps with LLM
            gaps_response = self._llm_analyzer.analyze_test_gaps(
                error_type=str(result.rag_result.base_result.failures[0].error_type) if result.rag_result.base_result.failures else "unknown",
                context=context_chunks,
            )

            gaps_json = gaps_response.extract_json({
                "gap_type": "string",
                "severity": "string",
                "suggested_action": "string",
                "test_coverage_needed": "string",
            })

            if gaps_json:
                result.llm_test_gaps.append(gaps_json)

                # Add as insight
                result.llm_insights.append(HybridInsight(
                    insight_type="test_gap",
                    title="LLM Test Gap Analysis",
                    description=gaps_json.get("suggested_action", ""),
                    confidence=0.7,
                    source="llm",
                    severity=gaps_json.get("severity", "medium"),
                    llm_reasoning=gaps_json.get("test_coverage_needed"),
                ))

        except Exception as e:
            result.errors.append(f"LLM enhancement error: {str(e)}")
            # Fall back to RAG-only mode
            result.analysis_mode = "rag_only"

    def _calculate_hybrid_confidence(self, result: HybridAnalysisResult) -> float:
        """Calculate combined confidence score."""
        # Start with RAG confidence if available
        rag_confidence = result.rag_result.risk_assessment.get(
            "retrieval_confidence", 0.3
        )

        if result.llm_used and result.llm_root_cause:
            # Blend RAG and LLM confidences
            llm_confidence = result.llm_root_cause.get("confidence", 0.5)
            # Weight LLM slightly higher if available
            return round((rag_confidence * 0.4 + llm_confidence * 0.6), 4)

        return round(rag_confidence, 4)


def hybrid_analyze(
    test_report_content: str,
    repo_path: Optional[str] = None,
    requirements_docs: Optional[Sequence[tuple]] = None,
    system_analysis_docs: Optional[Sequence[tuple]] = None,
    knowledge_docs: Optional[Sequence[tuple]] = None,
    artifact_bundles: Optional[Sequence[ArtifactBundle]] = None,
    ingestion_records: Optional[Sequence[IngestionRecord]] = None,
    query_for_context: Optional[str] = None,
    use_llm: bool = True,
    llm_provider: LLMProvider = LLMProvider.MOCK,
    model: str = "gpt-4o-mini",
) -> HybridAnalysisResult:
    """Convenience function for hybrid RAG + LLM analysis.

    Args:
        test_report_content: Raw test report content
        repo_path: Optional path to code repository
        requirements_docs: Optional requirements documents
        system_analysis_docs: Optional system analysis docs
        knowledge_docs: Optional knowledge base documents
        artifact_bundles: Optional artifact bundles (multimodal)
        ingestion_records: Optional ingestion records
        query_for_context: Optional additional query context
        use_llm: Whether to use LLM for analysis
        llm_provider: LLM provider to use
        model: Model name

    Returns:
        HybridAnalysisResult with combined insights
    """
    llm_config = None
    if use_llm:
        llm_config = LLMConfig(provider=llm_provider, model=model)

    analyzer = HybridAnalyzer(use_llm=use_llm, llm_config=llm_config)

    return analyzer.analyze(
        test_report_content=test_report_content,
        repo_path=repo_path,
        requirements_docs=requirements_docs,
        system_analysis_docs=system_analysis_docs,
        knowledge_docs=knowledge_docs,
        artifact_bundles=artifact_bundles,
        ingestion_records=ingestion_records,
        query_for_context=query_for_context,
    )
