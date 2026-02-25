"""RAG-augmented test analysis with retrieval-enhanced insights."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .analyzer import analyze_report_text
from .models import AnalysisResult, FailureCluster, FixSuggestion
from .retrieval import (
    IngestDocument,
    MultiSourceIngestor,
    RankedChunk,
    RetrievalEngine,
    SourceType,
    build_analysis_prompt,
)


@dataclass
class RetrievalInsight:
    """Insight from retrieval-augmented analysis."""

    insight_type: str  # test_gap, risk_factor, root_cause_evidence, coverage_recommendation
    title: str
    description: str
    confidence: float
    evidence_chunks: List[str] = field(default_factory=list)
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class RAGAnalysisResult:
    """Result of RAG-augmented test analysis."""

    base_result: AnalysisResult
    retrieval_insights: List[RetrievalInsight] = field(default_factory=list)
    test_gap_analysis: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    augmented_prompt: str = ""
    evidence_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_result": self.base_result.to_dict(),
            "retrieval_insights": [
                {
                    "insight_type": i.insight_type,
                    "title": i.title,
                    "description": i.description,
                    "confidence": i.confidence,
                    "evidence_chunks": i.evidence_chunks,
                    "severity": i.severity,
                }
                for i in self.retrieval_insights
            ],
            "test_gap_analysis": self.test_gap_analysis,
            "risk_assessment": self.risk_assessment,
            "evidence_sources": self.evidence_sources,
        }


class RAGAnalyzer:
    """RAG-augmented test analysis engine."""

    def __init__(self, chunk_size: int = 360, chunk_overlap: int = 40) -> None:
        self._engine = RetrievalEngine(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._ingestor = MultiSourceIngestor(self._engine)
        self._initialized = False

    def initialize_corpus(
        self,
        repo_path: Optional[str] = None,
        requirements_docs: Optional[Sequence[str]] = None,
        system_analysis_docs: Optional[Sequence[str]] = None,
        knowledge_docs: Optional[Sequence[str]] = None,
    ) -> int:
        """Initialize the retrieval corpus with various document sources.

        Args:
            repo_path: Path to repository to ingest
            requirements_docs: List of (source_id, markdown_content) tuples
            system_analysis_docs: List of (source_id, content) tuples
            knowledge_docs: List of (source_id, content) tuples

        Returns:
            Total number of chunks indexed
        """
        total_chunks = 0

        if repo_path:
            chunks = self._ingestor.ingest_repository(repo_path)
            total_chunks += len(chunks)

        if requirements_docs:
            for source_id, content in requirements_docs:
                chunks = self._ingestor.ingest_requirements_markdown(source_id, content)
                total_chunks += len(chunks)

        if system_analysis_docs:
            for source_id, content in system_analysis_docs:
                chunks = self._ingestor.ingest_raw(
                    source_id=source_id,
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    content=content,
                )
                total_chunks += len(chunks)

        if knowledge_docs:
            for source_id, content in knowledge_docs:
                chunks = self._ingestor.ingest_raw(
                    source_id=source_id,
                    source_type=SourceType.KNOWLEDGE,
                    content=content,
                )
                total_chunks += len(chunks)

        self._initialized = total_chunks > 0
        return total_chunks

    def add_knowledge(self, source_id: str, content: str) -> int:
        """Add a knowledge document to the corpus."""
        chunks = self._ingestor.ingest_raw(
            source_id=source_id,
            source_type=SourceType.KNOWLEDGE,
            content=content,
        )
        self._initialized = True
        return len(chunks)

    def analyze(
        self,
        test_report_content: str,
        query_for_context: Optional[str] = None,
    ) -> RAGAnalysisResult:
        """Perform RAG-augmented test analysis.

        Args:
            test_report_content: Raw test report (JUnit XML or pytest text)
            query_for_context: Optional additional query for retrieval context

        Returns:
            RAGAnalysisResult with augmented insights
        """
        # Base analysis from test report
        base_result = analyze_report_text(test_report_content)

        if not self._initialized or not self._engine._chunks:
            # No corpus initialized, return base result only
            return RAGAnalysisResult(
                base_result=base_result,
                test_gap_analysis=["Initialize corpus with requirements/code for gap analysis"],
                risk_assessment={"status": "no_context", "message": "No corpus initialized"},
            )

        # Build queries for retrieval
        queries = self._build_analysis_queries(base_result, query_for_context)

        # Retrieve context for each analysis type
        all_insights: List[RetrievalInsight] = []
        all_evidence_sources: List[str] = []
        test_gaps: List[str] = []
        risk_factors: Dict[str, Any] = {"factors": [], "overall_risk": "low"}

        for query in queries:
            ranked = self._engine.query(query, top_k=5, diversify=True)
            if not ranked:
                continue

            # Track evidence sources
            all_evidence_sources.extend(item.chunk.source_id for item in ranked)

            # Generate insights based on query type
            insights = self._generate_insights(query, ranked)
            all_insights.extend(insights)

            # Categorize insights
            for insight in insights:
                if insight.insight_type == "test_gap":
                    test_gaps.append(insight.description)
                elif insight.insight_type == "risk_factor":
                    if insight.severity in ("high", "critical"):
                        risk_factors["factors"].append({
                            "title": insight.title,
                            "severity": insight.severity,
                            "description": insight.description,
                        })

        # Calculate overall risk
        if risk_factors["factors"]:
            severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for f in risk_factors["factors"]:
                severity_counts[f.get("severity", "low")] = (
                    severity_counts.get(f.get("severity", "low"), 0) + 1
                )
            if severity_counts.get("critical", 0) > 0:
                risk_factors["overall_risk"] = "critical"
            elif severity_counts.get("high", 0) > 0:
                risk_factors["overall_risk"] = "high"
            elif severity_counts.get("medium", 0) > 0:
                risk_factors["overall_risk"] = "medium"

        # Build augmented prompt for LLM
        ranked_for_prompt = self._engine.query(
            "failure clustering root cause test gap risk prioritization",
            top_k=10,
            diversify=True,
        )
        augmented_prompt = build_analysis_prompt(
            question=query_for_context or "Analyze test failures with context from retrieved documents",
            ranked_context=ranked_for_prompt,
        )

        return RAGAnalysisResult(
            base_result=base_result,
            retrieval_insights=all_insights,
            test_gap_analysis=test_gaps,
            risk_assessment=risk_factors,
            augmented_prompt=augmented_prompt,
            evidence_sources=list(set(all_evidence_sources)),
        )

    def _build_analysis_queries(
        self,
        base_result: AnalysisResult,
        extra_query: Optional[str],
    ) -> List[str]:
        """Build queries for retrieval based on analysis results."""
        queries = []

        # Query for test gaps based on failure types
        if base_result.clusters:
            error_types = [c.error_type for c in base_result.clusters]
            queries.append(f"test gap {', '.join(error_types)} missing coverage")

        # Query for root cause evidence
        if base_result.failures:
            sample_errors = [f.error_type for f in base_result.failures[:3]]
            queries.append(f"root cause analysis {', '.join(sample_errors)}")

        # Query for risk factors
        queries.append("release risk prioritization blocking issues")

        # Query for requirements-based coverage gaps
        queries.append("requirements coverage missing test scenarios")

        # Add user-provided query
        if extra_query:
            queries.append(extra_query)

        return queries

    def _generate_insights(
        self,
        query: str,
        ranked: List[RankedChunk],
    ) -> List[RetrievalInsight]:
        """Generate insights from retrieved chunks."""
        insights = []

        # Determine insight type from query
        query_lower = query.lower()
        if "test gap" in query_lower or "missing" in query_lower or "coverage" in query_lower:
            insight_type = "test_gap"
        elif "risk" in query_lower or "release" in query_lower or "blocking" in query_lower:
            insight_type = "risk_factor"
        elif "root cause" in query_lower:
            insight_type = "root_cause_evidence"
        else:
            insight_type = "coverage_recommendation"

        # Generate insights from top chunks
        for chunk in ranked[:3]:
            if chunk.confidence < 0.1:
                continue

            title = f"Insight from {chunk.chunk.source_id}"
            description = chunk.chunk.text[:200] + ("..." if len(chunk.chunk.text) > 200 else "")
            severity = _assess_severity(chunk.chunk.text, insight_type)

            insights.append(
                RetrievalInsight(
                    insight_type=insight_type,
                    title=title,
                    description=description,
                    confidence=chunk.confidence,
                    evidence_chunks=[chunk.chunk.source_id],
                    severity=severity,
                )
            )

        return insights


def _assess_severity(text: str, insight_type: str) -> str:
    """Assess severity based on text content."""
    text_lower = text.lower()

    # Critical indicators
    if any(kw in text_lower for kw in ["critical", "security", "breach", "data loss"]):
        return "critical"

    # High severity indicators
    if any(kw in text_lower for kw in ["release blocking", "p0", "high risk", "blocking"]):
        return "high"

    # Medium severity indicators
    if any(kw in text_lower for kw in ["medium", "moderate", "should", "recommended"]):
        return "medium"

    return "low"


def rag_analyze(
    test_report_content: str,
    repo_path: Optional[str] = None,
    requirements_docs: Optional[Sequence[tuple]] = None,
    query: Optional[str] = None,
) -> RAGAnalysisResult:
    """Convenience function for RAG-augmented analysis.

    Args:
        test_report_content: Raw test report (JUnit XML or pytest text)
        repo_path: Optional path to repository
        requirements_docs: Optional list of (source_id, markdown) tuples
        query: Optional additional query

    Returns:
        RAGAnalysisResult with augmented insights
    """
    analyzer = RAGAnalyzer()

    # Initialize corpus
    if repo_path or requirements_docs:
        analyzer.initialize_corpus(
            repo_path=repo_path,
            requirements_docs=requirements_docs,
        )

    return analyzer.analyze(test_report_content, query_for_context=query)
