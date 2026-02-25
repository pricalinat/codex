"""RAG-augmented test analysis with retrieval-enhanced insights."""

import inspect
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .analyzer import analyze_report_text
from .models import AnalysisResult, FailureCluster, FixSuggestion
from .retrieval import (
    AnalysisEvidencePack,
    ArtifactBundle,
    CodeAwareIngestor,
    DummyEmbeddingProvider,
    HybridRetrievalEngine,
    IngestionRecord,
    IngestDocument,
    MultiSourceIngestor,
    RankedChunk,
    RetrievalEngine,
    SourceType,
    build_analysis_prompt_from_pack,
    create_hybrid_engine,
)


class ChunkerType(str, Enum):
    """Type of chunking strategy to use."""
    BASIC = "basic"  # Simple fixed-size text chunking
    CODE_AWARE = "code_aware"  # Code-structure-aware chunking
    AUTO = "auto"  # Auto-detect content type and use appropriate chunker


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
class RequirementTrace:
    """Traceability between requirements and test coverage."""

    requirement_id: str
    requirement_text: str
    covered_by_tests: List[str] = field(default_factory=list)
    gap_type: str = "none"  # none, uncovered, partially_covered, needs_negative, needs_edge_case
    coverage_confidence: float = 0.0
    related_failures: List[str] = field(default_factory=list)


@dataclass
class TestGapAnalysis:
    """Structured test gap analysis."""

    gap_id: str
    gap_type: str  # missing_test, edge_case, negative_case, boundary, error_handling
    description: str
    severity: str  # low, medium, high, critical
    related_requirements: List[str] = field(default_factory=list)
    suggested_test_count: int = 1
    confidence: float = 0.0


@dataclass
class RAGAnalysisResult:
    """Result of RAG-augmented test analysis."""

    base_result: AnalysisResult
    retrieval_insights: List[RetrievalInsight] = field(default_factory=list)
    test_gap_analysis: List[str] = field(default_factory=list)
    structured_gaps: List[TestGapAnalysis] = field(default_factory=list)
    requirement_traces: List[RequirementTrace] = field(default_factory=list)
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
            "structured_gaps": [
                {
                    "gap_id": g.gap_id,
                    "gap_type": g.gap_type,
                    "description": g.description,
                    "severity": g.severity,
                    "related_requirements": g.related_requirements,
                    "suggested_test_count": g.suggested_test_count,
                    "confidence": g.confidence,
                }
                for g in self.structured_gaps
            ],
            "requirement_traces": [
                {
                    "requirement_id": r.requirement_id,
                    "requirement_text": r.requirement_text[:100] + "..." if len(r.requirement_text) > 100 else r.requirement_text,
                    "covered_by_tests": r.covered_by_tests,
                    "gap_type": r.gap_type,
                    "coverage_confidence": r.coverage_confidence,
                    "related_failures": r.related_failures,
                }
                for r in self.requirement_traces
            ],
            "risk_assessment": self.risk_assessment,
            "evidence_sources": self.evidence_sources,
        }


class RAGAnalyzer:
    """RAG-augmented test analysis engine."""

    def __init__(
        self,
        chunk_size: int = 360,
        chunk_overlap: int = 40,
        use_hybrid: bool = True,
        lexical_weight: float = 0.5,
        chunker_type: ChunkerType = ChunkerType.AUTO,
    ) -> None:
        if use_hybrid:
            self._engine = create_hybrid_engine(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                lexical_weight=lexical_weight,
            )
        else:
            self._engine = RetrievalEngine(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._chunker_type = chunker_type
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Set up ingestor based on chunker type
        if chunker_type == ChunkerType.CODE_AWARE:
            self._ingestor = CodeAwareIngestor(
                self._engine,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self._code_ingestor: Optional[CodeAwareIngestor] = self._ingestor
            self._basic_ingestor: Optional[MultiSourceIngestor] = None
        elif chunker_type == ChunkerType.AUTO:
            # Auto mode: we'll use both and pick appropriate
            self._code_ingestor = CodeAwareIngestor(
                self._engine,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self._basic_ingestor = MultiSourceIngestor(self._engine)
            self._ingestor = self._basic_ingestor  # Default to basic
        else:
            self._ingestor = MultiSourceIngestor(self._engine)
            self._basic_ingestor = self._ingestor
            self._code_ingestor = None
        self._initialized = False

    def _select_ingestor(self, source_type: SourceType, source_id: str) -> Any:
        """Select appropriate ingestor based on source type and content.

        Args:
            source_type: The type of source being ingested
            source_id: The source identifier (file path)

        Returns:
            The appropriate ingestor instance
        """
        if self._chunker_type == ChunkerType.CODE_AWARE:
            return self._ingestor
        if self._chunker_type == ChunkerType.AUTO:
            # Auto-select based on source type and file extension
            if source_type in (SourceType.REPOSITORY, SourceType.CODE_SNIPPET):
                # Check if it's a code file
                if self._is_code_file(source_id):
                    return self._code_ingestor
            return self._basic_ingestor
        return self._ingestor

    def _is_code_file(self, source_id: str) -> bool:
        """Check if source is a code file based on extension.

        Args:
            source_id: Source identifier (file path)

        Returns:
            True if appears to be a code file
        """
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.c', '.cpp', '.h'}
        lower_id = source_id.lower()
        return any(lower_id.endswith(ext) for ext in code_extensions)

    def initialize_corpus(
        self,
        repo_path: Optional[str] = None,
        requirements_docs: Optional[Sequence[tuple]] = None,
        system_analysis_docs: Optional[Sequence[tuple]] = None,
        knowledge_docs: Optional[Sequence[tuple]] = None,
        artifact_bundles: Optional[Sequence[ArtifactBundle]] = None,
        ingestion_records: Optional[Sequence[IngestionRecord]] = None,
        generate_source_summaries: bool = False,
    ) -> int:
        """Initialize the retrieval corpus with various document sources.

        Args:
            repo_path: Path to repository to ingest
            requirements_docs: List of (source_id, markdown_content) tuples
            system_analysis_docs: List of (source_id, content) tuples
            knowledge_docs: List of (source_id, content) tuples
            artifact_bundles: Mixed-modality extraction/OCR payloads
            ingestion_records: Unified mixed-source records (multimodal/code/docs)
            generate_source_summaries: Whether to synthesize per-source summary chunks

        Returns:
            Total number of chunks indexed
        """
        total_chunks = 0

        if repo_path:
            chunks = self._call_ingestion_method(
                self._ingestor.ingest_repository,
                repo_path,
                generate_source_summaries=generate_source_summaries,
            )
            total_chunks += len(chunks)

        if requirements_docs:
            for source_id, content in requirements_docs:
                ingestor = self._select_ingestor(SourceType.REQUIREMENTS, source_id)
                chunks = self._call_ingestion_method(
                    ingestor.ingest_requirements_markdown,
                    source_id,
                    content,
                    generate_source_summaries=generate_source_summaries,
                )
                total_chunks += len(chunks)

        if system_analysis_docs:
            for source_id, content in system_analysis_docs:
                ingestor = self._select_ingestor(SourceType.SYSTEM_ANALYSIS, source_id)
                chunks = self._call_ingestion_method(
                    ingestor.ingest_raw,
                    source_id=source_id,
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    content=content,
                    generate_source_summaries=generate_source_summaries,
                )
                total_chunks += len(chunks)

        if knowledge_docs:
            for source_id, content in knowledge_docs:
                ingestor = self._select_ingestor(SourceType.KNOWLEDGE, source_id)
                chunks = self._call_ingestion_method(
                    ingestor.ingest_raw,
                    source_id=source_id,
                    source_type=SourceType.KNOWLEDGE,
                    content=content,
                    generate_source_summaries=generate_source_summaries,
                )
                total_chunks += len(chunks)

        if artifact_bundles:
            for bundle in artifact_bundles:
                ingestor = self._select_ingestor(bundle.source_type, bundle.source_id)
                if hasattr(ingestor, "ingest_artifact_bundle"):
                    chunks = self._call_ingestion_method(
                        ingestor.ingest_artifact_bundle,
                        bundle,
                        generate_source_summaries=generate_source_summaries,
                    )
                else:
                    chunks = self._call_ingestion_method(
                        ingestor.ingest_raw,
                        source_id=bundle.source_id,
                        source_type=bundle.source_type,
                        content={
                            "text": bundle.text,
                            "tables": list(bundle.tables),
                            "images": list(bundle.images),
                        },
                        modality="compound",
                        metadata=bundle.metadata,
                        generate_source_summaries=generate_source_summaries,
                    )
                total_chunks += len(chunks)

        if ingestion_records:
            for record in ingestion_records:
                total_chunks += self._ingest_record(
                    record,
                    generate_source_summaries=generate_source_summaries,
                )

        self._initialized = total_chunks > 0
        return total_chunks

    def _ingest_record(
        self,
        record: IngestionRecord,
        generate_source_summaries: bool = False,
    ) -> int:
        if (
            record.source_type == SourceType.CODE_SNIPPET
            and self._code_ingestor is not None
        ):
            code_payload = self._extract_code_payload(record.payload)
            if code_payload:
                chunks = self._code_ingestor.ingest_code_snippet(
                    source_id=record.source_id,
                    code_content=code_payload,
                    metadata=dict(record.metadata),
                )
                return len(chunks)

        ingestor = self._select_ingestor(record.source_type, record.source_id)
        if hasattr(ingestor, "ingest_records"):
            chunks = self._call_ingestion_method(
                ingestor.ingest_records,
                [record],
                generate_source_summaries=generate_source_summaries,
            )
            return len(chunks)

        fallback_ingestor = self._basic_ingestor or MultiSourceIngestor(self._engine)
        chunks = self._call_ingestion_method(
            fallback_ingestor.ingest_records,
            [record],
            generate_source_summaries=generate_source_summaries,
        )
        return len(chunks)

    def _extract_code_payload(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            for key in ("code", "content", "text", "snippet"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    def _call_ingestion_method(
        self,
        method: Any,
        *args: Any,
        generate_source_summaries: bool = False,
        **kwargs: Any,
    ) -> Any:
        if generate_source_summaries and _method_supports_keyword(method, "generate_source_summaries"):
            kwargs["generate_source_summaries"] = True
        return method(*args, **kwargs)

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
        missing_artifacts = {"source_types": set(), "modalities": set()}
        unavailable_artifacts = {"source_types": set(), "modalities": set()}
        retrieval_confidences: List[float] = []
        retrieval_confidences_raw: List[float] = []

        for query in queries:
            evidence = self._engine.retrieve_evidence(query, top_k=5, diversify=True)
            ranked = evidence.ranked_chunks
            if not ranked:
                continue

            retrieval_confidences.append(evidence.calibrated_confidence)
            retrieval_confidences_raw.append(evidence.aggregate_confidence)
            for missing_source in evidence.missing_source_types:
                missing_artifacts["source_types"].add(missing_source.value)
            for missing_modality in evidence.missing_modalities:
                missing_artifacts["modalities"].add(missing_modality)
            for unavailable_source in evidence.unavailable_preferred_source_types:
                unavailable_artifacts["source_types"].add(unavailable_source.value)
            for unavailable_modality in evidence.unavailable_preferred_modalities:
                unavailable_artifacts["modalities"].add(unavailable_modality)

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
        if retrieval_confidences:
            risk_factors["retrieval_confidence"] = round(
                sum(retrieval_confidences) / len(retrieval_confidences),
                4,
            )
        if retrieval_confidences_raw:
            risk_factors["retrieval_confidence_raw"] = round(
                sum(retrieval_confidences_raw) / len(retrieval_confidences_raw),
                4,
            )
        if missing_artifacts["source_types"] or missing_artifacts["modalities"]:
            risk_factors["missing_evidence"] = {
                "source_types": sorted(missing_artifacts["source_types"]),
                "modalities": sorted(missing_artifacts["modalities"]),
            }
        if unavailable_artifacts["source_types"] or unavailable_artifacts["modalities"]:
            risk_factors["unavailable_evidence"] = {
                "source_types": sorted(unavailable_artifacts["source_types"]),
                "modalities": sorted(unavailable_artifacts["modalities"]),
            }

        # Generate structured test gaps and requirement traces
        structured_gaps = self._analyze_test_gaps(base_result)
        requirement_traces = self._trace_requirements(base_result)

        # Build augmented prompt for LLM
        analysis_pack: AnalysisEvidencePack = self._engine.retrieve_analysis_evidence_pack(
            query_for_context or "failure clustering root cause test gap risk prioritization",
            top_k_per_focus=3,
            diversify=True,
            use_expansion=True,
            adaptive_recovery=True,
        )
        augmented_prompt = build_analysis_prompt_from_pack(
            question=query_for_context or "Analyze test failures with context from retrieved documents",
            pack=analysis_pack,
        )
        if analysis_pack.focus_confidence:
            risk_factors["focus_confidence"] = dict(sorted(analysis_pack.focus_confidence.items()))
            risk_factors["focus_overall_confidence"] = analysis_pack.overall_confidence

        return RAGAnalysisResult(
            base_result=base_result,
            retrieval_insights=all_insights,
            test_gap_analysis=test_gaps,
            structured_gaps=structured_gaps,
            requirement_traces=requirement_traces,
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

    def _analyze_test_gaps(self, base_result: AnalysisResult) -> List[TestGapAnalysis]:
        """Analyze and identify structured test gaps.

        Args:
            base_result: The base analysis result from test report

        Returns:
            List of identified test gaps with structured information
        """
        gaps: List[TestGapAnalysis] = []
        gap_id_counter = 1

        # Analyze failure clusters for gap patterns
        for cluster in base_result.clusters:
            error_type = cluster.error_type
            count = cluster.count

            # Determine gap type based on error pattern
            gap_type, severity, description = self._classify_error_gap(error_type, count)

            if gap_type != "none":
                gaps.append(
                    TestGapAnalysis(
                        gap_id=f"G{gap_id_counter:03d}",
                        gap_type=gap_type,
                        description=description,
                        severity=severity,
                        related_requirements=[],
                        suggested_test_count=max(1, count),
                        confidence=0.75,
                    )
                )
                gap_id_counter += 1

        # Query for requirements-based gaps
        ranked = self._engine.query("requirement missing test case", top_k=5, diversify=True)
        for chunk in ranked[:3]:
            if chunk.confidence < 0.2:
                continue
            gaps.append(
                TestGapAnalysis(
                    gap_id=f"G{gap_id_counter:03d}",
                    gap_type="missing_test",
                    description=chunk.chunk.text[:150],
                    severity="medium",
                    related_requirements=[chunk.chunk.source_id],
                    suggested_test_count=1,
                    confidence=chunk.confidence,
                )
            )
            gap_id_counter += 1

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        gaps.sort(key=lambda g: (severity_order.get(g.severity, 3), g.gap_id))

        return gaps

    def _classify_error_gap(self, error_type: str, count: int) -> tuple:
        """Classify error type into test gap category.

        Returns:
            Tuple of (gap_type, severity, description)
        """
        if error_type in ("ModuleNotFoundError", "ImportError", "SyntaxError"):
            return ("missing_test", "high", f"Missing dependency: {error_type}")
        if error_type == "AssertionError":
            return ("edge_case", "medium", "Assertion failures suggest missing edge case tests")
        if error_type == "TypeError":
            return ("boundary", "medium", "Type errors suggest missing boundary condition tests")
        if error_type == "TimeoutError":
            return ("error_handling", "high", "Timeout suggests missing timeout/error handling tests")
        if error_type == "RuntimeError":
            return ("error_handling", "medium", "Runtime errors suggest missing exception handling tests")
        return ("none", "low", "")

    def _trace_requirements(self, base_result: AnalysisResult) -> List[RequirementTrace]:
        """Trace requirements to test coverage.

        Args:
            base_result: The base analysis result from test report

        Returns:
            List of requirement traces with coverage information
        """
        traces: List[RequirementTrace] = []

        # Get requirement chunks
        req_chunks = [
            chunk for chunk in self._engine._chunks
            if chunk.source_type == SourceType.REQUIREMENTS
        ]

        # Get failing test names
        failing_tests = [f"{f.suite}::{f.test_name}" for f in base_result.failures]

        for chunk in req_chunks:
            # Extract requirement ID from source
            req_id = chunk.source_id
            req_text = chunk.text[:200]

            # Check if requirement is covered by tests
            covered = self._check_coverage(chunk.text, failing_tests)

            # Determine gap type
            if not covered:
                gap_type = "uncovered"
                confidence = 0.6
            elif failing_tests:
                gap_type = "partially_covered"
                confidence = 0.7
            else:
                gap_type = "none"
                confidence = 0.9

            traces.append(
                RequirementTrace(
                    requirement_id=req_id,
                    requirement_text=req_text,
                    covered_by_tests=covered,
                    gap_type=gap_type,
                    coverage_confidence=confidence,
                    related_failures=[t for t in failing_tests if any(c in t.lower() for c in covered)] if covered else [],
                )
            )

        return traces

    def _check_coverage(self, req_text: str, test_names: List[str]) -> List[str]:
        """Check which tests cover a requirement.

        Args:
            req_text: Requirement text
            test_names: List of test names

        Returns:
            List of test names that appear to cover this requirement
        """
        req_tokens = set(re.findall(r"[a-zA-Z0-9_]+", req_text.lower()))
        covered = []

        for test in test_names:
            test_tokens = set(re.findall(r"[a-zA-Z0-9_]+", test.lower()))
            # Simple token overlap check
            overlap = req_tokens.intersection(test_tokens)
            if len(overlap) >= 2:  # At least 2 matching tokens
                covered.append(test)

        return covered


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
    system_analysis_docs: Optional[Sequence[tuple]] = None,
    knowledge_docs: Optional[Sequence[tuple]] = None,
    artifact_bundles: Optional[Sequence[ArtifactBundle]] = None,
    ingestion_records: Optional[Sequence[IngestionRecord]] = None,
    query: Optional[str] = None,
    use_hybrid: bool = True,
    lexical_weight: float = 0.5,
    chunker_type: ChunkerType = ChunkerType.AUTO,
    generate_source_summaries: bool = False,
) -> RAGAnalysisResult:
    """Convenience function for RAG-augmented analysis.

    Args:
        test_report_content: Raw test report (JUnit XML or pytest text)
        repo_path: Optional path to repository
        requirements_docs: Optional list of (source_id, markdown) tuples
        system_analysis_docs: Optional list of (source_id, content) tuples
        knowledge_docs: Optional list of (source_id, content) tuples
        artifact_bundles: Optional mixed-modality extraction payloads
        ingestion_records: Optional normalized ingestion records
        query: Optional additional query
        use_hybrid: Whether to use hybrid (lexical + semantic) retrieval
        lexical_weight: Weight for lexical search when using hybrid mode (0-1)
        chunker_type: Chunker strategy (basic, code_aware, auto)
        generate_source_summaries: Whether to synthesize per-source summary chunks

    Returns:
        RAGAnalysisResult with augmented insights
    """
    analyzer = RAGAnalyzer(
        use_hybrid=use_hybrid,
        lexical_weight=lexical_weight,
        chunker_type=chunker_type,
    )

    # Initialize corpus
    if (
        repo_path
        or requirements_docs
        or system_analysis_docs
        or knowledge_docs
        or artifact_bundles
        or ingestion_records
    ):
        analyzer.initialize_corpus(
            repo_path=repo_path,
            requirements_docs=requirements_docs,
            system_analysis_docs=system_analysis_docs,
            knowledge_docs=knowledge_docs,
            artifact_bundles=artifact_bundles,
            ingestion_records=ingestion_records,
            generate_source_summaries=generate_source_summaries,
        )

    return analyzer.analyze(test_report_content, query_for_context=query)


def _method_supports_keyword(method: Any, keyword: str) -> bool:
    try:
        params = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
    return keyword in params
