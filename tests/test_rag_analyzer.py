import unittest

from src.test_analysis_assistant.rag_analyzer import (
    ChunkerType,
    RAGAnalyzer,
    RAGAnalysisResult,
    RetrievalInsight,
    rag_analyze,
)
from src.test_analysis_assistant.ingestion_pipeline import UnifiedIngestionPipeline
from src.test_analysis_assistant.retrieval import (
    ArtifactBundle,
    Chunk,
    IngestDocument,
    IngestionRecord,
    SourceType,
    compute_enhanced_confidence,
)


class TestRAGAnalyzer(unittest.TestCase):
    def test_rag_analyzer_initialization(self):
        analyzer = RAGAnalyzer()
        self.assertIsNotNone(analyzer._engine)
        self.assertFalse(analyzer._initialized)

    def test_add_knowledge(self):
        analyzer = RAGAnalyzer()
        chunks = analyzer.add_knowledge("test-doc-1", "Test knowledge content about auth failures.")
        self.assertEqual(chunks, 1)
        self.assertTrue(analyzer._initialized)

    def test_analyze_without_corpus(self):
        test_report = """<testsuite name="pytest" errors="0" failures="2" tests="5">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
            <testcase classname="test_auth" name="test_logout">
                <failure type="TypeError">Cannot read property 'user' of undefined</failure>
            </testcase>
        </testsuite>"""

        analyzer = RAGAnalyzer()
        result = analyzer.analyze(test_report)

        self.assertIsInstance(result, RAGAnalysisResult)
        self.assertEqual(result.base_result.total_failures, 2)
        self.assertIn("no_context", result.risk_assessment.get("status", ""))

    def test_analyze_with_knowledge_corpus(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="3">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
        </testsuite>"""

        analyzer = RAGAnalyzer()
        analyzer.add_knowledge(
            "auth-req",
            "Authentication failures are release blocking. Missing negative test cases for invalid credentials.",
        )
        analyzer.add_knowledge(
            "test-gap",
            "No test coverage for expired token scenarios in auth module.",
        )

        result = analyzer.analyze(test_report, query_for_context="authentication test gaps")

        self.assertIsInstance(result, RAGAnalysisResult)
        self.assertEqual(result.base_result.total_failures, 1)
        self.assertTrue(len(result.retrieval_insights) > 0 or len(result.test_gap_analysis) > 0)

    def test_rag_analyze_convenience_function(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_api" name="test_status">
                <failure type="RuntimeError">Invalid response</failure>
            </testcase>
        </testsuite>"""

        result = rag_analyze(
            test_report_content=test_report,
            requirements_docs=[("req-1", "API should return 200 OK for health checks.")],
            query="API test gaps",
        )

        self.assertIsInstance(result, RAGAnalysisResult)
        self.assertEqual(result.base_result.total_failures, 1)

    def test_retrieval_insight_dataclass(self):
        insight = RetrievalInsight(
            insight_type="test_gap",
            title="Missing auth tests",
            description="No negative test for invalid tokens",
            confidence=0.85,
            evidence_chunks=["req:auth"],
            severity="high",
        )

        self.assertEqual(insight.insight_type, "test_gap")
        self.assertEqual(insight.severity, "high")
        self.assertEqual(insight.confidence, 0.85)

    def test_rag_result_to_dict(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_x" name="test_y">
                <failure type="Error">Failed</failure>
            </testcase>
        </testsuite>"""

        analyzer = RAGAnalyzer()
        analyzer.add_knowledge("doc1", "Some knowledge content.")
        result = analyzer.analyze(test_report)

        result_dict = result.to_dict()

        self.assertIn("base_result", result_dict)
        self.assertIn("retrieval_insights", result_dict)
        self.assertIn("test_gap_analysis", result_dict)
        self.assertIn("risk_assessment", result_dict)
        self.assertIn("evidence_sources", result_dict)

    def test_analyze_includes_retrieval_confidence_metadata(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
        </testsuite>"""

        analyzer = RAGAnalyzer()
        analyzer.add_knowledge(
            "auth-req",
            "Authentication release risk requires negative tests and mitigation planning.",
        )
        result = analyzer.analyze(test_report, query_for_context="auth release risk")

        self.assertIn("retrieval_confidence", result.risk_assessment)
        self.assertGreaterEqual(result.risk_assessment["retrieval_confidence"], 0.0)
        self.assertLessEqual(result.risk_assessment["retrieval_confidence"], 1.0)
        self.assertIn("retrieval_confidence_raw", result.risk_assessment)
        self.assertGreaterEqual(
            result.risk_assessment["retrieval_confidence_raw"],
            result.risk_assessment["retrieval_confidence"],
        )

    def test_analyze_prompt_includes_source_bundle_summary(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
        </testsuite>"""
        markdown = """
# Auth Requirements
Missing negative authorization tests are release blocking.

| risk | severity |
| ---- | -------- |
| auth | high     |
""".strip()

        analyzer = RAGAnalyzer()
        analyzer.initialize_corpus(requirements_docs=[("req-md-summary", markdown)])

        result = analyzer.analyze(test_report, query_for_context="auth release risk")

        self.assertIn("Source bundle summary", result.augmented_prompt)
        self.assertIn("req-md-summary", result.augmented_prompt)

    def test_initialize_corpus_accepts_artifact_bundles(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_refresh">
                <failure type="AssertionError">token refresh failed</failure>
            </testcase>
        </testsuite>"""
        analyzer = RAGAnalyzer()
        indexed = analyzer.initialize_corpus(
            artifact_bundles=[
                ArtifactBundle(
                    source_id="incident:auth",
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    text="Token refresh failures increase under high request concurrency.",
                    tables=[{"rows": [{"component": "auth", "risk": "high"}]}],
                    images=[{"image_path": "artifacts/auth-heatmap.png", "alt_text": "auth retry heatmap"}],
                    metadata={"origin_path": "docs/incidents/auth.md"},
                )
            ]
        )

        self.assertGreaterEqual(indexed, 3)
        result = analyzer.analyze(test_report, query_for_context="auth retry heatmap risk")
        self.assertTrue(any(source.startswith("incident:auth") for source in result.evidence_sources))

    def test_analyze_reports_unavailable_corpus_evidence(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
        </testsuite>"""
        analyzer = RAGAnalyzer()
        analyzer.initialize_corpus(
            requirements_docs=[
                ("req-auth", "Authentication release risk requires stricter negative test coverage."),
            ]
        )

        result = analyzer.analyze(test_report, query_for_context="image table risk matrix for auth")

        self.assertIn("unavailable_evidence", result.risk_assessment)
        unavailable = result.risk_assessment["unavailable_evidence"]
        self.assertIn("modalities", unavailable)
        self.assertIn("table", unavailable["modalities"])
        self.assertIn("image", unavailable["modalities"])

    def test_analyze_reports_focus_confidence_and_prompt_focus_section(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_refresh">
                <failure type="RuntimeError">token refresh failure</failure>
            </testcase>
        </testsuite>"""
        analyzer = RAGAnalyzer()
        analyzer.initialize_corpus(
            requirements_docs=[
                ("req-auth", "Missing auth negative tests and release mitigation steps."),
            ],
            system_analysis_docs=[
                ("sys-auth", "Root cause analysis for auth retry storm and failure cluster."),
            ],
            knowledge_docs=[
                ("kb-auth", "Risk prioritization guidance for release blockers."),
            ],
        )

        result = analyzer.analyze(test_report, query_for_context="auth retry failure")

        self.assertIn("focus_confidence", result.risk_assessment)
        self.assertIn("root_cause", result.risk_assessment["focus_confidence"])
        self.assertIn("test_gap", result.risk_assessment["focus_confidence"])
        self.assertIn("Analysis focus coverage", result.augmented_prompt)

    def test_initialize_corpus_accepts_ingestion_records(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_refresh">
                <failure type="RuntimeError">token refresh failure</failure>
            </testcase>
        </testsuite>"""
        analyzer = RAGAnalyzer()
        indexed = analyzer.initialize_corpus(
            ingestion_records=[
                IngestionRecord(
                    source_id="record:auth-compound",
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    payload={
                        "text": "Auth retry storms are release blocking.",
                        "tables": [{"rows": [{"component": "auth", "risk": "high"}]}],
                        "images": [{"ocr_text": "auth retry heatmap risk matrix"}],
                    },
                )
            ]
        )

        self.assertGreaterEqual(indexed, 3)
        result = analyzer.analyze(test_report, query_for_context="auth retry heatmap risk matrix")
        self.assertTrue(any(source.startswith("record:auth-compound") for source in result.evidence_sources))

    def test_initialize_corpus_records_use_pipeline_when_enabled(self):
        class SpyPipeline:
            def __init__(self):
                self.calls = 0
                self._delegate = UnifiedIngestionPipeline()

            def process_batch(self, documents):
                self.calls += 1
                return self._delegate.process_batch(documents)

        analyzer = RAGAnalyzer()
        spy_pipeline = SpyPipeline()
        indexed = analyzer.initialize_corpus(
            ingestion_records=[
                IngestionRecord(
                    source_id="record:auth-multi",
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    payload={
                        "text": "Auth retry storms are release blocking.",
                        "tables": [{"rows": [{"component": "auth", "risk": "high"}]}],
                    },
                )
            ],
            prefer_pipeline_for_records=True,
            record_pipeline=spy_pipeline,
        )

        self.assertGreaterEqual(indexed, 2)
        self.assertEqual(1, spy_pipeline.calls)
        ingested = [chunk for chunk in analyzer._engine._chunks if chunk.source_id == "record:auth-multi"]
        self.assertGreaterEqual(len(ingested), 1)
        self.assertTrue(any(str(chunk.metadata.get("ingestion_route", "")).startswith("pipeline") for chunk in ingested))

    def test_initialize_corpus_records_fallback_when_pipeline_errors(self):
        class FailingPipeline:
            def __init__(self):
                self.calls = 0

            def process_batch(self, documents):
                self.calls += 1
                raise RuntimeError("pipeline unavailable")

        analyzer = RAGAnalyzer()
        failing_pipeline = FailingPipeline()
        indexed = analyzer.initialize_corpus(
            ingestion_records=[
                IngestionRecord(
                    source_id="record:req-auth-fallback",
                    source_type=SourceType.REQUIREMENTS,
                    payload="Authentication release risk requires stricter negative tests.",
                )
            ],
            prefer_pipeline_for_records=True,
            record_pipeline=failing_pipeline,
        )

        self.assertGreaterEqual(indexed, 1)
        self.assertEqual(1, failing_pipeline.calls)
        ingested = [chunk for chunk in analyzer._engine._chunks if chunk.source_id == "record:req-auth-fallback"]
        self.assertGreaterEqual(len(ingested), 1)
        self.assertTrue(all(chunk.text.strip() for chunk in ingested))

    def test_initialize_corpus_accepts_bundled_ingestion_record_artifacts(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_refresh">
                <failure type="RuntimeError">token refresh failure</failure>
            </testcase>
        </testsuite>"""
        analyzer = RAGAnalyzer()
        indexed = analyzer.initialize_corpus(
            ingestion_records=[
                IngestionRecord(
                    source_id="record:bundle-auth",
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    payload={
                        "artifacts": [
                            {"artifact_id": "summary", "content": "Auth retry storms are release blocking."},
                            {
                                "artifact_id": "risk-table",
                                "modality": "table",
                                "content": {"rows": [{"component": "auth", "risk": "high"}]},
                            },
                            {
                                "artifact_id": "heatmap",
                                "modality": "image",
                                "content": {"image_path": "screens/auth.png", "alt_text": "auth retry heatmap"},
                            },
                        ]
                    },
                )
            ]
        )

        self.assertGreaterEqual(indexed, 3)
        result = analyzer.analyze(test_report, query_for_context="auth retry heatmap risk")
        self.assertTrue(
            any(source.startswith("record:bundle-auth::artifact:") for source in result.evidence_sources)
        )

    def test_initialize_corpus_routes_code_snippet_records_to_code_aware_chunking(self):
        analyzer = RAGAnalyzer(chunker_type=ChunkerType.CODE_AWARE)
        indexed = analyzer.initialize_corpus(
            ingestion_records=[
                IngestionRecord(
                    source_id="snippet:auth.py",
                    source_type=SourceType.CODE_SNIPPET,
                    payload="""
def refresh_token(token: str) -> str:
    if not token:
        raise ValueError("token missing")
    return token.strip()
""".strip(),
                )
            ]
        )

        self.assertGreaterEqual(indexed, 1)
        code_chunks = [chunk for chunk in analyzer._engine._chunks if chunk.source_id == "snippet:auth.py"]
        self.assertGreaterEqual(len(code_chunks), 1)
        self.assertTrue(all(chunk.modality == "code" for chunk in code_chunks))
        self.assertTrue(any("chunk_type" in chunk.metadata for chunk in code_chunks))

    def test_initialize_corpus_from_pipeline_documents_indexes_multimodal_chunks(self):
        analyzer = RAGAnalyzer()
        indexed = analyzer.initialize_corpus_from_pipeline_documents(
            [
                IngestDocument(
                    source_id="pipeline:incident-auth",
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    modality="compound",
                    content={
                        "text": "Auth retry storms increase release risk and require mitigation.",
                        "tables": [{"rows": [{"component": "auth", "severity": "high"}]}],
                        "images": [{"image_path": "artifacts/auth-heatmap.png", "alt_text": "auth retry heatmap"}],
                    },
                )
            ]
        )

        self.assertGreaterEqual(indexed, 3)
        result = analyzer.analyze(
            """<testsuite name="pytest" errors="0" failures="1" tests="2">
                <testcase classname="test_auth" name="test_retry">
                    <failure type="RuntimeError">auth retry failure</failure>
                </testcase>
            </testsuite>""",
            query_for_context="auth retry heatmap risk matrix",
        )
        self.assertTrue(any(source.startswith("pipeline:incident-auth") for source in result.evidence_sources))

    def test_rag_analyze_accepts_ingestion_records(self):
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
        </testsuite>"""

        result = rag_analyze(
            test_report_content=test_report,
            ingestion_records=[
                IngestionRecord(
                    source_id="record:req-auth",
                    source_type=SourceType.REQUIREMENTS,
                    payload="Authentication release risk requires stricter negative tests.",
                )
            ],
            query="authentication release risk",
        )

        self.assertEqual(result.base_result.total_failures, 1)
        self.assertIn("retrieval_confidence", result.risk_assessment)


class TestRAGSeverityAssessment(unittest.TestCase):
    def test_critical_severity_detection(self):
        from src.test_analysis_assistant.rag_analyzer import _assess_severity

        # Security issues are critical
        self.assertEqual(
            _assess_severity("Security vulnerability discovered in auth module", "risk_factor"),
            "critical",
        )
        self.assertEqual(
            _assess_severity("Data loss possible during migration", "risk_factor"),
            "critical",
        )

    def test_high_severity_detection(self):
        from src.test_analysis_assistant.rag_analyzer import _assess_severity

        self.assertEqual(
            _assess_severity("This is a P0 release blocking issue", "risk_factor"),
            "high",
        )
        self.assertEqual(
            _assess_severity("High risk of failure in production", "risk_factor"),
            "high",
        )

    def test_medium_severity_detection(self):
        from src.test_analysis_assistant.rag_analyzer import _assess_severity

        self.assertEqual(
            _assess_severity("Moderate impact on system performance", "test_gap"),
            "medium",
        )

    def test_low_severity_default(self):
        from src.test_analysis_assistant.rag_analyzer import _assess_severity

        self.assertEqual(
            _assess_severity("Some minor documentation issue", "test_gap"),
            "low",
        )


class TestChunkerType(unittest.TestCase):
    def test_chunker_type_enum(self):
        """Test ChunkerType enum values."""
        self.assertEqual(ChunkerType.BASIC.value, "basic")
        self.assertEqual(ChunkerType.CODE_AWARE.value, "code_aware")
        self.assertEqual(ChunkerType.AUTO.value, "auto")


class TestRAGAnalyzerChunkingStrategies(unittest.TestCase):
    def test_rag_analyzer_basic_chunker(self):
        """Test RAGAnalyzer with basic chunker."""
        analyzer = RAGAnalyzer(chunker_type=ChunkerType.BASIC)
        self.assertIsNotNone(analyzer._engine)
        self.assertEqual(analyzer._chunker_type, ChunkerType.BASIC)
        self.assertIsNone(analyzer._code_ingestor)

    def test_rag_analyzer_code_aware_chunker(self):
        """Test RAGAnalyzer with code-aware chunker."""
        analyzer = RAGAnalyzer(chunker_type=ChunkerType.CODE_AWARE)
        self.assertIsNotNone(analyzer._engine)
        self.assertEqual(analyzer._chunker_type, ChunkerType.CODE_AWARE)
        self.assertIsNotNone(analyzer._code_ingestor)

    def test_rag_analyzer_auto_chunker(self):
        """Test RAGAnalyzer with auto chunker selection."""
        analyzer = RAGAnalyzer(chunker_type=ChunkerType.AUTO)
        self.assertIsNotNone(analyzer._engine)
        self.assertEqual(analyzer._chunker_type, ChunkerType.AUTO)
        self.assertIsNotNone(analyzer._code_ingestor)
        self.assertIsNotNone(analyzer._basic_ingestor)

    def test_is_code_file_detection(self):
        """Test file type detection."""
        analyzer = RAGAnalyzer(chunker_type=ChunkerType.AUTO)

        # Code files
        self.assertTrue(analyzer._is_code_file("path/to/file.py"))
        self.assertTrue(analyzer._is_code_file("path/to/file.js"))
        self.assertTrue(analyzer._is_code_file("path/to/file.ts"))
        self.assertTrue(analyzer._is_code_file("path/to/file.java"))

        # Non-code files
        self.assertFalse(analyzer._is_code_file("path/to/file.md"))
        self.assertFalse(analyzer._is_code_file("path/to/file.txt"))
        self.assertFalse(analyzer._is_code_file("path/to/file.json"))

    def test_select_ingestor_auto_mode(self):
        """Test ingestor selection in auto mode."""
        analyzer = RAGAnalyzer(chunker_type=ChunkerType.AUTO)

        # Code file should use code-aware ingestor
        code_ingestor = analyzer._select_ingestor(SourceType.REPOSITORY, "path/to/module.py")
        self.assertEqual(code_ingestor, analyzer._code_ingestor)

        # Non-code file should use basic ingestor
        basic_ingestor = analyzer._select_ingestor(SourceType.REQUIREMENTS, "doc.md")
        self.assertEqual(basic_ingestor, analyzer._basic_ingestor)

    def test_rag_analyze_with_code_aware_chunker(self):
        """Test rag_analyze convenience function with code-aware chunker."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_api" name="test_status">
                <failure type="RuntimeError">Invalid response</failure>
            </testcase>
        </testsuite>"""

        result = rag_analyze(
            test_report_content=test_report,
            requirements_docs=[("req-1", "API should return 200 OK for health checks.")],
            query="API test gaps",
            chunker_type=ChunkerType.CODE_AWARE,
        )

        self.assertIsInstance(result, RAGAnalysisResult)
        self.assertEqual(result.base_result.total_failures, 1)

    def test_rag_analyze_with_test_aware_chunker(self):
        """Test rag_analyze convenience function with test-aware chunker."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Login failed</failure>
            </testcase>
        </testsuite>"""

        result = rag_analyze(
            test_report_content=test_report,
            requirements_docs=[
                ("req-auth", "Authentication must handle invalid credentials."),
                ("test-auth", "Test coverage for login with invalid credentials."),
            ],
            query="auth test gaps",
            chunker_type=ChunkerType.TEST_AWARE,
        )

        self.assertIsInstance(result, RAGAnalysisResult)
        self.assertEqual(result.base_result.total_failures, 1)

    def test_rag_analyzer_test_aware_initialization(self):
        """Test RAGAnalyzer with test-aware chunker initialization."""
        analyzer = RAGAnalyzer(chunker_type=ChunkerType.TEST_AWARE)
        self.assertIsNotNone(analyzer._engine)
        self.assertEqual(analyzer._chunker_type, ChunkerType.TEST_AWARE)
        self.assertIsNotNone(analyzer._test_ingestor)


class TestEnhancedConfidenceScoring(unittest.TestCase):
    def test_compute_enhanced_confidence_basic(self):
        """Test basic enhanced confidence computation."""
        chunk = Chunk(
            chunk_id="test-1",
            source_id="test-source",
            source_type=SourceType.REQUIREMENTS,
            modality="text",
            text="This is a test chunk with important content.",
            token_count=10,
            metadata={"unit_index": 0, "start_line": 1, "extraction_confidence": 0.9},
        )

        query_tokens = ["test", "important", "content"]
        matched_terms = ["test", "important"]
        score_breakdown = {"lexical": 0.5, "semantic": 0.3}

        confidence = compute_enhanced_confidence(
            chunk=chunk,
            query_tokens=query_tokens,
            matched_terms=matched_terms,
            score_breakdown=score_breakdown,
            rank=0,
            total_results=5,
        )

        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_compute_enhanced_confidence_no_matches(self):
        """Test confidence with no matched terms."""
        chunk = Chunk(
            chunk_id="test-2",
            source_id="test-source",
            source_type=SourceType.KNOWLEDGE,
            modality="text",
            text="Unrelated content here.",
            token_count=5,
            metadata={"unit_index": 5, "start_line": 100},
        )

        query_tokens = ["test", "important"]
        matched_terms = []
        score_breakdown = {"lexical": 0.0, "semantic": 0.0}

        confidence = compute_enhanced_confidence(
            chunk=chunk,
            query_tokens=query_tokens,
            matched_terms=matched_terms,
            score_breakdown=score_breakdown,
            rank=2,
            total_results=5,
        )

        self.assertEqual(confidence, 0.0)

    def test_compute_enhanced_confidence_code_chunk(self):
        """Test confidence computation for code chunks."""
        chunk = Chunk(
            chunk_id="test-3",
            source_id="path/to/module.py",
            source_type=SourceType.CODE_SNIPPET,
            modality="code",
            text="def authenticate(user): return user.is_valid",
            token_count=8,
            metadata={
                "unit_index": 0,
                "start_line": 10,
                "extraction_confidence": 0.95,
                "unit_type": "function",
                "unit_name": "authenticate",
            },
        )

        query_tokens = ["authenticate", "function", "user"]
        matched_terms = ["authenticate", "function"]
        score_breakdown = {"lexical": 0.6, "semantic": 0.4}

        confidence = compute_enhanced_confidence(
            chunk=chunk,
            query_tokens=query_tokens,
            matched_terms=matched_terms,
            score_breakdown=score_breakdown,
            rank=0,
            total_results=3,
        )

        # Code chunks with good metadata should have high confidence
        self.assertGreater(confidence, 0.3)


class TestQueryReformulationIntegration(unittest.TestCase):
    """Tests for query reformulation integration in RAG analyzer."""

    def test_rag_analyzer_has_reformulator(self):
        """Test that RAGAnalyzer has a query reformulator."""
        analyzer = RAGAnalyzer()
        self.assertIsNotNone(analyzer._reformulator)
        from src.test_analysis_assistant.query_reformulator import QueryReformulator
        self.assertIsInstance(analyzer._reformulator, QueryReformulator)

    def test_analyze_generates_reformulated_queries(self):
        """Test that analyze generates reformulated queries from failures."""
        test_report = """<testsuite name="pytest" errors="0" failures="2" tests="5">
            <testcase classname="test_auth" name="test_login_failure">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
            <testcase classname="test_auth" name="test_token_refresh">
                <failure type="RuntimeError">token refresh failed</failure>
            </testcase>
        </testsuite>"""

        analyzer = RAGAnalyzer()
        analyzer.add_knowledge(
            "auth-module",
            "Authentication module handles token refresh and login validation.",
        )

        result = analyzer.analyze(test_report)

        # Check reformulated queries were stored
        self.assertIsNotNone(analyzer._reformulated_queries)
        self.assertGreater(len(analyzer._reformulated_queries.variants), 0)

        # Verify queries include extracted symbols from test names
        query_texts = [v.query_text for v in analyzer._reformulated_queries.variants]
        combined_queries = " ".join(query_texts).lower()

        # Should contain symbols from test names
        self.assertTrue(
            "login" in combined_queries or "test" in combined_queries or "auth" in combined_queries,
            f"Expected login/auth in queries, got: {query_texts}",
        )

    def test_reformulated_queries_have_weights(self):
        """Test that reformulated queries have associated weights."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_api" name="test_status_check">
                <failure type="TypeError">invalid type</failure>
            </testcase>
        </testsuite>"""

        analyzer = RAGAnalyzer()
        analyzer.add_knowledge("api-doc", "API status endpoint documentation.")
        result = analyzer.analyze(test_report)

        # Verify variants have weights
        self.assertIsNotNone(analyzer._reformulated_queries)
        for variant in analyzer._reformulated_queries.variants:
            self.assertGreater(variant.weight, 0.0)
            self.assertLessEqual(variant.weight, 1.0)

    def test_reformulated_queries_include_original(self):
        """Test that original query is included in reformulated variants."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_x" name="test_y">
                <failure type="Error">failed</failure>
            </testcase>
        </testsuite>"""

        analyzer = RAGAnalyzer()
        analyzer.add_knowledge("doc", "content")
        result = analyzer.analyze(test_report, query_for_context="custom query about auth")

        # Original query should be included
        self.assertIsNotNone(analyzer._reformulated_queries)
        original_queries = [
            v.query_text for v in analyzer._reformulated_queries.variants
            if v.intent == "original"
        ]
        self.assertTrue(
            any("custom query" in q.lower() for q in original_queries),
            f"Expected original query in variants, got: {original_queries}",
        )

    def test_reformulator_extracts_error_specific_queries(self):
        """Test that reformulator generates error-specific query variants."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_db" name="test_connection">
                <failure type="ModuleNotFoundError">No module named 'psycopg2'</failure>
            </testcase>
        </testsuite>"""

        analyzer = RAGAnalyzer()
        analyzer.add_knowledge("db-doc", "Database connection module.")
        result = analyzer.analyze(test_report)

        self.assertIsNotNone(analyzer._reformulated_queries)
        query_texts = " ".join(
            v.query_text for v in analyzer._reformulated_queries.variants
        ).lower()

        # Should include import-related queries
        self.assertTrue(
            "import" in query_texts or "module" in query_texts or "dependency" in query_texts,
            f"Expected import-related queries, got: {query_texts}",
        )

    def test_rag_analyze_preserves_functionality(self):
        """Test that rag_analyze convenience function works with reformulation."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Login failed</failure>
            </testcase>
        </testsuite>"""

        result = rag_analyze(
            test_report_content=test_report,
            requirements_docs=[("req-1", "Authentication must work correctly.")],
            query="auth test gaps",
        )

        self.assertIsInstance(result, RAGAnalysisResult)
        self.assertEqual(result.base_result.total_failures, 1)
        # Should still have retrieval insights
        self.assertTrue(len(result.retrieval_insights) > 0 or len(result.test_gap_analysis) > 0)


if __name__ == "__main__":
    unittest.main()
