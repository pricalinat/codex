import unittest

from src.test_analysis_assistant.rag_analyzer import (
    ChunkerType,
    RAGAnalyzer,
    RAGAnalysisResult,
    RetrievalInsight,
    rag_analyze,
)
from src.test_analysis_assistant.retrieval import (
    Chunk,
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


if __name__ == "__main__":
    unittest.main()
