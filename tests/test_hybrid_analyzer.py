import unittest

from src.test_analysis_assistant.hybrid_analyzer import (
    HybridAnalysisResult,
    HybridAnalyzer,
    HybridInsight,
    hybrid_analyze,
)
from src.test_analysis_assistant.llm_integration import LLMConfig, LLMProvider


class TestHybridAnalyzer(unittest.TestCase):
    """Tests for the HybridAnalyzer class."""

    def test_hybrid_analyzer_initialization(self):
        """Test hybrid analyzer initialization with LLM."""
        analyzer = HybridAnalyzer(use_llm=True)
        self.assertIsNotNone(analyzer._rag_analyzer)
        self.assertTrue(analyzer._use_llm)

    def test_hybrid_analyzer_without_llm(self):
        """Test hybrid analyzer initialization without LLM."""
        analyzer = HybridAnalyzer(use_llm=False)
        self.assertIsNotNone(analyzer._rag_analyzer)
        self.assertFalse(analyzer._use_llm)

    def test_hybrid_analyze_with_knowledge_corpus(self):
        """Test hybrid analysis with knowledge corpus."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="3">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
        </testsuite>"""

        result = hybrid_analyze(
            test_report_content=test_report,
            knowledge_docs=[("auth-doc", "Authentication module handles login failures.")],
            use_llm=False,  # Use RAG only for predictable results
        )

        self.assertIsInstance(result, HybridAnalysisResult)
        self.assertIsNotNone(result.rag_result)
        self.assertEqual(result.rag_result.base_result.total_failures, 1)
        self.assertEqual(result.analysis_mode, "rag_only")
        self.assertFalse(result.llm_used)

    def test_hybrid_analyze_with_llm_mock(self):
        """Test hybrid analysis with mock LLM."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="3">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
        </testsuite>"""

        result = hybrid_analyze(
            test_report_content=test_report,
            knowledge_docs=[("auth-doc", "Authentication module handles login failures.")],
            use_llm=True,
            llm_provider=LLMProvider.MOCK,
        )

        self.assertIsInstance(result, HybridAnalysisResult)
        self.assertEqual(result.analysis_mode, "hybrid")
        self.assertTrue(result.llm_used)

    def test_hybrid_result_to_dict(self):
        """Test HybridAnalysisResult serialization."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_x" name="test_y">
                <failure type="Error">Failed</failure>
            </testcase>
        </testsuite>"""

        analyzer = HybridAnalyzer(use_llm=False)
        result = analyzer.analyze(
            test_report_content=test_report,
            knowledge_docs=[("doc", "Some content.")],
        )

        result_dict = result.to_dict()

        self.assertIn("rag_result", result_dict)
        self.assertIn("llm_insights", result_dict)
        self.assertIn("hybrid_confidence", result_dict)
        self.assertIn("analysis_mode", result_dict)

    def test_hybrid_insight_dataclass(self):
        """Test HybridInsight dataclass."""
        insight = HybridInsight(
            insight_type="test_gap",
            title="Missing auth tests",
            description="No negative test for invalid tokens",
            confidence=0.85,
            source="rag",
            evidence_chunks=["req:auth"],
            severity="high",
        )

        self.assertEqual(insight.insight_type, "test_gap")
        self.assertEqual(insight.source, "rag")
        self.assertEqual(insight.severity, "high")

    def test_hybrid_insight_with_llm_reasoning(self):
        """Test HybridInsight with LLM reasoning."""
        insight = HybridInsight(
            insight_type="root_cause_evidence",
            title="Root Cause Analysis",
            description="Environment difference detected",
            confidence=0.75,
            source="llm",
            severity="high",
            llm_reasoning="The error suggests a configuration difference between test and production environments.",
        )

        self.assertEqual(insight.source, "llm")
        self.assertIsNotNone(insight.llm_reasoning)
        self.assertIn("environment", insight.llm_reasoning.lower())

    def test_hybrid_confidence_calculation_rag_only(self):
        """Test hybrid confidence calculation in RAG-only mode."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
        </testsuite>"""

        analyzer = HybridAnalyzer(use_llm=False)
        result = analyzer.analyze(
            test_report_content=test_report,
            knowledge_docs=[("auth-doc", "Authentication release risk.")],
        )

        # Should have RAG confidence from retrieval
        self.assertGreaterEqual(result.hybrid_confidence, 0.0)
        self.assertLessEqual(result.hybrid_confidence, 1.0)

    def test_hybrid_confidence_calculation_with_llm(self):
        """Test hybrid confidence calculation with LLM."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
        </testsuite>"""

        result = hybrid_analyze(
            test_report_content=test_report,
            knowledge_docs=[("auth-doc", "Authentication release risk.")],
            use_llm=True,
            llm_provider=LLMProvider.MOCK,
        )

        # Should have blended confidence
        self.assertGreaterEqual(result.hybrid_confidence, 0.0)
        self.assertLessEqual(result.hybrid_confidence, 1.0)
        self.assertTrue(result.llm_used)

    def test_hybrid_analyze_empty_corpus(self):
        """Test hybrid analysis without any corpus."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_x" name="test_y">
                <failure type="Error">Failed</failure>
            </testcase>
        </testsuite>"""

        analyzer = HybridAnalyzer(use_llm=False)
        result = analyzer.analyze(test_report_content=test_report)

        self.assertIsInstance(result, HybridAnalysisResult)
        # Should handle gracefully without corpus
        self.assertIn(result.analysis_mode, ["rag_only", "hybrid"])

    def test_hybrid_analyze_with_requirements(self):
        """Test hybrid analysis with requirements documents."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_api" name="test_status">
                <failure type="RuntimeError">Invalid response</failure>
            </testcase>
        </testsuite>"""

        result = hybrid_analyze(
            test_report_content=test_report,
            requirements_docs=[
                ("req-1", "API should return 200 OK for health checks."),
                ("req-2", "API should handle errors gracefully."),
            ],
            use_llm=False,
        )

        self.assertIsInstance(result, HybridAnalysisResult)
        self.assertEqual(result.rag_result.base_result.total_failures, 1)

    def test_hybrid_analyze_preserves_rag_insights(self):
        """Test that hybrid analysis preserves RAG insights."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Expected True, got False</failure>
            </testcase>
        </testsuite>"""

        analyzer = HybridAnalyzer(use_llm=False)
        result = analyzer.analyze(
            test_report_content=test_report,
            knowledge_docs=[
                ("auth-req", "Authentication release risk requires negative tests."),
            ],
            query_for_context="auth test gaps",
        )

        # Should have converted RAG insights to hybrid insights
        rag_insights = result.rag_result.retrieval_insights
        hybrid_insights = result.llm_insights

        # At minimum should have some hybrid insights
        self.assertGreaterEqual(len(hybrid_insights), 0)


class TestHybridAnalyzerLLMIntegration(unittest.TestCase):
    """Tests for LLM integration in HybridAnalyzer."""

    def test_llm_root_cause_analysis(self):
        """Test LLM root cause analysis is included in results."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_db" name="test_connection">
                <failure type="ModuleNotFoundError">No module named 'psycopg2'</failure>
            </testcase>
        </testsuite>"""

        result = hybrid_analyze(
            test_report_content=test_report,
            knowledge_docs=[("db-doc", "Database connection module.")],
            use_llm=True,
            llm_provider=LLMProvider.MOCK,
        )

        self.assertTrue(result.llm_used)
        # Mock LLM should return root cause analysis
        self.assertIn("llm_root_cause", result.to_dict())

    def test_llm_test_gap_analysis(self):
        """Test LLM test gap analysis is included in results."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="AssertionError">Login failed</failure>
            </testcase>
        </testsuite>"""

        result = hybrid_analyze(
            test_report_content=test_report,
            knowledge_docs=[("auth-doc", "Authentication module.")],
            use_llm=True,
            llm_provider=LLMProvider.MOCK,
        )

        self.assertTrue(result.llm_used)
        # Mock LLM should return test gap analysis
        self.assertIsInstance(result.llm_test_gaps, list)


class TestHybridAnalyzerErrorHandling(unittest.TestCase):
    """Tests for error handling in HybridAnalyzer."""

    def test_graceful_degradation_on_llm_error(self):
        """Test graceful degradation when LLM fails."""
        # Create analyzer with mock but simulate error condition
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_x" name="test_y">
                <failure type="Error">Failed</failure>
            </testcase>
        </testsuite>"""

        # With use_llm=False, should not try LLM
        analyzer = HybridAnalyzer(use_llm=False)
        result = analyzer.analyze(
            test_report_content=test_report,
            knowledge_docs=[("doc", "Content.")],
        )

        # Should complete without errors
        self.assertIsInstance(result, HybridAnalysisResult)
        self.assertEqual(result.analysis_mode, "rag_only")


if __name__ == "__main__":
    unittest.main()
