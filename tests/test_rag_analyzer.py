import unittest

from src.test_analysis_assistant.rag_analyzer import (
    RAGAnalyzer,
    RAGAnalysisResult,
    RetrievalInsight,
    rag_analyze,
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


if __name__ == "__main__":
    unittest.main()
