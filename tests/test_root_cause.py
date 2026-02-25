"""Tests for root cause hypothesis generation module."""

import unittest

from src.test_analysis_assistant.analyzer import analyze_report_text
from src.test_analysis_assistant.root_cause import (
    HypothesisConfidence,
    RootCauseAnalyzer,
    RootCauseAnalysis,
    RootCauseCategory,
    RootCauseHypothesis,
    generate_root_cause_hypotheses,
)


class TestRootCauseHypothesis(unittest.TestCase):
    """Test the RootCauseHypothesis dataclass."""

    def test_hypothesis_creation(self):
        hypothesis = RootCauseHypothesis(
            hypothesis_id="H001",
            category=RootCauseCategory.CODE_DEFECT,
            description="Test hypothesis",
            confidence=0.85,
            confidence_level=HypothesisConfidence.HIGH,
            evidence=["evidence1"],
            supporting_failures=["test_foo"],
            related_code_locations=["src/foo.py"],
            remediation_suggestions=["Fix the bug"],
            priority=1,
        )

        self.assertEqual(hypothesis.hypothesis_id, "H001")
        self.assertEqual(hypothesis.category, RootCauseCategory.CODE_DEFECT)
        self.assertEqual(hypothesis.confidence_level, HypothesisConfidence.HIGH)


class TestRootCauseCategory(unittest.TestCase):
    """Test RootCauseCategory enum."""

    def test_category_values(self):
        self.assertEqual(RootCauseCategory.CODE_DEFECT.value, "code_defect")
        self.assertEqual(RootCauseCategory.DEPENDENCY.value, "dependency")
        self.assertEqual(RootCauseCategory.ENVIRONMENT.value, "environment")
        self.assertEqual(RootCauseCategory.TEST_DEFECT.value, "test_defect")


class TestRootCauseAnalyzer(unittest.TestCase):
    """Test the RootCauseAnalyzer class."""

    def test_analyzer_initialization(self):
        analyzer = RootCauseAnalyzer()
        self.assertIsNotNone(analyzer)

    def test_analyze_with_no_failures(self):
        # Create an analysis result with no failures
        from src.test_analysis_assistant.models import AnalysisResult

        result = AnalysisResult(
            input_format="pytest",
            total_failures=0,
            failures=[],
            clusters=[],
            root_cause_hypotheses=[],
            fix_suggestions=[],
        )

        analyzer = RootCauseAnalyzer()
        analysis = analyzer.analyze(result)

        self.assertIsInstance(analysis, RootCauseAnalysis)
        self.assertEqual(analysis.total_failures_analyzed, 0)
        self.assertEqual(len(analysis.hypotheses), 0)
        self.assertEqual(analysis.overall_confidence, 0.0)

    def test_analyze_with_module_not_found_error(self):
        """Test analysis of ModuleNotFoundError - should identify as dependency issue."""
        test_report = """<testsuite name="pytest" errors="0" failures="2" tests="5">
            <testcase classname="test_auth" name="test_login">
                <failure type="ModuleNotFoundError">No module named 'requests'</failure>
            </testcase>
            <testcase classname="test_api" name="test_call">
                <failure type="ModuleNotFoundError">No module named 'requests'</failure>
            </testcase>
        </testsuite>"""

        result = analyze_report_text(test_report)
        analyzer = RootCauseAnalyzer()
        analysis = analyzer.analyze(result)

        self.assertIsInstance(analysis, RootCauseAnalysis)
        self.assertEqual(analysis.total_failures_analyzed, 2)

        # Should have identified dependency issue
        dep_hypotheses = [h for h in analysis.hypotheses if h.category == RootCauseCategory.DEPENDENCY]
        self.assertGreater(len(dep_hypotheses), 0)

    def test_analyze_with_type_error(self):
        """Test analysis of TypeError - should identify as code defect."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="3">
            <testcase classname="test_calc" name="test_add">
                <failure type="TypeError">unsupported operand type(s) for +: 'int' and 'str'</failure>
            </testcase>
        </testsuite>"""

        result = analyze_report_text(test_report)
        analyzer = RootCauseAnalyzer()
        analysis = analyzer.analyze(result)

        self.assertIsInstance(analysis, RootCauseAnalysis)
        self.assertEqual(analysis.total_failures_analyzed, 1)

        # Should have identified code defect
        code_defects = [h for h in analysis.hypotheses if h.category == RootCauseCategory.CODE_DEFECT]
        self.assertGreater(len(code_defects), 0)

    def test_analyze_with_timeout_error(self):
        """Test analysis of TimeoutError - should identify as timing issue."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_integration" name="test_slow_call">
                <failure type="TimeoutError">Operation timed out after 30 seconds</failure>
            </testcase>
        </testsuite>"""

        result = analyze_report_text(test_report)
        analyzer = RootCauseAnalyzer()
        analysis = analyzer.analyze(result)

        self.assertIsInstance(analysis, RootCauseAnalysis)

        # Should have identified timing issue
        timing_issues = [h for h in analysis.hypotheses if h.category == RootCauseCategory.TIMING_ISSUE]
        self.assertGreater(len(timing_issues), 0)

    def test_confidence_levels(self):
        """Test confidence level assignment."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_x" name="test_y">
                <failure type="SyntaxError">invalid syntax</failure>
            </testcase>
        </testsuite>"""

        result = analyze_report_text(test_report)
        analyzer = RootCauseAnalyzer()
        analysis = analyzer.analyze(result)

        # SyntaxError should have high confidence
        syntax_hypotheses = [h for h in analysis.hypotheses if h.category == RootCauseCategory.CODE_DEFECT]
        for h in syntax_hypotheses:
            self.assertIn(h.confidence_level, [HypothesisConfidence.HIGH, HypothesisConfidence.MEDIUM])

    def test_remediation_suggestions(self):
        """Test that remediation suggestions are generated."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_auth" name="test_login">
                <failure type="ModuleNotFoundError">No module named 'jwt'</failure>
            </testcase>
        </testsuite>"""

        result = analyze_report_text(test_report)
        analyzer = RootCauseAnalyzer()
        analysis = analyzer.analyze(result)

        # Should have remediation suggestions
        for hypothesis in analysis.hypotheses:
            if hypothesis.category == RootCauseCategory.DEPENDENCY:
                self.assertGreater(len(hypothesis.remediation_suggestions), 0)
                break

    def test_priority_assignment(self):
        """Test that priorities are assigned correctly."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_x" name="test_y">
                <failure type="SyntaxError">invalid syntax</failure>
            </testcase>
        </testsuite>"""

        result = analyze_report_text(test_report)
        analyzer = RootCauseAnalyzer()
        analysis = analyzer.analyze(result)

        # Primary hypothesis should have priority
        if analysis.primary_hypothesis:
            self.assertLess(analysis.primary_hypothesis.priority, 5)

    def test_context_chunks_included(self):
        """Test that context chunks are used in evidence."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_x" name="test_y">
                <failure type="TypeError">error</failure>
            </testcase>
        </testsuite>"""

        result = analyze_report_text(test_report)
        analyzer = RootCauseAnalyzer()

        context = ["TypeError occurs in data processing", "Check type conversions"]
        analysis = analyzer.analyze(result, context_chunks=context)

        # Evidence should include context
        for hypothesis in analysis.hypotheses:
            self.assertGreater(len(hypothesis.evidence), 0)


class TestGenerateRootCauseHypotheses(unittest.TestCase):
    """Test the convenience function."""

    def test_generate_function(self):
        """Test the generate_root_cause_hypotheses convenience function."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_x" name="test_y">
                <failure type="RuntimeError">Something went wrong</failure>
            </testcase>
        </testsuite>"""

        result = analyze_report_text(test_report)
        analysis = generate_root_cause_hypotheses(result)

        self.assertIsInstance(analysis, RootCauseAnalysis)
        self.assertEqual(analysis.total_failures_analyzed, 1)

    def test_generate_with_context(self):
        """Test with context chunks."""
        test_report = """<testsuite name="pytest" errors="0" failures="1" tests="2">
            <testcase classname="test_x" name="test_y">
                <failure type="ValueError">invalid value</failure>
            </testcase>
        </testsuite>"""

        result = analyze_report_text(test_report)
        analysis = generate_root_cause_hypotheses(
            result,
            context_chunks=["ValueError indicates validation issue"],
        )

        self.assertIsInstance(analysis, RootCauseAnalysis)


class TestRootCauseAnalysis(unittest.TestCase):
    """Test the RootCauseAnalysis dataclass."""

    def test_analysis_creation(self):
        analysis = RootCauseAnalysis(
            analysis_id="rca-001",
            total_failures_analyzed=5,
            hypotheses=[],
            primary_hypothesis=None,
            overall_confidence=0.75,
            evidence_summary={"by_category": {"code_defect": 2}},
        )

        self.assertEqual(analysis.analysis_id, "rca-001")
        self.assertEqual(analysis.total_failures_analyzed, 5)
        self.assertEqual(analysis.overall_confidence, 0.75)


if __name__ == "__main__":
    unittest.main()
