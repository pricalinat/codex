import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.test_analysis_assistant.analyzer import analyze_report_text
from src.test_analysis_assistant.cli import main
from src.test_analysis_assistant.parsers import (
    detect_format,
    parse_junit_xml,
    parse_pytest_text,
)
from src.test_analysis_assistant.rag_analyzer import (
    RAGAnalyzer,
    RequirementTrace,
    TestGapAnalysis,
    rag_analyze,
)
from src.test_analysis_assistant.retrieval import (
    DummyEmbeddingProvider,
    HybridRetrievalEngine,
    RetrievalEngine,
    SourceType,
    TFIDFEmbeddingProvider,
    create_hybrid_engine,
)


SAMPLE_JUNIT_XML = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<testsuite name=\"pytest\" tests=\"3\" failures=\"2\" errors=\"0\">
  <testcase classname=\"tests.test_math\" name=\"test_add\" file=\"tests/test_math.py\" />
  <testcase classname=\"tests.test_math\" name=\"test_sub\" file=\"tests/test_math.py\">
    <failure type=\"AssertionError\" message=\"assert 1 == 2\">Traceback line</failure>
  </testcase>
  <testcase classname=\"tests.test_api\" name=\"test_call\" file=\"tests/test_api.py\">
    <failure type=\"TypeError\" message=\"bad args\">TypeError details</failure>
  </testcase>
</testsuite>
"""


SAMPLE_PYTEST_TEXT = """============================= test session starts ==============================
_________________________________ tests/test_a.py::test_one __________________________________
    def test_one():
>       assert 1 == 2
E       AssertionError: assert 1 == 2

tests/test_a.py:10: AssertionError

_________________________________ tests/test_b.py::test_two __________________________________
    def test_two():
>       import missing
E       ModuleNotFoundError: No module named 'missing'

tests/test_b.py:22: ModuleNotFoundError
=========================== short test summary info ===========================
FAILED tests/test_a.py::test_one - AssertionError
FAILED tests/test_b.py::test_two - ModuleNotFoundError
"""


class TestTestAnalysisAssistant(unittest.TestCase):
    def test_detect_format_junit_xml(self):
        self.assertEqual(detect_format(SAMPLE_JUNIT_XML), "junit_xml")

    def test_detect_format_pytest_text(self):
        self.assertEqual(detect_format(SAMPLE_PYTEST_TEXT), "pytest_text")

    def test_parse_junit_xml_collects_failures(self):
        failures = parse_junit_xml(SAMPLE_JUNIT_XML)
        self.assertEqual(len(failures), 2)
        self.assertEqual(failures[0].error_type, "AssertionError")

    def test_parse_junit_xml_invalid_xml_raises(self):
        with self.assertRaises(ValueError):
            parse_junit_xml("<testsuite>")

    def test_parse_pytest_text_collects_failures(self):
        failures = parse_pytest_text(SAMPLE_PYTEST_TEXT)
        self.assertEqual(len(failures), 2)
        self.assertEqual(failures[1].error_type, "ModuleNotFoundError")

    def test_analyzer_returns_clusters(self):
        result = analyze_report_text(SAMPLE_PYTEST_TEXT)
        self.assertEqual(result.total_failures, 2)
        self.assertEqual(len(result.clusters), 2)

    def test_analyzer_builds_hypotheses(self):
        result = analyze_report_text(SAMPLE_JUNIT_XML)
        self.assertTrue(result.root_cause_hypotheses)
        self.assertIn("C01", result.root_cause_hypotheses[0])

    def test_analyzer_prioritizes_module_not_found_as_p0(self):
        result = analyze_report_text(SAMPLE_PYTEST_TEXT)
        suggestions = {item.title: item.priority for item in result.fix_suggestions}
        self.assertEqual(suggestions["Address ModuleNotFoundError cluster"], "P0")

    def test_analyzer_empty_input_raises(self):
        with self.assertRaises(ValueError):
            analyze_report_text("   ")

    def test_cli_analyze_json_success(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "report.txt"
            path.write_text(SAMPLE_PYTEST_TEXT, encoding="utf-8")
            buf = io.StringIO()
            with patch("sys.argv", ["prog", "analyze", "--input", str(path)]):
                with redirect_stdout(buf):
                    code = main()
            self.assertEqual(code, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["total_failures"], 2)

    def test_cli_missing_file_returns_1(self):
        buf = io.StringIO()
        with patch("sys.argv", ["prog", "analyze", "--input", "not-exist.txt"]):
            with redirect_stdout(buf):
                code = main()
        self.assertEqual(code, 1)
        self.assertIn("input file not found", buf.getvalue())

    def test_cli_no_subcommand_returns_2(self):
        buf = io.StringIO()
        with patch("sys.argv", ["prog"]):
            with redirect_stdout(buf):
                code = main()
        self.assertEqual(code, 2)


class TestHybridRetrieval(unittest.TestCase):
    """Tests for hybrid retrieval engine."""

    def test_create_hybrid_engine(self):
        engine = create_hybrid_engine(chunk_size=200, lexical_weight=0.6)
        self.assertIsInstance(engine, HybridRetrievalEngine)

    def test_dummy_embedding_provider(self):
        provider = DummyEmbeddingProvider()
        vectors = provider.encode(["hello world test"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(len(vectors[0]), 64)

    def test_tfidf_embedding_provider_basic(self):
        provider = TFIDFEmbeddingProvider(min_df=1, max_features=128)
        vectors = provider.encode(["hello world test", "test analysis report"])
        self.assertEqual(len(vectors), 2)
        # TF-IDF vectors have variable dimensions based on vocabulary
        self.assertGreater(len(vectors[0]), 0)

    def test_tfidf_embedding_provider_similarity(self):
        provider = TFIDFEmbeddingProvider(min_df=1, max_features=128)
        texts = ["test failure analysis", "test gap detection", "unrelated content"]
        vectors = provider.encode(texts)
        # Similar texts should have higher cosine similarity
        from src.test_analysis_assistant.retrieval import _cosine_similarity
        sim_fails = _cosine_similarity(vectors[0], vectors[1])
        sim_unrelated = _cosine_similarity(vectors[0], vectors[2])
        self.assertGreater(sim_fails, sim_unrelated)

    def test_hybrid_engine_ingests_and_queries(self):
        engine = create_hybrid_engine(chunk_size=100, lexical_weight=0.5)
        # Ingest a simple document
        from src.test_analysis_assistant.retrieval import IngestDocument

        doc = IngestDocument(
            source_id="test:doc1",
            source_type=SourceType.CODE_SNIPPET,
            content="def add(a, b): return a + b",
        )
        chunks = engine.ingest_documents([doc])
        self.assertGreater(len(chunks), 0)

        # Query should return results
        results = engine.query("add function", top_k=3, use_hybrid=True)
        self.assertLessEqual(len(results), 3)

    def test_lexical_only_mode(self):
        engine = create_hybrid_engine(chunk_size=100)
        from src.test_analysis_assistant.retrieval import IngestDocument

        doc = IngestDocument(
            source_id="test:doc2",
            source_type=SourceType.CODE_SNIPPET,
            content="def multiply(x, y): return x * y",
        )
        engine.ingest_documents([doc])

        # Query with hybrid disabled
        results = engine.query("multiply", top_k=3, use_hybrid=False)
        self.assertGreater(len(results), 0)


class TestRAGAnalyzer(unittest.TestCase):
    """Tests for RAG-augmented analysis."""

    def test_rag_analyzer_with_hybrid_enabled(self):
        analyzer = RAGAnalyzer(use_hybrid=True, lexical_weight=0.5)
        self.assertIsInstance(analyzer._engine, HybridRetrievalEngine)

    def test_rag_analyzer_with_hybrid_disabled(self):
        analyzer = RAGAnalyzer(use_hybrid=False)
        self.assertIsInstance(analyzer._engine, RetrievalEngine)
        self.assertNotIsInstance(analyzer._engine, HybridRetrievalEngine)

    def test_rag_analyze_structured_gaps(self):
        result = rag_analyze(
            test_report_content=SAMPLE_PYTEST_TEXT,
            use_hybrid=False,
        )
        # Should have structured gaps
        self.assertIsNotNone(result.structured_gaps)
        self.assertIsInstance(result.structured_gaps, list)

    def test_rag_analyze_requirement_traces(self):
        # Add some requirements for tracing
        requirements = [("req:001", "# Requirement 1\nAdd function must work")]
        result = rag_analyze(
            test_report_content=SAMPLE_PYTEST_TEXT,
            requirements_docs=requirements,
            use_hybrid=False,
        )
        # Should have requirement traces
        self.assertIsNotNone(result.requirement_traces)
        self.assertIsInstance(result.requirement_traces, list)

    def test_rag_analyze_returns_augmented_prompt(self):
        # Provide requirements to ensure corpus is initialized
        requirements = [("req:test", "# Test requirement\nAdd numbers")]
        result = rag_analyze(
            test_report_content=SAMPLE_PYTEST_TEXT,
            requirements_docs=requirements,
            use_hybrid=False,
        )
        self.assertIsInstance(result.augmented_prompt, str)
        self.assertGreater(len(result.augmented_prompt), 0)

    def test_rag_analyze_to_dict_includes_new_fields(self):
        requirements = [("req:test", "# Test requirement")]
        result = rag_analyze(
            test_report_content=SAMPLE_PYTEST_TEXT,
            requirements_docs=requirements,
            use_hybrid=False,
        )
        d = result.to_dict()
        self.assertIn("structured_gaps", d)
        self.assertIn("requirement_traces", d)


class TestTestGapAnalysis(unittest.TestCase):
    """Tests for structured test gap analysis."""

    def test_test_gap_dataclass(self):
        gap = TestGapAnalysis(
            gap_id="G001",
            gap_type="missing_test",
            description="Missing test for edge case",
            severity="high",
            related_requirements=["req:001"],
            suggested_test_count=2,
            confidence=0.8,
        )
        self.assertEqual(gap.gap_id, "G001")
        self.assertEqual(gap.gap_type, "missing_test")

    def test_requirement_trace_dataclass(self):
        trace = RequirementTrace(
            requirement_id="req:001",
            requirement_text="Add two numbers",
            covered_by_tests=["test_add_positive", "test_add_negative"],
            gap_type="partially_covered",
            coverage_confidence=0.7,
            related_failures=["test_add_error"],
        )
        self.assertEqual(trace.requirement_id, "req:001")
        self.assertEqual(len(trace.covered_by_tests), 2)


class TestActionablePlan(unittest.TestCase):
    """Tests for actionable plan generation."""

    def test_generate_actionable_plan_basic(self):
        from src.test_analysis_assistant.actionable_plan import (
            ActionablePlan,
            generate_actionable_plan,
        )

        result = rag_analyze(
            test_report_content=SAMPLE_PYTEST_TEXT,
            use_hybrid=False,
        )
        plan = generate_actionable_plan(result)
        self.assertIsInstance(plan, ActionablePlan)
        self.assertIsNotNone(plan.title)
        self.assertIsNotNone(plan.summary)
        self.assertGreater(len(plan.steps), 0)

    def test_actionable_plan_has_confidence(self):
        from src.test_analysis_assistant.actionable_plan import generate_actionable_plan

        requirements = [("req:test", "# Test requirement")]
        result = rag_analyze(
            test_report_content=SAMPLE_PYTEST_TEXT,
            requirements_docs=requirements,
            use_hybrid=False,
        )
        plan = generate_actionable_plan(result)
        self.assertGreaterEqual(plan.overall_confidence, 0.0)
        self.assertLessEqual(plan.overall_confidence, 1.0)

    def test_actionable_plan_has_risk_level(self):
        from src.test_analysis_assistant.actionable_plan import generate_actionable_plan

        result = rag_analyze(
            test_report_content=SAMPLE_PYTEST_TEXT,
            use_hybrid=False,
        )
        plan = generate_actionable_plan(result)
        self.assertIn(plan.risk_level, ["low", "medium", "high", "critical"])

    def test_build_plan_prompt(self):
        from src.test_analysis_assistant.actionable_plan import (
            ActionableStep,
            ActionablePlan,
            build_plan_prompt,
            generate_actionable_plan,
        )

        result = rag_analyze(
            test_report_content=SAMPLE_PYTEST_TEXT,
            use_hybrid=False,
        )
        plan = generate_actionable_plan(result)
        prompt = build_plan_prompt(plan)
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        self.assertIn(plan.title, prompt)


if __name__ == "__main__":
    unittest.main()
