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


if __name__ == "__main__":
    unittest.main()
