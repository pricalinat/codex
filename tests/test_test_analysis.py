import unittest

from src.multi_agent.test_analysis import analyze_unittest_output


PASS_OUTPUT = """\
test_register_and_dispatch (tests.test_agents.TestAgents) ... ok
test_executes_steps_successfully (tests.test_orchestrator.TestOrchestrator) ... ok
test_submit_and_run_task (tests.test_e2e.TestE2E) ... ok
test_goal_aligned_paper_is_ranked_first (tests.test_paper_retrieval.TestPaperRetrieval) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.012s

OK
"""


FAIL_OUTPUT = """\
test_register_and_dispatch (tests.test_agents.TestAgents) ... ok
test_goal_aligned_paper_is_ranked_first (tests.test_paper_retrieval.TestPaperRetrieval) ... FAIL
test_submit_and_run_task (tests.test_e2e.TestE2E) ... ERROR

======================================================================
FAIL: test_goal_aligned_paper_is_ranked_first (tests.test_paper_retrieval.TestPaperRetrieval)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/test.py", line 10, in test_goal_aligned_paper_is_ranked_first
    self.assertEqual("a", "b")
AssertionError: 'a' != 'b'

======================================================================
ERROR: test_submit_and_run_task (tests.test_e2e.TestE2E)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/tmp/test.py", line 20, in test_submit_and_run_task
    raise RuntimeError("boom")
RuntimeError: boom

----------------------------------------------------------------------
Ran 3 tests in 0.021s

FAILED (failures=1, errors=1)
"""


MULTILINE_VERBOSE_OUTPUT = """\
test_simple_case (tests.test_x.TestX) ... ok
test_with_docstring (tests.test_y.TestY)
Human readable description line. ... ok
test_another_with_doc (tests.test_z.TestZ)
Another description. ... FAIL

----------------------------------------------------------------------
Ran 3 tests in 0.010s

FAILED (failures=1)
"""


class TestTestAnalysis(unittest.TestCase):
    def test_analyze_pass_output(self):
        report = analyze_unittest_output("python3 -m unittest -v", PASS_OUTPUT, exit_code=0)
        self.assertEqual(report.overall_status, "pass")
        self.assertEqual(report.total_tests, 4)
        self.assertEqual(report.passed, 4)
        self.assertEqual(report.failed, 0)
        self.assertEqual(report.errors, 0)
        self.assertAlmostEqual(report.duration_seconds, 0.012, places=3)
        self.assertIn("paper_retrieval", report.module_breakdown)
        self.assertEqual(report.module_breakdown["paper_retrieval"]["pass"], 1)

    def test_analyze_fail_output(self):
        report = analyze_unittest_output("python3 -m unittest -v", FAIL_OUTPUT, exit_code=1)
        self.assertEqual(report.overall_status, "fail")
        self.assertEqual(report.total_tests, 3)
        self.assertEqual(report.failed, 1)
        self.assertEqual(report.errors, 1)
        self.assertEqual(len(report.failures), 2)
        severities = {item.severity for item in report.failures}
        self.assertIn("P0", severities)  # ERROR path
        self.assertIn("P1", severities)  # AssertionError path

        action_titles = [item.title for item in report.actions]
        self.assertIn("Fix blocking test failures", action_titles)

    def test_markdown_report_contains_expected_sections(self):
        report = analyze_unittest_output("python3 -m unittest -v", FAIL_OUTPUT, exit_code=1)
        markdown = report.to_markdown()
        self.assertIn("# Test Analysis Report", markdown)
        self.assertIn("## 1. Execution Summary", markdown)
        self.assertIn("## 3. Key Findings", markdown)
        self.assertIn("## 4. Recommended Actions", markdown)
        self.assertIn("test_goal_aligned_paper_is_ranked_first", markdown)

    def test_parses_multiline_verbose_unittest_entries(self):
        report = analyze_unittest_output("python3 -m unittest -v", MULTILINE_VERBOSE_OUTPUT, exit_code=1)
        self.assertEqual(report.total_tests, 3)
        self.assertEqual(report.passed, 2)
        self.assertEqual(report.failed, 1)


if __name__ == "__main__":
    unittest.main()
