import json
import unittest

from src.test_analysis_assistant.prompt_strategy import (
    AnalysisPromptConfig,
    AnalysisPromptStrategy,
    FewShotExample,
    PromptStrategy,
    PromptSection,
    TEST_GAP_EXAMPLES,
    ROOT_CAUSE_EXAMPLES,
    build_adaptive_prompt,
    build_cot_prompt,
    build_fewshot_prompt,
    build_hierarchical_prompt,
    extract_structured_response,
    estimate_token_count,
    truncate_for_context,
    default_strategy,
)
from src.test_analysis_assistant.rag_analyzer import RAGAnalysisResult
from src.test_analysis_assistant.models import AnalysisResult, FailureCluster


class TestPromptStrategyEnum(unittest.TestCase):
    def test_prompt_strategy_values(self):
        self.assertEqual(PromptStrategy.BASIC.value, "basic")
        self.assertEqual(PromptStrategy.COT.value, "chain_of_thought")
        self.assertEqual(PromptStrategy.FEWSHOT.value, "few_shot")
        self.assertEqual(PromptStrategy.ADAPTIVE.value, "adaptive")
        self.assertEqual(PromptStrategy.HIERARCHICAL.value, "hierarchical")


class TestFewShotExamples(unittest.TestCase):
    def test_test_gap_examples_exist(self):
        self.assertGreater(len(TEST_GAP_EXAMPLES), 0)
        for ex in TEST_GAP_EXAMPLES:
            self.assertIsInstance(ex, FewShotExample)
            self.assertTrue(ex.input_text)
            self.assertTrue(ex.output_format)

    def test_root_cause_examples_exist(self):
        self.assertGreater(len(ROOT_CAUSE_EXAMPLES), 0)
        for ex in ROOT_CAUSE_EXAMPLES:
            self.assertIsInstance(ex, FewShotExample)
            self.assertTrue(ex.input_text)
            self.assertTrue(ex.output_format)


class TestBuildCoTPrompt(unittest.TestCase):
    def test_build_cot_prompt_basic(self):
        question = "What is the root cause of this test failure?"
        context = ["Error: ModuleNotFoundError", "Import statement in test file"]

        prompt = build_cot_prompt(question, context, analysis_type="root_cause")

        self.assertIn("Task", prompt)
        self.assertIn("Context", prompt)
        self.assertIn("Reasoning Guidelines", prompt)
        self.assertIn(question, prompt)

    def test_build_cot_prompt_with_test_gap(self):
        question = "What test gaps exist?"
        context = ["No test for expired tokens", "Missing negative cases"]

        prompt = build_cot_prompt(question, context, analysis_type="test_gap")

        self.assertIn("Classification Guidelines", prompt)
        self.assertIn("Context", prompt)

    def test_build_cot_prompt_empty_context(self):
        question = "Analyze this"
        prompt = build_cot_prompt(question, [])

        self.assertIn(question, prompt)


class TestBuildFewshotPrompt(unittest.TestCase):
    def test_build_fewshot_prompt_basic(self):
        question = "Classify this test failure"
        context = ["Some context about the failure"]
        examples = TEST_GAP_EXAMPLES[:2]
        output_schema = {
            "gap_type": "string",
            "severity": "string",
        }

        prompt = build_fewshot_prompt(question, context, examples, output_schema)

        self.assertIn("Task", prompt)
        self.assertIn("Examples", prompt)
        self.assertIn("Output Schema", prompt)
        self.assertIn(question, prompt)

    def test_build_fewshot_prompt_includes_examples(self):
        question = "Find gaps"
        context = []
        examples = TEST_GAP_EXAMPLES[:1]
        output_schema = {"gap_type": "string"}

        prompt = build_fewshot_prompt(question, context, examples, output_schema)

        self.assertIn("Example 1:", prompt)
        self.assertIn("Input:", prompt)
        self.assertIn("Output:", prompt)


class TestBuildHierarchicalPrompt(unittest.TestCase):
    def test_build_hierarchical_prompt_with_clusters(self):
        # Create a mock analysis result with clusters
        result = RAGAnalysisResult(
            base_result=AnalysisResult(
                input_format="junit",
                total_failures=5,
                clusters=[
                    FailureCluster(
                        cluster_id="c1",
                        reason="import error",
                        error_type="ModuleNotFoundError",
                        count=3,
                        tests=["test_auth", "test_login"],
                    ),
                    FailureCluster(
                        cluster_id="c2",
                        reason="assertion failed",
                        error_type="AssertionError",
                        count=2,
                        tests=["test_validate"],
                    ),
                ],
            ),
            retrieval_insights=[],
            risk_assessment={"overall_risk": "high", "retrieval_confidence": 0.75},
        )

        prompt = build_hierarchical_prompt(result)

        self.assertIn("Stage 1: Failure Clustering", prompt)
        # Stage 2 and 3 only appear when there are retrieval insights
        self.assertIn("Stage 4: Risk Assessment", prompt)
        self.assertIn("Stage 5: Action Planning", prompt)
        self.assertIn("ModuleNotFoundError", prompt)
        self.assertIn("Overall risk: high", prompt)

    def test_build_hierarchical_prompt_empty(self):
        result = RAGAnalysisResult(
            base_result=AnalysisResult(
                input_format="junit",
                total_failures=0,
            ),
        )

        prompt = build_hierarchical_prompt(result)

        self.assertIn("Analysis Overview", prompt)
        self.assertIn("0 test failures", prompt)


class TestBuildAdaptivePrompt(unittest.TestCase):
    def test_build_adaptive_prompt_uses_basic(self):
        config = AnalysisPromptConfig(strategy=PromptStrategy.BASIC)
        question = "Test question"
        context = ["Context 1", "Context 2"]

        prompt = build_adaptive_prompt(question, context, config=config)

        self.assertIn(question, prompt)

    def test_build_adaptive_prompt_uses_cot(self):
        config = AnalysisPromptConfig(strategy=PromptStrategy.COT)
        question = "Root cause?"
        context = ["ctx1", "ctx2", "ctx3"]

        prompt = build_adaptive_prompt(question, context, config=config)

        self.assertIn("Reasoning Guidelines", prompt)

    def test_build_adaptive_prompt_auto_selects_hierarchical(self):
        # With enough context and analysis result, should use hierarchical
        config = AnalysisPromptConfig(strategy=PromptStrategy.ADAPTIVE)
        result = RAGAnalysisResult(
            base_result=AnalysisResult(
                input_format="junit",
                total_failures=3,
                clusters=[
                    FailureCluster(
                        cluster_id="c1",
                        reason="error",
                        error_type="Error",
                        count=1,
                    )
                ],
            ),
        )

        # Need at least 5 context items for hierarchical
        context = ["ctx1", "ctx2", "ctx3", "ctx4", "ctx5"]

        prompt = build_adaptive_prompt(
            "analyze failures",
            context,
            analysis_result=result,
            config=config,
        )

        # Should have hierarchical structure
        self.assertIn("## Analysis Overview", prompt)


class TestExtractStructuredResponse(unittest.TestCase):
    def test_extract_json_from_text(self):
        text = 'Some text before {"key": "value", "number": 42} some text after'

        result = extract_structured_response(text, {"key": str, "number": int})

        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)

    def test_extract_json_partial_match(self):
        text = '{"gap_type": "missing_test", "severity": "high"}'

        result = extract_structured_response(text, {"gap_type": str})

        self.assertIsNotNone(result)
        self.assertEqual(result["gap_type"], "missing_test")

    def test_extract_json_invalid_json(self):
        text = "Not valid JSON at all"

        result = extract_structured_response(text, {"key": str})

        self.assertIsNone(result)


class TestTokenEstimation(unittest.TestCase):
    def test_estimate_token_count(self):
        text = "This is a test sentence with some words."
        # ~7 words / 4 = ~2 tokens
        estimated = estimate_token_count(text)

        self.assertGreater(estimated, 0)
        self.assertLess(estimated, len(text))

    def test_estimate_token_count_empty(self):
        self.assertEqual(estimate_token_count(""), 0)


class TestTruncateForContext(unittest.TestCase):
    def test_truncate_short_text(self):
        text = "Short text"
        result = truncate_for_context(text, max_tokens=100)

        self.assertEqual(result, text)

    def test_truncate_long_text(self):
        text = "A" * 1000
        result = truncate_for_context(text, max_tokens=100)

        self.assertLess(len(result), len(text))
        self.assertIn("truncated", result)


class TestAnalysisPromptStrategy(unittest.TestCase):
    def test_default_strategy(self):
        self.assertIsNotNone(default_strategy)
        self.assertEqual(default_strategy.strategy, PromptStrategy.ADAPTIVE)

    def test_strategy_configuration(self):
        config = AnalysisPromptConfig(
            strategy=PromptStrategy.COT,
            temperature=0.5,
            include_reasoning=True,
        )
        strategy = AnalysisPromptStrategy(config)

        self.assertEqual(strategy.strategy, PromptStrategy.COT)
        self.assertEqual(strategy.temperature, 0.5)
        self.assertTrue(strategy.should_include_reasoning())

    def test_get_examples_test_gap(self):
        config = AnalysisPromptConfig(max_examples=2)
        strategy = AnalysisPromptStrategy(config)

        examples = strategy.get_examples("test_gap")

        self.assertLessEqual(len(examples), 2)

    def test_get_examples_root_cause(self):
        config = AnalysisPromptConfig(max_examples=1)
        strategy = AnalysisPromptStrategy(config)

        examples = strategy.get_examples("root_cause")

        self.assertLessEqual(len(examples), 1)

    def test_build_with_strategy(self):
        config = AnalysisPromptConfig(strategy=PromptStrategy.BASIC)
        strategy = AnalysisPromptStrategy(config)

        prompt = strategy.build("Test question?", ["Context 1"])

        self.assertIn("Test question?", prompt)


class TestPromptSection(unittest.TestCase):
    def test_prompt_section_defaults(self):
        section = PromptSection(title="Test", content="Content")

        self.assertEqual(section.title, "Test")
        self.assertEqual(section.content, "Content")
        self.assertTrue(section.required)


class TestSelfConsistencyChecker(unittest.TestCase):
    def test_check_perfect_consensus(self):
        from src.test_analysis_assistant.prompt_strategy import SelfConsistencyChecker

        checker = SelfConsistencyChecker(num_paths=3)
        answers = ["environment_difference", "environment_difference", "environment_difference"]

        result = checker.check(answers)

        self.assertEqual(result.consensus_answer, "environment_difference")
        self.assertEqual(result.consistency_score, 1.0)
        self.assertEqual(result.answer_votes["environment_difference"], 3)
        self.assertEqual(len(result.disagreeing_indices), 0)

    def test_check_partial_consensus(self):
        from src.test_analysis_assistant.prompt_strategy import SelfConsistencyChecker

        checker = SelfConsistencyChecker(num_paths=4)
        answers = ["environment_difference", "environment_difference", "race_condition", "data_issue"]

        result = checker.check(answers)

        self.assertEqual(result.consensus_answer, "environment_difference")
        self.assertEqual(result.consistency_score, 0.5)  # 2/4
        self.assertEqual(result.answer_votes["environment_difference"], 2)
        self.assertEqual(len(result.disagreeing_indices), 2)

    def test_check_no_consensus(self):
        from src.test_analysis_assistant.prompt_strategy import SelfConsistencyChecker

        # Use unique values that normalize to different strings
        checker = SelfConsistencyChecker(num_paths=3, normalize_answers=False)
        answers = ["answer_A", "answer_B", "answer_C"]

        result = checker.check(answers)

        self.assertIn(result.consensus_answer, ["answer_A", "answer_B", "answer_C"])
        self.assertEqual(result.consistency_score, 1/3)
        self.assertEqual(len(result.disagreeing_indices), 2)

    def test_check_with_reasoning_paths(self):
        from src.test_analysis_assistant.prompt_strategy import SelfConsistencyChecker

        checker = SelfConsistencyChecker(num_paths=3)
        answers = ["root_cause_A", "root_cause_A", "root_cause_B"]
        reasoning = [
            "Error suggests database is not available in CI",
            "Missing database configuration causes connection failure",
            "The test setup doesn't include proper fixtures"
        ]

        result = checker.check(answers, reasoning)

        self.assertEqual(len(result.reasoning_paths), 3)
        self.assertIn("database", result.reasoning_paths[0])

    def test_check_empty_answers(self):
        from src.test_analysis_assistant.prompt_strategy import SelfConsistencyChecker

        checker = SelfConsistencyChecker()
        result = checker.check([])

        self.assertIsNone(result.consensus_answer)
        self.assertEqual(result.consistency_score, 0.0)
        self.assertEqual(len(result.answer_votes), 0)

    def test_normalization_case_insensitive(self):
        from src.test_analysis_assistant.prompt_strategy import SelfConsistencyChecker

        checker = SelfConsistencyChecker(normalize_answers=True)
        answers = ["Environment_Difference", "ENVIRONMENT_DIFFERENCE", "environment_difference"]

        result = checker.check(answers)

        self.assertEqual(result.consistency_score, 1.0)

    def test_normalization_whitespace(self):
        from src.test_analysis_assistant.prompt_strategy import SelfConsistencyChecker

        checker = SelfConsistencyChecker(normalize_answers=True)
        answers = ["answer ", " answer", "answer"]

        result = checker.check(answers)

        self.assertEqual(result.consistency_score, 1.0)

    def test_confidence_calculation(self):
        from src.test_analysis_assistant.prompt_strategy import SelfConsistencyChecker

        checker = SelfConsistencyChecker()
        # Perfect consensus
        answers = ["A", "A", "A"]
        result = checker.check(answers)

        self.assertEqual(result.confidence, 1.0)

    def test_generate_diversity_prompts(self):
        from src.test_analysis_assistant.prompt_strategy import SelfConsistencyChecker

        checker = SelfConsistencyChecker(num_paths=3)
        base = "What is the root cause?"

        prompts = checker.generate_diversity_prompts(base)

        self.assertEqual(len(prompts), 3)
        self.assertEqual(prompts[0], base)  # Original
        self.assertIn("step by step", prompts[1])

    def test_to_dict(self):
        from src.test_analysis_assistant.prompt_strategy import SelfConsistencyChecker

        checker = SelfConsistencyChecker()
        answers = ["A", "A", "B"]
        result = checker.check(answers)

        d = result.to_dict()

        self.assertIn("consensus_answer", d)
        self.assertIn("consistency_score", d)
        self.assertIn("confidence", d)
        self.assertIn("answer_votes", d)


class TestCheckSelfConsistency(unittest.TestCase):
    def test_convenience_function(self):
        from src.test_analysis_assistant.prompt_strategy import check_self_consistency

        answers = ["root_cause", "root_cause", "root_cause"]
        result = check_self_consistency(answers)

        self.assertEqual(result.consensus_answer, "root_cause")
        self.assertEqual(result.consistency_score, 1.0)


if __name__ == "__main__":
    unittest.main()
