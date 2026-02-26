"""Advanced prompt strategy for test analysis LLM interactions.

This module provides sophisticated prompt building techniques including:
- Chain-of-thought reasoning for root cause analysis
- Few-shot learning for test gap classification
- Confidence-aware prompting
- Multi-step analysis workflows
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .models import AnalysisResult, FailureCluster
from .rag_analyzer import (
    RAGAnalysisResult,
    RetrievalInsight,
    TestGapAnalysis,
)


class PromptStrategy(str, Enum):
    """Prompt strategy types."""
    BASIC = "basic"  # Simple direct prompting
    COT = "chain_of_thought"  # Chain-of-thought reasoning
    FEWSHOT = "few_shot"  # Few-shot learning with examples
    ADAPTIVE = "adaptive"  # Strategy selection based on analysis type
    HIERARCHICAL = "hierarchical"  # Multi-step hierarchical analysis


@dataclass
class FewShotExample:
    """A few-shot example for prompting."""
    input_text: str
    output_format: str
    explanation: str = ""


@dataclass
class AnalysisPromptConfig:
    """Configuration for analysis prompting."""
    strategy: PromptStrategy = PromptStrategy.ADAPTIVE
    include_reasoning: bool = True
    max_examples: int = 3
    temperature: float = 0.3
    include_confidence_guidance: bool = True
    structured_output: bool = True


@dataclass
class PromptSection:
    """A section in a prompt."""
    title: str
    content: str
    required: bool = True


# Few-shot examples for test gap classification
TEST_GAP_EXAMPLES = [
    FewShotExample(
        input_text="ModuleNotFoundError: No module named 'psycopg2'",
        output_format=json.dumps({
            "gap_type": "missing_dependency",
            "severity": "critical",
            "suggested_action": "Add dependency to requirements.txt or pyproject.toml",
            "test_coverage_needed": "Import validation tests"
        }),
        explanation="Import errors indicate missing dependencies - high severity"
    ),
    FewShotExample(
        input_text="AssertionError: assert user.is_valid == True",
        output_format=json.dumps({
            "gap_type": "edge_case",
            "severity": "medium",
            "suggested_action": "Add test cases for None, empty string, and invalid types",
            "test_coverage_needed": "Boundary value analysis tests"
        }),
        explanation="Assertion failures often indicate missing edge case coverage"
    ),
    FewShotExample(
        input_text="TimeoutError: operation timed out after 30s",
        output_format=json.dumps({
            "gap_type": "performance",
            "severity": "high",
            "suggested_action": "Add timeout handling and async test coverage",
            "test_coverage_needed": "Performance and timeout tests"
        }),
        explanation="Timeouts indicate need for async and performance testing"
    ),
]


# Few-shot examples for root cause analysis
ROOT_CAUSE_EXAMPLES = [
    FewShotExample(
        input_text="Test fails in CI but passes locally. Error: database connection refused",
        output_format=json.dumps({
            "root_cause_category": "environment_difference",
            "confidence": 0.85,
            "reasoning": "Connection refused suggests different database state between CI and local",
            "recommended_actions": [
                "Check CI database configuration",
                "Ensure test data is seeded consistently",
                "Use containerized database for CI"
            ]
        }),
        explanation="Environment differences are common root causes for flaky tests"
    ),
    FewShotExample(
        input_text="Flaky test: sometimes passes, sometimes fails with IndexError",
        output_format=json.dumps({
            "root_cause_category": "race_condition",
            "confidence": 0.70,
            "reasoning": "Intermittent IndexError suggests unordered collection access",
            "recommended_actions": [
                "Add proper synchronization",
                "Sort collections before iteration",
                "Add retry logic with deterministic ordering"
            ]
        }),
        explanation="Intermittent errors often indicate race conditions or timing issues"
    ),
]


def build_cot_prompt(
    question: str,
    context: Sequence[str],
    analysis_type: str = "root_cause",
) -> str:
    """Build a chain-of-thought prompt for reasoning tasks.

    Args:
        question: The question to answer
        context: Retrieved context strings
        analysis_type: Type of analysis (root_cause, test_gap, risk)

    Returns:
        Formatted chain-of-thought prompt
    """
    sections = [
        PromptSection(
            title="Task",
            content=f"You are analyzing test failures. {question}",
        ),
        PromptSection(
            title="Context",
            content="\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(context)),
        ),
    ]

    if analysis_type == "root_cause":
        sections.append(PromptSection(
            title="Reasoning Guidelines",
            content="\n".join([
                "1. First, identify the error type and pattern",
                "2. Consider the test environment and timing",
                "3. Look for similar issues in the context",
                "4. Evaluate confidence based on evidence strength",
                "5. Propose actionable next steps",
            ]),
        ))
        sections.append(PromptSection(
            title="Output Format",
            content=json.dumps({
                "root_cause_category": "environment_difference | code_bug | race_condition | missing_mock | data_issue | timing | unknown",
                "confidence": "0.0-1.0",
                "reasoning": "Step-by-step explanation",
                "recommended_actions": ["action1", "action2"],
            }),
        ))
    elif analysis_type == "test_gap":
        sections.append(PromptSection(
            title="Classification Guidelines",
            content="\n".join([
                "1. Analyze the error type to determine gap category",
                "2. Assess severity based on release impact",
                "3. Identify specific missing test scenarios",
                "4. Estimate testing effort required",
            ]),
        ))

    return _build_prompt_from_sections(sections)


def build_fewshot_prompt(
    question: str,
    context: Sequence[str],
    examples: Sequence[FewShotExample],
    output_schema: Dict[str, Any],
) -> str:
    """Build a few-shot prompt with examples.

    Args:
        question: The question to answer
        context: Retrieved context strings
        examples: Few-shot examples to include
        output_schema: Expected output JSON schema

    Returns:
        Formatted few-shot prompt
    """
    sections = [
        PromptSection(
            title="Task",
            content=question,
        ),
    ]

    # Add examples
    example_texts = []
    for i, ex in enumerate(examples, start=1):
        example_texts.append(f"Example {i}:")
        example_texts.append(f"Input: {ex.input_text}")
        example_texts.append(f"Output: {ex.output_format}")
        if ex.explanation:
            example_texts.append(f"Why: {ex.explanation}")
        example_texts.append("")

    sections.append(PromptSection(
        title="Examples",
        content="\n".join(example_texts),
    ))

    # Add context
    if context:
        sections.append(PromptSection(
            title="Context",
            content="\n\n".join(f"[{i+1}] {ctx}" for i, ctx in enumerate(context)),
        ))

    # Add output schema
    sections.append(PromptSection(
        title="Output Schema",
        content=json.dumps(output_schema, indent=2),
    ))

    return _build_prompt_from_sections(sections)


def build_hierarchical_prompt(
    analysis_result: RAGAnalysisResult,
    focus_areas: Optional[List[str]] = None,
) -> str:
    """Build a hierarchical multi-step analysis prompt.

    This creates a structured prompt that guides through multiple analysis stages:
    1. Failure clustering review
    2. Root cause hypothesis generation
    3. Test gap identification
    4. Risk prioritization
    5. Action planning

    Args:
        analysis_result: The RAG analysis result
        focus_areas: Optional specific areas to focus on

    Returns:
        Hierarchical prompt string
    """
    sections = [
        PromptSection(
            title="Analysis Overview",
            content=f"Analyze {analysis_result.base_result.total_failures} test failures "
                   f"across {len(analysis_result.base_result.clusters)} clusters.",
        ),
    ]

    # Stage 1: Failure clustering
    if analysis_result.base_result.clusters:
        cluster_summary = "\n".join(
            f"- {c.error_type}: {c.count} failures ({c.reason})"
            for c in analysis_result.base_result.clusters[:5]
        )
        sections.append(PromptSection(
            title="Stage 1: Failure Clustering",
            content=f"Failure clusters:\n{cluster_summary}",
        ))

    # Stage 2: Root cause hypotheses
    if analysis_result.retrieval_insights:
        rc_insights = [
            i for i in analysis_result.retrieval_insights
            if i.insight_type == "root_cause_evidence"
        ]
        if rc_insights:
            insight_summary = "\n".join(
                f"- {i.title}: {i.description[:100]} (confidence: {i.confidence:.0%})"
                for i in rc_insights[:3]
            )
            sections.append(PromptSection(
                title="Stage 2: Root Cause Evidence",
                content=insight_summary,
            ))

    # Stage 3: Test gaps
    if analysis_result.structured_gaps:
        gap_summary = "\n".join(
            f"- [{g.gap_id}] {g.gap_type} ({g.severity}): {g.description[:80]}"
            for g in analysis_result.structured_gaps[:5]
        )
        sections.append(PromptSection(
            title="Stage 3: Test Gaps",
            content=gap_summary,
        ))

    # Stage 4: Risk assessment
    if analysis_result.risk_assessment:
        risk_info = analysis_result.risk_assessment
        risk_summary = f"Overall risk: {risk_info.get('overall_risk', 'unknown')}"
        if 'retrieval_confidence' in risk_info:
            risk_summary += f", Retrieval confidence: {risk_info['retrieval_confidence']:.0%}"
        sections.append(PromptSection(
            title="Stage 4: Risk Assessment",
            content=risk_summary,
        ))

    # Stage 5: Action planning
    sections.append(PromptSection(
        title="Stage 5: Action Planning",
        content="Based on the above analysis, generate prioritized remediation steps.",
    ))

    # Add output requirements
    sections.append(PromptSection(
        title="Output Requirements",
        content="\n".join([
            "1. Provide confidence level for each stage",
            "2. Identify missing information that would improve analysis",
            "3. List actionable next steps with priority",
        ]),
    ))

    return _build_prompt_from_sections(sections)


def build_adaptive_prompt(
    question: str,
    context: Sequence[str],
    analysis_result: Optional[RAGAnalysisResult] = None,
    config: Optional[AnalysisPromptConfig] = None,
) -> str:
    """Build an adaptive prompt based on analysis context.

    Automatically selects the best prompting strategy based on:
    - Available context
    - Analysis type
    - Confidence levels

    Args:
        question: The question to answer
        context: Retrieved context strings
        analysis_result: Optional existing analysis result
        config: Prompt configuration

    Returns:
        Optimized prompt string
    """
    if config is None:
        config = AnalysisPromptConfig()

    strategy = config.strategy

    # Auto-select strategy based on context
    if strategy == PromptStrategy.ADAPTIVE:
        if len(context) >= 5 and analysis_result:
            strategy = PromptStrategy.HIERARCHICAL
        elif len(context) >= 3:
            strategy = PromptStrategy.COT
        else:
            strategy = PromptStrategy.BASIC

    # Build appropriate prompt
    if strategy == PromptStrategy.HIERARCHICAL and analysis_result:
        return build_hierarchical_prompt(analysis_result)
    elif strategy == PromptStrategy.COT:
        return build_cot_prompt(question, context)
    elif strategy == PromptStrategy.FEWSHOT:
        # Determine example set based on question content
        question_lower = question.lower()
        if "gap" in question_lower or "coverage" in question_lower:
            examples = TEST_GAP_EXAMPLES[:config.max_examples]
            output_schema = {
                "gap_type": "string",
                "severity": "critical|high|medium|low",
                "suggested_action": "string",
                "test_coverage_needed": "string",
            }
        else:
            examples = ROOT_CAUSE_EXAMPLES[:config.max_examples]
            output_schema = {
                "root_cause_category": "string",
                "confidence": "0.0-1.0",
                "reasoning": "string",
                "recommended_actions": ["string"],
            }
        return build_fewshot_prompt(question, context, examples, output_schema)
    else:
        return _build_basic_prompt(question, context)


def _build_basic_prompt(question: str, context: Sequence[str]) -> str:
    """Build a basic prompt without special strategies."""
    lines = [
        f"Question: {question}",
        "",
        "Context:",
    ]

    if not context:
        lines.append("(no context available)")
    else:
        for i, ctx in enumerate(context, start=1):
            lines.append(f"[{i}] {ctx}")

    return "\n".join(lines)


def _build_prompt_from_sections(sections: List[PromptSection]) -> str:
    """Build a prompt from structured sections."""
    lines = []

    for section in sections:
        if not section.required and not section.content:
            continue

        lines.append(f"## {section.title}")
        lines.append(section.content)
        lines.append("")

    return "\n".join(lines)


def extract_structured_response(
    text: str,
    schema: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Extract structured JSON from LLM response.

    Args:
        text: Raw LLM response text
        schema: Expected schema keys

    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Try to find JSON in the text
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not json_match:
        # Try broader match
        json_matches = re.findall(r'\{.*\}', text, re.DOTALL)
        if json_matches:
            json_match = json_matches[0]

    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            # Validate against schema
            return {
                k: v for k, v in parsed.items()
                if k in schema
            }
        except json.JSONDecodeError:
            pass

    return None


def estimate_token_count(text: str) -> int:
    """Rough estimate of token count.

    Uses a simple heuristic: ~4 characters per token.
    More accurate for English text.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4


def truncate_for_context(
    text: str,
    max_tokens: int = 2000,
) -> str:
    """Truncate text to fit within token budget.

    Args:
        text: Input text
        max_tokens: Maximum tokens allowed

    Returns:
        Truncated text
    """
    estimated = estimate_token_count(text)
    if estimated <= max_tokens:
        return text

    # Truncate and add indicator
    char_limit = max_tokens * 4
    return text[:char_limit] + "\n\n[truncated for context length...]"


class AnalysisPromptStrategy:
    """Manages prompt strategy selection and configuration."""

    def __init__(self, config: Optional[AnalysisPromptConfig] = None):
        self._config = config or AnalysisPromptConfig()

    @property
    def strategy(self) -> PromptStrategy:
        return self._config.strategy

    @property
    def temperature(self) -> float:
        return self._config.temperature

    def should_include_reasoning(self) -> bool:
        return self._config.include_reasoning

    def should_include_confidence(self) -> bool:
        return self._config.include_confidence_guidance

    def get_examples(
        self,
        analysis_type: str,
    ) -> List[FewShotExample]:
        """Get appropriate few-shot examples for analysis type.

        Args:
            analysis_type: Type of analysis (test_gap, root_cause, risk)

        Returns:
            List of relevant examples
        """
        if analysis_type == "test_gap":
            return TEST_GAP_EXAMPLES[:self._config.max_examples]
        elif analysis_type == "root_cause":
            return ROOT_CAUSE_EXAMPLES[:self._config.max_examples]
        return []

    def build(
        self,
        question: str,
        context: Sequence[str],
        analysis_result: Optional[RAGAnalysisResult] = None,
    ) -> str:
        """Build a prompt using the configured strategy.

        Args:
            question: The question to answer
            context: Retrieved context
            analysis_result: Optional existing analysis result

        Returns:
            Formatted prompt string
        """
        return build_adaptive_prompt(
            question=question,
            context=context,
            analysis_result=analysis_result,
            config=self._config,
        )


# Default strategy instance
default_strategy = AnalysisPromptStrategy()


@dataclass
class ConsistencyResult:
    """Result from self-consistency checking."""
    consensus_answer: Any
    consistency_score: float  # 0-1, higher is more consistent
    answer_votes: Dict[str, int]  # counts per unique answer
    reasoning_paths: List[str]  # different reasoning paths taken
    confidence: float  # confidence based on consistency
    disagreeing_indices: List[int]  # indices of inconsistent answers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "consensus_answer": self.consensus_answer,
            "consistency_score": round(self.consistency_score, 3),
            "answer_votes": self.answer_votes,
            "reasoning_paths": self.reasoning_paths,
            "confidence": round(self.confidence, 3),
            "disagreeing_indices": self.disagreeing_indices,
        }


class SelfConsistencyChecker:
    """Self-consistency checking for improved LLM reliability.

    This implements the self-consistency technique from Wang et al. (2022):
    - Generate multiple diverse reasoning paths
    - Aggregate answers by voting
    - Select the most consistent answer

    This significantly improves reliability especially for complex reasoning tasks.
    """

    def __init__(
        self,
        num_paths: int = 3,
        temperature: float = 0.5,
        normalize_answers: bool = True,
    ) -> None:
        """Initialize the self-consistency checker.

        Args:
            num_paths: Number of diverse reasoning paths to generate
            temperature: Temperature for generating diverse paths (higher = more diverse)
            normalize_answers: Whether to normalize answers before voting
        """
        self._num_paths = num_paths
        self._temperature = temperature
        self._normalize_answers = normalize_answers

    @property
    def num_paths(self) -> int:
        """Get number of reasoning paths."""
        return self._num_paths

    def check(
        self,
        answers: List[Any],
        reasoning_paths: Optional[List[str]] = None,
    ) -> ConsistencyResult:
        """Check consistency across multiple answers.

        Args:
            answers: List of answers from different reasoning paths
            reasoning_paths: Optional reasoning paths for each answer

        Returns:
            ConsistencyResult with consensus and confidence metrics
        """
        if not answers:
            return ConsistencyResult(
                consensus_answer=None,
                consistency_score=0.0,
                answer_votes={},
                reasoning_paths=[],
                confidence=0.0,
                disagreeing_indices=[],
            )

        # Normalize answers for comparison
        normalized = self._normalize(answers)

        # Count votes for each unique answer
        answer_votes: Dict[str, int] = {}
        answer_mapping: Dict[int, str] = {}  # index -> normalized

        for i, norm_ans in enumerate(normalized):
            if norm_ans in answer_votes:
                answer_votes[norm_ans] += 1
            else:
                answer_votes[norm_ans] = 1
            answer_mapping[i] = norm_ans

        # Find consensus answer (most voted)
        if answer_votes:
            consensus = max(answer_votes, key=answer_votes.get)
            consensus_votes = answer_votes[consensus]
        else:
            consensus = str(answers[0])
            consensus_votes = 1

        # Calculate consistency score
        total = len(answers)
        consistency_score = consensus_votes / total if total > 0 else 0.0

        # Identify disagreeing indices
        disagreeing = [
            i for i, norm in enumerate(normalized)
            if norm != consensus
        ]

        # Confidence is consistency weighted by agreement strength
        confidence = self._calculate_confidence(
            consistency_score, consensus_votes, total
        )

        return ConsistencyResult(
            consensus_answer=consensus,
            consistency_score=consistency_score,
            answer_votes=answer_votes,
            reasoning_paths=reasoning_paths or [],
            confidence=confidence,
            disagreeing_indices=disagreeing,
        )

    def _normalize(self, answers: List[Any]) -> List[str]:
        """Normalize answers for comparison.

        Args:
            answers: Raw answers

        Returns:
            List of normalized answer strings
        """
        if not self._normalize_answers:
            return [str(a) for a in answers]

        normalized = []
        for ans in answers:
            ans_str = str(ans).lower().strip()
            # Remove extra whitespace
            ans_str = re.sub(r'\s+', ' ', ans_str)
            # Remove common variations
            ans_str = ans_str.strip('.,;:!?')
            normalized.append(ans_str)

        return normalized

    def _calculate_confidence(
        self,
        consistency_score: float,
        votes: int,
        total: int,
    ) -> float:
        """Calculate confidence based on consistency.

        Args:
            consistency_score: Fraction agreeing with consensus
            votes: Number of votes for consensus
            total: Total number of answers

        Returns:
            Confidence score 0-1
        """
        # Base confidence on consistency
        base = consistency_score

        # Bonus for more votes (more certainty)
        vote_bonus = (votes - 1) / (total - 1) if total > 1 else 0.0

        # Combine with weights
        confidence = (base * 0.7) + (vote_bonus * 0.3)

        return round(confidence, 3)

    def generate_diversity_prompts(
        self,
        base_prompt: str,
        strategy: PromptStrategy = PromptStrategy.COT,
    ) -> List[str]:
        """Generate diverse prompts for self-consistency checking.

        Args:
            base_prompt: The base prompt to diversify
            strategy: Prompt strategy to use

        Returns:
            List of diverse prompts
        """
        diversifiers = [
            "",  # Original
            "Think step by step and explain your reasoning.",
            "Consider alternative approaches and justify your answer.",
            "Analyze the problem from first principles.",
            "Use a different method to verify your answer.",
        ]

        prompts = []
        for i, diver in enumerate(diversifiers[:self._num_paths]):
            if i == 0:
                prompts.append(base_prompt)
            else:
                # Insert diversifier into the prompt
                prompts.append(f"{base_prompt}\n\n{diver}")

        return prompts[:self._num_paths]


def check_self_consistency(
    answers: List[Any],
    reasoning_paths: Optional[List[str]] = None,
    num_paths: int = 3,
) -> ConsistencyResult:
    """Convenience function for self-consistency checking.

    Args:
        answers: List of answers to check
        reasoning_paths: Optional reasoning paths
        num_paths: Expected number of paths

    Returns:
        ConsistencyResult with consensus and metrics
    """
    checker = SelfConsistencyChecker(num_paths=num_paths)
    return checker.check(answers, reasoning_paths)
