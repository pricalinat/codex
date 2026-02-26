"""LLM integration module for enhanced test analysis.

This module provides a unified interface for calling external LLMs to enhance
test analysis capabilities. It integrates with the existing prompt strategy
and provides a fallback for when retrieval alone isn't sufficient.

Features:
- Simple LLM client interface with retry logic
- Integration with existing prompt strategies (CoT, few-shot, hierarchical)
- Support for multiple LLM providers (OpenAI, Anthropic, local models)
- Streaming support for long responses
- Token usage tracking
"""

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence

from .prompt_strategy import (
    AnalysisPromptConfig,
    PromptStrategy,
    build_adaptive_prompt,
    build_cot_prompt,
    build_fewshot_prompt,
    build_hierarchical_prompt,
    extract_structured_response,
    estimate_token_count,
)
from .rag_analyzer import RAGAnalysisResult

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # For local models like llama.cpp
    MOCK = "mock"  # For testing


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    provider: LLMProvider
    usage: Dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens, total_tokens
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw_response: Optional[Dict[str, Any]] = None

    def extract_json(self, schema: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Extract JSON from response content.

        Args:
            schema: Optional schema to validate against

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        return extract_structured_response(self.content, schema or {})


@dataclass
class LLMConfig:
    """Configuration for LLM calls."""

    provider: LLMProvider = LLMProvider.MOCK
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout_seconds: int = 30
    max_retries: int = 2
    retry_delay_seconds: float = 1.0

    # Provider-specific settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    def complete(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> LLMResponse:
        """Complete a prompt."""
        ...

    def stream_complete(
        self,
        prompt: str,
        config: LLMConfig,
    ) -> Iterator[str]:
        """Stream completion tokens."""
        ...


def _create_mock_client() -> LLMClient:
    """Create a mock client for testing."""

    class MockClient:
        def complete(self, prompt: str, config: LLMConfig) -> LLMResponse:
            # a simple mock response based on the prompt content
            content = "Analysis complete. This is a mock response for testing."
            if "root cause" in prompt.lower():
                content = json.dumps({
                    "root_cause_category": "environment_difference",
                    "confidence": 0.75,
                    "reasoning": "Mock analysis suggests checking environment configuration",
                    "recommended_actions": ["Verify test environment", "Check configuration files"]
                })
            elif "test gap" in prompt.lower():
                content = json.dumps({
                    "gap_type": "edge_case",
                    "severity": "medium",
                    "suggested_action": "Add boundary value tests",
                    "test_coverage_needed": "Edge case coverage"
                })
            return LLMResponse(
                content=content,
                model=config.model,
                provider=config.provider,
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                latency_ms=100.0,
            )

        def stream_complete(self, prompt: str, config: LLMConfig) -> Iterator[str]:
            # Yield mock tokens
            content = "Mock streaming response."
            for token in content.split():
                yield token + " "

    return MockClient()


def _create_openai_client(api_key: Optional[str] = None) -> LLMClient:
    """Create an OpenAI client."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed, falling back to mock client")
        return _create_mock_client()

    class OpenAIClient:
        def __init__(self, key: Optional[str]):
            self._client = OpenAI(api_key=key)

        def complete(self, prompt: str, config: LLMConfig) -> LLMResponse:
            start = time.time()
            try:
                response = self._client.chat.completions.create(
                    model=config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    timeout=config.timeout_seconds,
                )
                latency = (time.time() - start) * 1000
                return LLMResponse(
                    content=response.choices[0].message.content or "",
                    model=response.model,
                    provider=LLMProvider.OPENAI,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    },
                    latency_ms=latency,
                    finish_reason=response.choices[0].finish_reason,
                    raw_response=response.model_dump(),
                )
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                raise

        def stream_complete(self, prompt: str, config: LLMConfig) -> Iterator[str]:
            response = self._client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=True,
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    return OpenAIClient(api_key)


def _create_anthropic_client(api_key: Optional[str] = None) -> LLMClient:
    """Create an Anthropic client."""
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed, falling back to mock client")
        return _create_mock_client()

    class AnthropicClient:
        def __init__(self, key: Optional[str]):
            self._client = anthropic.Anthropic(api_key=key)

        def complete(self, prompt: str, config: LLMConfig) -> LLMResponse:
            start = time.time()
            try:
                response = self._client.messages.create(
                    model=config.model,
                    max_tokens=config.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=config.temperature,
                )
                latency = (time.time() - start) * 1000
                return LLMResponse(
                    content=response.content[0].text if response.content else "",
                    model=response.model,
                    provider=LLMProvider.ANTHROPIC,
                    usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                    },
                    latency_ms=latency,
                )
            except Exception as e:
                logger.error(f"Anthropic API error: {e}")
                raise

        def stream_complete(self, prompt: str, config: LLMConfig) -> Iterator[str]:
            with self._client.messages.stream(
                model=config.model,
                max_tokens=config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
            ) as stream:
                for text in stream.text_stream:
                    yield text

    return AnthropicClient(api_key)


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client based on configuration.

    Args:
        config: LLM configuration

    Returns:
        LLM client instance
    """
    if config.provider == LLMProvider.MOCK:
        return _create_mock_client()
    elif config.provider == LLMProvider.OPENAI:
        return _create_openai_client(config.api_key)
    elif config.provider == LLMProvider.ANTHROPIC:
        return _create_anthropic_client(config.api_key)
    elif config.provider == LLMProvider.LOCAL:
        # Local models would use OpenAI-compatible API
        return _create_openai_client(config.api_key)
    else:
        return _create_mock_client()


class LLMAnalyzer:
    """Enhanced analyzer that uses LLMs for deeper analysis."""

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        prompt_config: Optional[AnalysisPromptConfig] = None,
    ):
        self._config = config or LLMConfig()
        self._prompt_config = prompt_config or AnalysisPromptConfig()
        self._client = create_llm_client(self._config)

    def analyze_with_llm(
        self,
        question: str,
        context: Sequence[str],
        analysis_result: Optional[RAGAnalysisResult] = None,
        use_hierarchical: bool = True,
    ) -> LLMResponse:
        """Perform LLM-augmented analysis.

        Args:
            question: The analysis question
            context: Retrieved context strings
            analysis_result: Optional existing RAG result
            use_hierarchical: Whether to use hierarchical prompting

        Returns:
            LLM response with analysis
        """
        if use_hierarchical and analysis_result:
            prompt = build_hierarchical_prompt(analysis_result)
        else:
            prompt = build_adaptive_prompt(
                question=question,
                context=context,
                analysis_result=analysis_result,
                config=self._prompt_config,
            )

        return self._call_with_retry(prompt)

    def analyze_root_cause(
        self,
        failure_description: str,
        context: Sequence[str],
    ) -> LLMResponse:
        """Analyze root cause of a failure.

        Args:
            failure_description: Description of the failure
            context: Retrieved context

        Returns:
            LLM response with root cause analysis
        """
        prompt = build_cot_prompt(
            question=f"Analyze the root cause of: {failure_description}",
            context=context,
            analysis_type="root_cause",
        )
        return self._call_with_retry(prompt)

    def analyze_test_gaps(
        self,
        error_type: str,
        context: Sequence[str],
    ) -> LLMResponse:
        """Analyze test gaps for an error type.

        Args:
            error_type: Type of error
            context: Retrieved context

        Returns:
            LLM response with test gap analysis
        """
        prompt = build_cot_prompt(
            question=f"Identify test gaps for: {error_type}",
            context=context,
            analysis_type="test_gap",
        )
        return self._call_with_retry(prompt)

    def extract_structured_analysis(
        self,
        prompt: str,
        schema: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Extract structured analysis from LLM response.

        Args:
            prompt: The prompt to send
            schema: Expected output schema

        Returns:
            Parsed structured response or None
        """
        response = self._call_with_retry(prompt)
        return response.extract_json(schema)

    def _call_with_retry(self, prompt: str) -> LLMResponse:
        """Call LLM with retry logic.

        Args:
            prompt: The prompt to send

        Returns:
            LLM response

        Raises:
            Exception: If all retries fail
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries + 1):
            try:
                return self._client.complete(prompt, self._config)
            except Exception as e:
                last_error = e
                if attempt < self._config.max_retries:
                    delay = self._config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(f"LLM call failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)

        raise last_error or Exception("LLM call failed")

    @property
    def config(self) -> LLMConfig:
        """Get current LLM configuration."""
        return self._config

    @property
    def prompt_config(self) -> AnalysisPromptConfig:
        """Get current prompt configuration."""
        return self._prompt_config


def create_analyzer(
    provider: LLMProvider = LLMProvider.MOCK,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> LLMAnalyzer:
    """Convenience function to create an LLM analyzer.

    Args:
        provider: LLM provider to use
        model: Model name
        api_key: API key for the provider

    Returns:
        Configured LLMAnalyzer instance
    """
    config = LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
    )
    return LLMAnalyzer(config=config)
