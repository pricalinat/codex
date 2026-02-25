"""Tests for confidence_analysis module."""

import pytest

from test_analysis_assistant.confidence_analysis import (
    CalibrationMetrics,
    ConfidenceBand,
    ConfidenceInterval,
    ConfidenceIntervalAnalyzer,
    MetaConfidenceResult,
    UncertaintyType,
    analyze_confidence,
)


class TestConfidenceInterval:
    """Tests for ConfidenceInterval class."""

    def test_interval_width(self):
        """Test interval width calculation."""
        interval = ConfidenceInterval(
            point_estimate=0.7,
            lower_bound=0.5,
            upper_bound=0.9,
            confidence_level=0.90,
        )
        assert interval.interval_width == 0.4

    def test_relative_uncertainty(self):
        """Test relative uncertainty calculation."""
        interval = ConfidenceInterval(
            point_estimate=0.5,
            lower_bound=0.3,
            upper_bound=0.7,
            confidence_level=0.90,
        )
        assert abs(interval.relative_uncertainty - 0.8) < 0.001  # 0.4 / 0.5

    def test_confidence_band_very_high(self):
        """Test confidence band for very high confidence."""
        interval = ConfidenceInterval(
            point_estimate=0.9,
            lower_bound=0.8,
            upper_bound=1.0,
            confidence_level=0.90,
        )
        assert interval.band == ConfidenceBand.VERY_HIGH

    def test_confidence_band_high(self):
        """Test confidence band for high confidence."""
        interval = ConfidenceInterval(
            point_estimate=0.75,
            lower_bound=0.65,
            upper_bound=0.85,
            confidence_level=0.90,
        )
        assert interval.band == ConfidenceBand.HIGH

    def test_confidence_band_medium(self):
        """Test confidence band for medium confidence."""
        interval = ConfidenceInterval(
            point_estimate=0.6,
            lower_bound=0.45,
            upper_bound=0.75,
            confidence_level=0.90,
        )
        assert interval.band == ConfidenceBand.MEDIUM

    def test_confidence_band_low(self):
        """Test confidence band for low confidence."""
        interval = ConfidenceInterval(
            point_estimate=0.4,
            lower_bound=0.25,
            upper_bound=0.55,
            confidence_level=0.90,
        )
        assert interval.band == ConfidenceBand.LOW

    def test_confidence_band_very_low(self):
        """Test confidence band for very low confidence."""
        interval = ConfidenceInterval(
            point_estimate=0.2,
            lower_bound=0.05,
            upper_bound=0.35,
            confidence_level=0.90,
        )
        assert interval.band == ConfidenceBand.VERY_LOW

    def test_to_dict(self):
        """Test serialization to dictionary."""
        interval = ConfidenceInterval(
            point_estimate=0.7,
            lower_bound=0.5,
            upper_bound=0.9,
            confidence_level=0.90,
            aleatoric_uncertainty=0.1,
            epistemic_uncertainty=0.05,
            model_uncertainty=0.08,
            data_uncertainty=0.07,
            sample_size=10,
            retrieval_count=5,
            source_diversity=3,
        )
        d = interval.to_dict()
        assert d["point_estimate"] == 0.7
        assert d["lower_bound"] == 0.5
        assert d["upper_bound"] == 0.9
        assert d["confidence_level"] == 0.90
        assert d["interval_width"] == 0.4
        assert d["band"] == "medium"
        assert d["quality_indicators"]["sample_size"] == 10


class TestCalibrationMetrics:
    """Tests for CalibrationMetrics class."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = CalibrationMetrics(
            expected_calibration_error=0.08,
            maximum_calibration_error=0.15,
            calibration_slope=0.95,
            calibration_intercept=0.02,
            is_well_calibrated=True,
            calibration_grade="good",
            bin_counts=[1, 2, 3],
            bin_accuracies=[0.8, 0.7, 0.6],
            bin_confidences=[0.85, 0.65, 0.45],
        )
        d = metrics.to_dict()
        assert d["expected_calibration_error"] == 0.08
        assert d["calibration_grade"] == "good"
        assert d["is_well_calibrated"] is True


class TestConfidenceIntervalAnalyzer:
    """Tests for ConfidenceIntervalAnalyzer class."""

    def test_analyze_basic(self):
        """Test basic confidence analysis."""
        analyzer = ConfidenceIntervalAnalyzer()
        result = analyzer.analyze(base_confidence=0.75)

        assert isinstance(result, MetaConfidenceResult)
        assert isinstance(result.confidence_interval, ConfidenceInterval)
        assert result.confidence_interval.point_estimate == 0.75
        assert isinstance(result.calibration, CalibrationMetrics)
        assert 0 <= result.risk_adjusted_confidence <= 1
        assert result.overall_reliability > 0

    def test_analyze_with_retrieval_evidence(self):
        """Test confidence analysis with retrieval evidence."""
        from test_analysis_assistant.retrieval import QueryPlan, SourceType

        analyzer = ConfidenceIntervalAnalyzer()

        # Create mock retrieval evidence
        mock_evidence = type('MockEvidence', (), {
            'aggregate_confidence': 0.8,
            'calibrated_confidence': 0.75,
            'missing_source_types': [SourceType.REQUIREMENTS],
            'missing_modalities': [],
            'covered_source_types': [SourceType.REPOSITORY],
            'covered_modalities': ['text'],
            'unavailable_preferred_source_types': [],
            'unavailable_preferred_modalities': [],
        })()

        result = analyzer.analyze(
            base_confidence=0.75,
            retrieval_evidence=mock_evidence,
        )

        assert result.confidence_interval.epistemic_uncertainty > 0
        assert len(result.recommendations) > 0

    def test_analyze_with_analysis_result(self):
        """Test confidence analysis with analysis result."""
        from test_analysis_assistant.models import AnalysisResult, FailureCluster, FailureRecord

        analyzer = ConfidenceIntervalAnalyzer()

        # Create mock analysis result with multiple clusters
        mock_result = AnalysisResult(
            input_format="pytest",
            total_failures=10,
            failures=[
                FailureRecord(
                    suite="test_suite",
                    test_name="test_example",
                    file_path="test_example.py",
                    error_type="AssertionError",
                    message="expected True, got False",
                    traceback_excerpt="",
                )
            ],
            clusters=[
                FailureCluster(
                    cluster_id="cluster_1",
                    error_type="AssertionError",
                    count=10,
                    reason="assertion failure",
                    tests=["test_example"],
                )
            ],
        )

        result = analyzer.analyze(
            base_confidence=0.6,
            analysis_result=mock_result,
        )

        assert result.confidence_interval.aleatoric_uncertainty > 0

    def test_risk_factors(self):
        """Test risk factor calculation."""
        analyzer = ConfidenceIntervalAnalyzer()

        result = analyzer.analyze(
            base_confidence=0.3,  # Low confidence
        )

        assert "low_confidence" in result.risk_factors
        assert result.risk_adjusted_confidence < 0.3

    def test_wide_interval(self):
        """Test wide interval detection."""
        from test_analysis_assistant.retrieval import QueryPlan, SourceType

        analyzer = ConfidenceIntervalAnalyzer()

        # Create mock evidence with missing sources
        mock_evidence = type('MockEvidence', (), {
            'aggregate_confidence': 0.9,
            'calibrated_confidence': 0.3,  # Large gap = high model uncertainty
            'missing_source_types': [SourceType.REQUIREMENTS, SourceType.SYSTEM_ANALYSIS],
            'missing_modalities': ['image'],
            'covered_source_types': [SourceType.REPOSITORY],
            'covered_modalities': ['text'],
            'unavailable_preferred_source_types': [],
            'unavailable_preferred_modalities': [],
        })()

        result = analyzer.analyze(
            base_confidence=0.5,
            retrieval_evidence=mock_evidence,
        )

        assert result.confidence_interval.interval_width > 0.2
        assert len(result.recommendations) > 0


class TestAnalyzeConfidence:
    """Tests for convenience function analyze_confidence."""

    def test_convenience_function(self):
        """Test the convenience function."""
        result = analyze_confidence(base_confidence=0.8)

        assert isinstance(result, MetaConfidenceResult)
        assert result.confidence_interval.point_estimate == 0.8

    def test_with_context_chunks(self):
        """Test with context chunks."""
        from test_analysis_assistant.retrieval import Chunk, SourceType

        # Create mock chunks
        mock_chunks = [
            type('MockChunk', (), {
                'chunk': Chunk(
                    chunk_id=f"chunk_{i}",
                    source_id=f"source_{i}",
                    source_type=SourceType.REPOSITORY,
                    modality="text",
                    text=f"content {i}",
                    token_count=100,
                ),
                'score': 0.8 - (i * 0.1),
                'confidence': 0.9 - (i * 0.1),
                'matched_terms': ['test'],
                'score_breakdown': {},
            })()
            for i in range(3)
        ]

        result = analyze_confidence(
            base_confidence=0.7,
            context_chunks=mock_chunks,
        )

        assert result.confidence_interval.retrieval_count == 3


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_confidence(self):
        """Test handling of zero confidence."""
        analyzer = ConfidenceIntervalAnalyzer()
        result = analyzer.analyze(base_confidence=0.0)

        assert result.confidence_interval.point_estimate == 0.0
        assert result.risk_adjusted_confidence == 0.0

    def test_perfect_confidence(self):
        """Test handling of perfect confidence."""
        analyzer = ConfidenceIntervalAnalyzer()
        result = analyzer.analyze(base_confidence=1.0)

        assert result.confidence_interval.point_estimate == 1.0
        # Even perfect confidence gets some uncertainty
        assert result.confidence_interval.interval_width > 0

    def test_recommendations_for_missing_sources(self):
        """Test recommendations include missing sources."""
        from test_analysis_assistant.retrieval import QueryPlan, SourceType

        analyzer = ConfidenceIntervalAnalyzer()

        mock_evidence = type('MockEvidence', (), {
            'aggregate_confidence': 0.7,
            'calibrated_confidence': 0.65,
            'missing_source_types': [SourceType.REQUIREMENTS, SourceType.KNOWLEDGE],
            'missing_modalities': [],
            'covered_source_types': [SourceType.REPOSITORY],
            'covered_modalities': ['text'],
            'unavailable_preferred_source_types': [],
            'unavailable_preferred_modalities': [],
        })()

        result = analyzer.analyze(
            base_confidence=0.7,
            retrieval_evidence=mock_evidence,
        )

        # Check that recommendations mention missing sources
        rec_text = " ".join(result.recommendations).lower()
        assert "requirements" in rec_text or "missing" in rec_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
