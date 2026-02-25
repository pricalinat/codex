"""Confidence intervals and uncertainty quantification for RAG analysis.

This module provides sophisticated confidence analysis including:
- Confidence intervals based on retrieval quality
- Uncertainty quantification from multiple factors
- Calibration assessment (how well confidence matches actual accuracy)
- Meta-confidence (confidence in the confidence scores)
- Risk-adjusted confidence for decision making
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import AnalysisResult, FailureCluster
from .retrieval import RetrievalEvidence, RankedChunk


class ConfidenceBand(str, Enum):
    """Confidence band categories."""
    VERY_HIGH = "very_high"  # > 0.85
    HIGH = "high"  # 0.70 - 0.85
    MEDIUM = "medium"  # 0.50 - 0.70
    LOW = "low"  # 0.30 - 0.50
    VERY_LOW = "very_low"  # < 0.30


class UncertaintyType(str, Enum):
    """Types of uncertainty in analysis."""
    ALEATORIC = "aleatoric"  # Inherent randomness/irreducible
    EPISTEMIC = "epistemic"  # Due to lack of knowledge (reducible)
    MODEL = "model"  # Uncertainty from model/retrieval quality
    DATA = "data"  # Uncertainty from missing/partial data


@dataclass
class ConfidenceInterval:
    """A confidence interval for analysis confidence."""

    point_estimate: float  # The central confidence value
    lower_bound: float  # Lower bound (typically 5th percentile)
    upper_bound: float  # Upper bound (typically 95th percentile)
    confidence_level: float  # e.g., 0.90 for 90% confidence interval

    # Uncertainty breakdown
    aleatoric_uncertainty: float = 0.0
    epistemic_uncertainty: float = 0.0
    model_uncertainty: float = 0.0
    data_uncertainty: float = 0.0

    # Quality indicators
    sample_size: int = 0
    retrieval_count: int = 0
    source_diversity: int = 0

    @property
    def interval_width(self) -> float:
        """Width of the confidence interval."""
        return self.upper_bound - self.lower_bound

    @property
    def relative_uncertainty(self) -> float:
        """Relative uncertainty as ratio of width to point estimate."""
        if self.point_estimate == 0:
            return 1.0
        return self.interval_width / self.point_estimate

    @property
    def band(self) -> ConfidenceBand:
        """Get the confidence band category."""
        if self.point_estimate > 0.85:
            return ConfidenceBand.VERY_HIGH
        elif self.point_estimate > 0.70:
            return ConfidenceBand.HIGH
        elif self.point_estimate > 0.50:
            return ConfidenceBand.MEDIUM
        elif self.point_estimate > 0.30:
            return ConfidenceBand.LOW
        return ConfidenceBand.VERY_LOW

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "point_estimate": round(self.point_estimate, 3),
            "lower_bound": round(self.lower_bound, 3),
            "upper_bound": round(self.upper_bound, 3),
            "confidence_level": self.confidence_level,
            "interval_width": round(self.interval_width, 3),
            "relative_uncertainty": round(self.relative_uncertainty, 3),
            "band": self.band.value,
            "uncertainty_breakdown": {
                "aleatoric": round(self.aleatoric_uncertainty, 3),
                "epistemic": round(self.epistemic_uncertainty, 3),
                "model": round(self.model_uncertainty, 3),
                "data": round(self.data_uncertainty, 3),
            },
            "quality_indicators": {
                "sample_size": self.sample_size,
                "retrieval_count": self.retrieval_count,
                "source_diversity": self.source_diversity,
            },
        }


@dataclass
class CalibrationMetrics:
    """Metrics for assessing confidence calibration quality."""

    # Calibration metrics
    expected_calibration_error: float = 0.0  # ECE - average calibration error
    maximum_calibration_error: float = 0.0  # MCE - worst case calibration error
    calibration_slope: float = 1.0  # Should be close to 1.0
    calibration_intercept: float = 0.0  # Should be close to 0.0

    # Reliability metrics
    n_bins: int = 10  # Number of confidence bins
    bin_counts: List[int] = field(default_factory=list)
    bin_accuracies: List[float] = field(default_factory=list)
    bin_confidences: List[float] = field(default_factory=list)

    # Overall calibration assessment
    is_well_calibrated: bool = True
    calibration_grade: str = "unknown"  # excellent, good, fair, poor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "expected_calibration_error": round(self.expected_calibration_error, 3),
            "maximum_calibration_error": round(self.maximum_calibration_error, 3),
            "calibration_slope": round(self.calibration_slope, 3),
            "calibration_intercept": round(self.calibration_intercept, 3),
            "n_bins": self.n_bins,
            "bin_counts": self.bin_counts,
            "bin_accuracies": [round(a, 3) for a in self.bin_accuracies],
            "bin_confidences": [round(c, 3) for c in self.bin_confidences],
            "is_well_calibrated": self.is_well_calibrated,
            "calibration_grade": self.calibration_grade,
        }


@dataclass
class MetaConfidenceResult:
    """Meta-confidence analysis result."""

    # Primary confidence interval
    confidence_interval: ConfidenceInterval

    # Calibration assessment
    calibration: CalibrationMetrics

    # Risk-adjusted confidence
    risk_adjusted_confidence: float
    risk_factors: Dict[str, float] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Summary
    overall_reliability: float = 0.0  # 0-1 rating of confidence reliability
    reliability_grade: str = "unknown"  # excellent, good, fair, poor, unknown

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "confidence_interval": self.confidence_interval.to_dict(),
            "calibration": self.calibration.to_dict(),
            "risk_adjusted_confidence": self.risk_adjusted_confidence,
            "risk_factors": self.risk_factors,
            "recommendations": self.recommendations,
            "overall_reliability": self.overall_reliability,
            "reliability_grade": self.reliability_grade,
        }


class ConfidenceIntervalAnalyzer:
    """Analyzer for computing confidence intervals and uncertainty quantification."""

    # Standard deviation multipliers for confidence intervals
    _Z_SCORES = {
        0.80: 1.28,
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }

    # Minimum samples needed for reliable statistics
    _MIN_SAMPLES_FOR_STATS = 5
    _MIN_SAMPLES_FOR_RELIABLE_INTERVAL = 10

    def __init__(self) -> None:
        self._calibration_history: List[Tuple[float, float]] = []  # (confidence, accuracy)

    def analyze(
        self,
        base_confidence: float,
        retrieval_evidence: Optional[RetrievalEvidence] = None,
        analysis_result: Optional[AnalysisResult] = None,
        context_chunks: Optional[Sequence[RankedChunk]] = None,
    ) -> MetaConfidenceResult:
        """Analyze confidence and produce meta-confidence metrics.

        Args:
            base_confidence: The base confidence score (e.g., from retrieval)
            retrieval_evidence: Optional retrieval evidence for detailed analysis
            analysis_result: Optional analysis result for failure context
            context_chunks: Optional retrieved chunks for context quality

        Returns:
            MetaConfidenceResult with confidence intervals and uncertainty analysis
        """
        # Calculate uncertainty components
        uncertainties = self._calculate_uncertainties(
            base_confidence, retrieval_evidence, analysis_result, context_chunks
        )

        # Build confidence interval
        interval = self._build_confidence_interval(
            base_confidence, uncertainties, retrieval_evidence, context_chunks
        )

        # Assess calibration
        calibration = self._assess_calibration(base_confidence, analysis_result)

        # Calculate risk-adjusted confidence
        risk_adjusted, risk_factors = self._calculate_risk_adjusted_confidence(
            base_confidence, interval, retrieval_evidence, analysis_result
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            interval, calibration, risk_adjusted, retrieval_evidence
        )

        # Calculate overall reliability
        reliability, grade = self._calculate_reliability(interval, calibration)

        return MetaConfidenceResult(
            confidence_interval=interval,
            calibration=calibration,
            risk_adjusted_confidence=risk_adjusted,
            risk_factors=risk_factors,
            recommendations=recommendations,
            overall_reliability=reliability,
            reliability_grade=grade,
        )

    def _calculate_uncertainties(
        self,
        base_confidence: float,
        retrieval_evidence: Optional[RetrievalEvidence],
        analysis_result: Optional[AnalysisResult],
        context_chunks: Optional[Sequence[RankedChunk]],
    ) -> Dict[str, float]:
        """Calculate different types of uncertainty."""
        uncertainties = {
            "aleatoric": 0.0,
            "epistemic": 0.0,
            "model": 0.0,
            "data": 0.0,
        }

        # Model uncertainty based on retrieval quality
        if retrieval_evidence:
            # Use calibrated confidence vs raw confidence gap as model uncertainty
            raw = retrieval_evidence.aggregate_confidence
            calibrated = retrieval_evidence.calibrated_confidence
            model_uncertainty = abs(raw - calibrated) * 0.5
            uncertainties["model"] = min(0.3, model_uncertainty)

            # Low confidence band increases model uncertainty
            if retrieval_evidence.calibrated_confidence < 0.45:
                uncertainties["model"] += 0.15

            # Source coverage affects epistemic uncertainty
            missing_sources = len(retrieval_evidence.missing_source_types)
            total_sources = len(retrieval_evidence.covered_source_types) + missing_sources
            if total_sources > 0:
                source_gap = missing_sources / total_sources
                uncertainties["epistemic"] = min(0.25, source_gap * 0.3)

            # Modality coverage affects data uncertainty
            missing_modalities = len(retrieval_evidence.missing_modalities)
            total_modalities = len(retrieval_evidence.covered_modalities) + missing_modalities
            if total_modalities > 0:
                modality_gap = missing_modalities / total_modalities
                uncertainties["data"] = min(0.2, modality_gap * 0.25)

        # Base aleatoric uncertainty (inherent in the problem)
        # Lower for structured data, higher for ambiguous failures
        uncertainties["aleatoric"] = 0.08  # Base level

        if analysis_result:
            # More clusters = more uncertainty
            cluster_count = len(analysis_result.clusters)
            if cluster_count > 5:
                uncertainties["aleatoric"] += 0.05

            # Unknown error types increase uncertainty
            unknown_count = sum(
                1 for c in analysis_result.clusters
                if "Unknown" in c.error_type or "RuntimeError" in c.error_type
            )
            if unknown_count > 0:
                uncertainties["aleatoric"] += 0.03 * unknown_count

        # Context chunk quality affects epistemic uncertainty
        if context_chunks:
            chunk_count = len(context_chunks)
            if chunk_count < 3:
                uncertainties["epistemic"] += 0.1 * (3 - chunk_count) / 3

            # Low-scoring chunks increase uncertainty
            low_score_count = sum(1 for c in context_chunks if c.score < 0.3)
            if chunk_count > 0:
                low_ratio = low_score_count / chunk_count
                uncertainties["epistemic"] += min(0.15, low_ratio * 0.2)

        return uncertainties

    def _build_confidence_interval(
        self,
        base_confidence: float,
        uncertainties: Dict[str, float],
        retrieval_evidence: Optional[RetrievalEvidence],
        context_chunks: Optional[Sequence[RankedChunk]],
    ) -> ConfidenceInterval:
        """Build a confidence interval from base confidence and uncertainties."""
        # Total uncertainty is sum of components (conservative approach)
        total_uncertainty = sum(uncertainties.values())

        # Calculate interval width based on uncertainty
        # Higher uncertainty = wider interval
        z_score = self._Z_SCORES.get(0.90, 1.645)  # 90% confidence level
        interval_width = z_score * total_uncertainty * 0.5

        # Adjust width based on sample size
        sample_size = 0
        retrieval_count = 0
        source_diversity = 0

        if context_chunks:
            retrieval_count = len(context_chunks)
            sample_size = retrieval_count

            # Count unique sources
            sources = set(c.chunk.source_id for c in context_chunks)
            source_diversity = len(sources)

            # More samples = narrower interval
            if sample_size < self._MIN_SAMPLES_FOR_RELIABLE_INTERVAL:
                interval_width *= 1.0 + (self._MIN_SAMPLES_FOR_RELIABLE_INTERVAL - sample_size) / 20

        # Calculate bounds
        lower_bound = max(0.0, base_confidence - interval_width)
        upper_bound = min(1.0, base_confidence + interval_width)

        return ConfidenceInterval(
            point_estimate=base_confidence,
            lower_bound=round(lower_bound, 3),
            upper_bound=round(upper_bound, 3),
            confidence_level=0.90,
            aleatoric_uncertainty=uncertainties["aleatoric"],
            epistemic_uncertainty=uncertainties["epistemic"],
            model_uncertainty=uncertainties["model"],
            data_uncertainty=uncertainties["data"],
            sample_size=sample_size,
            retrieval_count=retrieval_count,
            source_diversity=source_diversity,
        )

    def _assess_calibration(
        self,
        confidence: float,
        analysis_result: Optional[AnalysisResult],
    ) -> CalibrationMetrics:
        """Assess how well confidence scores are calibrated."""
        metrics = CalibrationMetrics()

        # For now, use heuristic-based calibration assessment
        # In production, this would track historical accuracy

        # Bin the confidence
        bin_size = 1.0 / metrics.n_bins
        bin_index = min(metrics.n_bins - 1, int(confidence / bin_size))
        metrics.bin_counts = [0] * metrics.n_bins
        metrics.bin_confidences = [0.0] * metrics.n_bins
        metrics.bin_accuracies = [0.0] * metrics.n_bins

        metrics.bin_counts[bin_index] = 1
        metrics.bin_confidences[bin_index] = confidence

        # Estimate accuracy based on heuristics
        if analysis_result:
            # Higher cluster count = harder problem = likely lower accuracy
            cluster_count = len(analysis_result.clusters)
            if cluster_count <= 2:
                estimated_accuracy = confidence * 1.05  # Likely well-calibrated
            elif cluster_count <= 5:
                estimated_accuracy = confidence * 0.95
            else:
                estimated_accuracy = confidence * 0.85  # More uncertainty
        else:
            estimated_accuracy = confidence * 0.9  # Default assumption

        metrics.bin_accuracies[bin_index] = estimated_accuracy

        # Calculate ECE (Expected Calibration Error)
        total = sum(metrics.bin_counts)
        if total > 0:
            weighted_error = 0.0
            for i in range(metrics.n_bins):
                if metrics.bin_counts[i] > 0:
                    bin_conf = metrics.bin_confidences[i]
                    bin_acc = metrics.bin_accuracies[i]
                    weight = metrics.bin_counts[i] / total
                    weighted_error += weight * abs(bin_conf - bin_acc)

            metrics.expected_calibration_error = round(weighted_error, 3)
            metrics.maximum_calibration_error = abs(confidence - estimated_accuracy)

        # Assess calibration quality
        calibration_error = metrics.expected_calibration_error
        if calibration_error < 0.05:
            metrics.is_well_calibrated = True
            metrics.calibration_grade = "excellent"
        elif calibration_error < 0.10:
            metrics.is_well_calibrated = True
            metrics.calibration_grade = "good"
        elif calibration_error < 0.20:
            metrics.is_well_calibrated = False
            metrics.calibration_grade = "fair"
        else:
            metrics.is_well_calibrated = False
            metrics.calibration_grade = "poor"

        # Estimate slope and intercept (simplified)
        if confidence > 0.5:
            metrics.calibration_slope = 0.95
            metrics.calibration_intercept = 0.02
        else:
            metrics.calibration_slope = 1.05
            metrics.calibration_intercept = -0.02

        return metrics

    def _calculate_risk_adjusted_confidence(
        self,
        base_confidence: float,
        interval: ConfidenceInterval,
        retrieval_evidence: Optional[RetrievalEvidence],
        analysis_result: Optional[AnalysisResult],
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate risk-adjusted confidence with risk factors."""
        risk_factors: Dict[str, float] = {}
        risk_multiplier = 1.0

        # Wide interval = higher risk
        if interval.interval_width > 0.4:
            risk_factors["wide_interval"] = 0.15
            risk_multiplier -= 0.15
        elif interval.interval_width > 0.25:
            risk_factors["wide_interval"] = 0.08
            risk_multiplier -= 0.08

        # Low point estimate = higher risk
        if interval.point_estimate < 0.4:
            risk_factors["low_confidence"] = 0.2
            risk_multiplier -= 0.2
        elif interval.point_estimate < 0.6:
            risk_factors["low_confidence"] = 0.1
            risk_multiplier -= 0.1

        # High model uncertainty = higher risk
        if interval.model_uncertainty > 0.2:
            risk_factors["model_uncertainty"] = 0.12
            risk_multiplier -= 0.12

        # High epistemic uncertainty = higher risk
        if interval.epistemic_uncertainty > 0.15:
            risk_factors["epistemic_uncertainty"] = 0.08
            risk_multiplier -= 0.08

        # Missing source types = higher risk
        if retrieval_evidence and retrieval_evidence.missing_source_types:
            missing_count = len(retrieval_evidence.missing_source_types)
            risk_factors["missing_sources"] = min(0.15, missing_count * 0.05)
            risk_multiplier -= risk_factors["missing_sources"]

        # Many failure clusters = higher risk
        if analysis_result and len(analysis_result.clusters) > 5:
            cluster_risk = min(0.1, (len(analysis_result.clusters) - 5) * 0.02)
            risk_factors["complex_failure_pattern"] = cluster_risk
            risk_multiplier -= cluster_risk

        # Ensure reasonable bounds
        risk_multiplier = max(0.3, min(1.0, risk_multiplier))
        risk_adjusted = round(base_confidence * risk_multiplier, 3)

        return risk_adjusted, risk_factors

    def _generate_recommendations(
        self,
        interval: ConfidenceInterval,
        calibration: CalibrationMetrics,
        risk_adjusted: float,
        retrieval_evidence: Optional[RetrievalEvidence],
    ) -> List[str]:
        """Generate recommendations based on confidence analysis."""
        recommendations = []

        # Interval-based recommendations
        if interval.interval_width > 0.35:
            recommendations.append(
                "Wide confidence interval suggests high uncertainty - consider gathering more evidence"
            )

        if interval.point_estimate < 0.5:
            recommendations.append(
                "Low base confidence - review retrieval quality and corpus coverage"
            )

        # Source-based recommendations
        if retrieval_evidence:
            if retrieval_evidence.missing_source_types:
                recommendations.append(
                    f"Missing source types: {', '.join(s.value for s in retrieval_evidence.missing_source_types)}. "
                    "Consider adding these sources to improve analysis."
                )

            if retrieval_evidence.missing_modalities:
                recommendations.append(
                    f"Missing modalities: {', '.join(retrieval_evidence.missing_modalities)}. "
                    "Consider adding multimodal content."
                )

            if retrieval_evidence.calibrated_confidence < 0.45:
                recommendations.append(
                    "Low calibrated confidence - check retrieval query and corpus relevance"
                )

        # Risk-based recommendations
        if risk_adjusted < 0.4:
            recommendations.append(
                "High risk level - consider manual review before taking action"
            )

        # Calibration-based recommendations
        if calibration.calibration_grade in ("fair", "poor"):
            recommendations.append(
                f"Calibration is {calibration.calibration_grade} - confidence may not reflect actual accuracy"
            )

        # Sample size recommendations
        if interval.retrieval_count < 5:
            recommendations.append(
                "Low retrieval count - consider broader queries or larger corpus"
            )

        if interval.source_diversity < 2:
            recommendations.append(
                "Low source diversity - consider adding more diverse sources"
            )

        if not recommendations:
            recommendations.append("Confidence levels look good - proceed with analysis")

        return recommendations

    def _calculate_reliability(
        self,
        interval: ConfidenceInterval,
        calibration: CalibrationMetrics,
    ) -> Tuple[float, str]:
        """Calculate overall reliability score and grade."""
        # Weight factors
        interval_score = 1.0 - min(1.0, interval.relative_uncertainty * 2)
        calibration_score = 1.0 - calibration.expected_calibration_error * 2

        # Combine scores
        reliability = (interval_score * 0.6) + (calibration_score * 0.4)
        reliability = max(0.0, min(1.0, reliability))

        # Determine grade
        if reliability > 0.85:
            grade = "excellent"
        elif reliability > 0.70:
            grade = "good"
        elif reliability > 0.50:
            grade = "fair"
        elif reliability > 0.30:
            grade = "poor"
        else:
            grade = "unknown"

        return round(reliability, 3), grade


def analyze_confidence(
    base_confidence: float,
    retrieval_evidence: Optional[RetrievalEvidence] = None,
    analysis_result: Optional[AnalysisResult] = None,
    context_chunks: Optional[Sequence[RankedChunk]] = None,
) -> MetaConfidenceResult:
    """Convenience function for confidence analysis.

    Args:
        base_confidence: Base confidence score
        retrieval_evidence: Optional retrieval evidence
        analysis_result: Optional analysis result
        context_chunks: Optional retrieved chunks

    Returns:
        MetaConfidenceResult with full confidence analysis
    """
    analyzer = ConfidenceIntervalAnalyzer()
    return analyzer.analyze(base_confidence, retrieval_evidence, analysis_result, context_chunks)
