from collections import defaultdict
from typing import Dict, List

from .models import AnalysisResult, FailureCluster, FailureRecord, FixSuggestion
from .parsers import parse_input


def analyze_report_text(text: str) -> AnalysisResult:
    if not text or not text.strip():
        raise ValueError("Input report is empty.")

    input_format, failures = parse_input(text)
    clusters = _cluster_failures(failures)
    hypotheses = _build_hypotheses(clusters)
    suggestions = _build_suggestions(clusters)

    return AnalysisResult(
        input_format=input_format,
        total_failures=len(failures),
        failures=failures,
        clusters=clusters,
        root_cause_hypotheses=hypotheses,
        fix_suggestions=suggestions,
    )


def _cluster_failures(failures: List[FailureRecord]) -> List[FailureCluster]:
    by_error: Dict[str, List[FailureRecord]] = defaultdict(list)
    for item in failures:
        by_error[item.error_type].append(item)

    clusters: List[FailureCluster] = []
    for idx, (error_type, items) in enumerate(sorted(by_error.items()), start=1):
        reason = _reason_for_error(error_type)
        clusters.append(
            FailureCluster(
                cluster_id=f"C{idx:02d}",
                reason=reason,
                error_type=error_type,
                count=len(items),
                tests=[f"{it.suite}::{it.test_name}" for it in items],
            )
        )
    return clusters


def _reason_for_error(error_type: str) -> str:
    mapping = {
        "AssertionError": "Behavior mismatch or assertion expectation drift",
        "ImportError": "Dependency/import path issue",
        "ModuleNotFoundError": "Missing dependency or wrong environment",
        "TypeError": "API signature mismatch or invalid input type",
        "TimeoutError": "External dependency instability or slow path",
        "RuntimeError": "Runtime branch not handled",
    }
    return mapping.get(error_type, "Potential logic bug requiring traceback inspection")


def _build_hypotheses(clusters: List[FailureCluster]) -> List[str]:
    results: List[str] = []
    for cluster in clusters:
        results.append(
            f"{cluster.cluster_id}: {cluster.error_type} appears in {cluster.count} case(s), likely {cluster.reason.lower()}."
        )
    return results


def _build_suggestions(clusters: List[FailureCluster]) -> List[FixSuggestion]:
    suggestions: List[FixSuggestion] = []
    for cluster in clusters:
        priority = _priority(cluster.error_type, cluster.count)
        suggestions.append(
            FixSuggestion(
                priority=priority,
                title=f"Address {cluster.error_type} cluster",
                rationale=f"{cluster.count} failing case(s) linked to {cluster.error_type}.",
                related_cluster_ids=[cluster.cluster_id],
            )
        )

    order = {"P0": 0, "P1": 1, "P2": 2}
    suggestions.sort(key=lambda x: (order.get(x.priority, 99), x.title))
    return suggestions


def _priority(error_type: str, count: int) -> str:
    if error_type in {"ImportError", "ModuleNotFoundError", "SyntaxError"}:
        return "P0"
    if count >= 3:
        return "P0"
    if error_type in {"TypeError", "TimeoutError", "RuntimeError"}:
        return "P1"
    return "P2"
