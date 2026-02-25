from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FailureRecord:
    suite: str
    test_name: str
    file_path: str
    error_type: str
    message: str
    traceback_excerpt: str


@dataclass
class FailureCluster:
    cluster_id: str
    reason: str
    error_type: str
    count: int
    tests: List[str] = field(default_factory=list)


@dataclass
class FixSuggestion:
    priority: str
    title: str
    rationale: str
    related_cluster_ids: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    input_format: str
    total_failures: int
    failures: List[FailureRecord] = field(default_factory=list)
    clusters: List[FailureCluster] = field(default_factory=list)
    root_cause_hypotheses: List[str] = field(default_factory=list)
    fix_suggestions: List[FixSuggestion] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
