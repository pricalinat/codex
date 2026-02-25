import re
from dataclasses import asdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


TEST_RESULT_RE = re.compile(
    r"^(?P<name>test_[^\s]+)\s+\((?P<module>[^)]+)\)\s+\.\.\.\s+(?P<status>ok|FAIL|ERROR)$"
)
TEST_START_RE = re.compile(r"^(?P<name>test_[^\s]+)\s+\((?P<module>[^)]+)\)$")
TEST_CONT_RE = re.compile(r"^.*\.\.\.\s+(?P<status>ok|FAIL|ERROR)$")
FAIL_HEADER_RE = re.compile(
    r"^(?P<kind>FAIL|ERROR):\s+(?P<name>test_[^\s]+)\s+\((?P<module>[^)]+)\)$"
)
RAN_RE = re.compile(r"^Ran\s+(?P<count>\d+)\s+tests?\s+in\s+(?P<seconds>[\d.]+)s$")
FAILED_RE = re.compile(
    r"^FAILED\s+\((?:failures=(?P<failures>\d+))?(?:,\s*)?(?:errors=(?P<errors>\d+))?\)$"
)


@dataclass
class ParsedTestCase:
    test_name: str
    module_name: str
    status: str


@dataclass
class FailureDetail:
    severity: str
    test_name: str
    module_name: str
    error_type: str
    message: str
    traceback_excerpt: str


@dataclass
class ActionItem:
    priority: str
    title: str
    rationale: str
    command: str


@dataclass
class TestAnalysisReport:
    command: str
    generated_at_utc: str
    overall_status: str
    exit_code: int
    total_tests: int
    passed: int
    failed: int
    errors: int
    duration_seconds: float
    module_breakdown: Dict[str, Dict[str, int]]
    failures: List[FailureDetail] = field(default_factory=list)
    actions: List[ActionItem] = field(default_factory=list)
    residual_risks: List[str] = field(default_factory=list)
    risk_score: int = 0
    trend_summary: Optional[str] = None

    @property
    def pass_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return float(self.passed) / float(self.total_tests)

    def to_markdown(self) -> str:
        lines: List[str] = [
            "# Test Analysis Report",
            "",
            "## 1. Execution Summary",
            f"- `command`: `{self.command}`",
            f"- `generated_at_utc`: `{self.generated_at_utc}`",
            f"- `overall_status`: `{self.overall_status}`",
            f"- `exit_code`: `{self.exit_code}`",
            f"- `total_tests`: `{self.total_tests}`",
            f"- `passed`: `{self.passed}`",
            f"- `failed`: `{self.failed}`",
            f"- `errors`: `{self.errors}`",
            f"- `pass_rate`: `{self.pass_rate:.2%}`",
            f"- `duration_seconds`: `{self.duration_seconds:.3f}`",
            f"- `risk_score` (0-100): `{self.risk_score}`",
        ]
        if self.trend_summary:
            lines.append(f"- `trend`: {self.trend_summary}")
        lines.extend(
            [
                "",
                "## 2. Module Breakdown",
            ]
        )
        for module, stats in sorted(self.module_breakdown.items()):
            lines.append(
                f"- `{module}`: total={stats['total']}, pass={stats['pass']}, fail={stats['fail']}, error={stats['error']}"
            )

        lines.extend(["", "## 3. Key Findings"])
        if not self.failures:
            lines.append("- No blocking failures detected.")
        else:
            for item in self.failures:
                lines.append(
                    f"- [{item.severity}] `{item.test_name}` in `{item.module_name}`: {item.error_type} - {item.message}"
                )

        lines.extend(["", "## 4. Recommended Actions"])
        if not self.actions:
            lines.append("- No immediate action required.")
        else:
            for action in self.actions:
                lines.append(
                    f"- [{action.priority}] {action.title} | why: {action.rationale} | run: `{action.command}`"
                )

        lines.extend(["", "## 5. Residual Risks"])
        if not self.residual_risks:
            lines.append("- No residual risk recorded.")
        else:
            for risk in self.residual_risks:
                lines.append(f"- {risk}")

        return "\n".join(lines) + "\n"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def analyze_unittest_output(command: str, output: str, exit_code: int) -> TestAnalysisReport:
    parsed_tests: List[ParsedTestCase] = []
    failures: List[FailureDetail] = []
    lines = output.splitlines()
    pending_test = None

    for line in lines:
        stripped = line.strip()
        match = TEST_RESULT_RE.match(stripped)
        if match:
            parsed_tests.append(
                ParsedTestCase(
                    test_name=match.group("name"),
                    module_name=match.group("module"),
                    status=match.group("status"),
                )
            )
            pending_test = None
            continue

        start_match = TEST_START_RE.match(stripped)
        if start_match:
            pending_test = (start_match.group("name"), start_match.group("module"))
            continue

        if pending_test:
            cont_match = TEST_CONT_RE.match(stripped)
            if cont_match:
                parsed_tests.append(
                    ParsedTestCase(
                        test_name=pending_test[0],
                        module_name=pending_test[1],
                        status=cont_match.group("status"),
                    )
                )
                pending_test = None

    total_tests = len(parsed_tests)
    passed = sum(1 for item in parsed_tests if item.status == "ok")
    failed = sum(1 for item in parsed_tests if item.status == "FAIL")
    errors = sum(1 for item in parsed_tests if item.status == "ERROR")
    duration_seconds = _parse_duration(lines)

    # Parse failure/error sections with short traceback excerpt.
    i = 0
    while i < len(lines):
        header = FAIL_HEADER_RE.match(lines[i].strip())
        if not header:
            i += 1
            continue

        kind = header.group("kind")
        test_name = header.group("name")
        module_name = header.group("module")
        excerpt_lines: List[str] = []
        i += 1
        while i < len(lines):
            row = lines[i].rstrip("\n")
            if row.startswith("====") or FAIL_HEADER_RE.match(row.strip()):
                break
            if row.strip():
                excerpt_lines.append(row.strip())
            i += 1

        error_type, message = _extract_error_type_and_message(excerpt_lines)
        severity = _classify_severity(error_type, kind)
        failures.append(
            FailureDetail(
                severity=severity,
                test_name=test_name,
                module_name=module_name,
                error_type=error_type,
                message=message,
                traceback_excerpt="\n".join(excerpt_lines[-6:]),
            )
        )

    module_breakdown = _build_module_breakdown(parsed_tests)
    overall_status = "pass" if exit_code == 0 and failed == 0 and errors == 0 else "fail"
    actions = _build_action_items(overall_status, failures)
    residual_risks = _build_residual_risks(overall_status, module_breakdown)
    risk_score = _build_risk_score(passed, failed, errors, total_tests)

    return TestAnalysisReport(
        command=command,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        overall_status=overall_status,
        exit_code=exit_code,
        total_tests=total_tests,
        passed=passed,
        failed=failed,
        errors=errors,
        duration_seconds=duration_seconds,
        module_breakdown=module_breakdown,
        failures=failures,
        actions=actions,
        residual_risks=residual_risks,
        risk_score=risk_score,
    )


def summarize_trend(current: TestAnalysisReport, previous: Optional[Dict[str, object]]) -> str:
    if not previous:
        return "baseline report created"

    prev_total = int(previous.get("total_tests", 0))
    prev_failed = int(previous.get("failed", 0))
    prev_errors = int(previous.get("errors", 0))
    prev_risk = int(previous.get("risk_score", 0))

    return (
        f"tests {current.total_tests - prev_total:+d}, "
        f"failed {current.failed - prev_failed:+d}, "
        f"errors {current.errors - prev_errors:+d}, "
        f"risk {current.risk_score - prev_risk:+d}"
    )


def _parse_duration(lines: List[str]) -> float:
    for line in lines:
        match = RAN_RE.match(line.strip())
        if match:
            return float(match.group("seconds"))
    return 0.0


def _extract_error_type_and_message(excerpt_lines: List[str]) -> (str, str):
    for line in reversed(excerpt_lines):
        if ":" in line:
            left, right = line.split(":", 1)
            if left and left[0].isupper():
                return left.strip(), right.strip() or "no message"
    return "UnknownError", "no message"


def _classify_severity(error_type: str, kind: str) -> str:
    if kind == "ERROR":
        return "P0"
    if error_type in {"IndentationError", "SyntaxError", "ImportError"}:
        return "P0"
    if error_type == "AssertionError":
        return "P1"
    return "P2"


def _build_module_breakdown(parsed_tests: List[ParsedTestCase]) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for item in parsed_tests:
        module = _module_bucket(item.module_name)
        if module not in result:
            result[module] = {"total": 0, "pass": 0, "fail": 0, "error": 0}
        result[module]["total"] += 1
        if item.status == "ok":
            result[module]["pass"] += 1
        elif item.status == "FAIL":
            result[module]["fail"] += 1
        elif item.status == "ERROR":
            result[module]["error"] += 1
    return result


def _module_bucket(module_name: str) -> str:
    if "paper_retrieval" in module_name:
        return "paper_retrieval"
    if "orchestrator" in module_name:
        return "orchestrator"
    if "agents" in module_name:
        return "agents"
    if "e2e" in module_name:
        return "e2e"
    if "store" in module_name:
        return "store"
    return "other"


def _build_action_items(overall_status: str, failures: List[FailureDetail]) -> List[ActionItem]:
    if overall_status == "pass":
        return [
            ActionItem(
                priority="P2",
                title="Preserve current green baseline",
                rationale="All tests passed; keep regression signal stable.",
                command="python3 -m unittest -v",
            )
        ]

    items: List[ActionItem] = [
        ActionItem(
            priority="P0",
            title="Fix blocking test failures",
            rationale="Test suite is failing and blocks reliable release decisions.",
            command="python3 -m unittest -v",
        )
    ]

    if any(item.error_type in {"IndentationError", "SyntaxError"} for item in failures):
        items.append(
            ActionItem(
                priority="P0",
                title="Resolve syntax/parse errors first",
                rationale="Parser errors invalidate downstream test signals.",
                command="python3 -m py_compile src/multi_agent/*.py tests/*.py",
            )
        )

    if any(item.module_name.find("paper_retrieval") >= 0 for item in failures):
        items.append(
            ActionItem(
                priority="P1",
                title="Run retrieval-focused regression subset",
                rationale="Concentrate feedback loop on retrieval risk surface.",
                command="python3 -m unittest tests/test_paper_retrieval.py tests/test_agents.py tests/test_e2e.py -v",
            )
        )

    return items


def _build_residual_risks(overall_status: str, module_breakdown: Dict[str, Dict[str, int]]) -> List[str]:
    if overall_status == "fail":
        return ["Release confidence is low until all blocking failures are resolved."]

    risks: List[str] = []
    if module_breakdown.get("paper_retrieval", {}).get("total", 0) == 0:
        risks.append("No retrieval tests detected in current run; ranking regressions may go unnoticed.")
    if module_breakdown.get("e2e", {}).get("total", 0) == 0:
        risks.append("No e2e tests detected in current run; integration regressions may remain hidden.")
    return risks


def _build_risk_score(passed: int, failed: int, errors: int, total: int) -> int:
    if total <= 0:
        return 100
    raw = (failed * 12) + (errors * 20)
    pass_penalty = int((1.0 - (float(passed) / float(total))) * 40.0)
    return max(0, min(100, raw + pass_penalty))
