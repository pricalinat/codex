import re
import xml.etree.ElementTree as ET
from typing import List, Tuple

from .models import FailureRecord


_PYTEST_FAILURE_HEADER_RE = re.compile(r"^_{3,}\s+(.+?)\s+_{3,}$")
_PYTEST_FILE_LINE_RE = re.compile(r"^(?P<path>[^:]+):\d+:\s+(?P<error_type>[A-Za-z_][\w.]*)")


def detect_format(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("<") and ("<testsuite" in stripped or "<testsuites" in stripped):
        return "junit_xml"
    return "pytest_text"


def parse_input(text: str) -> Tuple[str, List[FailureRecord]]:
    fmt = detect_format(text)
    if fmt == "junit_xml":
        return fmt, parse_junit_xml(text)
    return fmt, parse_pytest_text(text)


def parse_junit_xml(text: str) -> List[FailureRecord]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid junit xml: {exc}") from exc

    testcases = list(root.iter("testcase"))
    failures: List[FailureRecord] = []
    for case in testcases:
        failure_node = case.find("failure")
        error_node = case.find("error")
        node = failure_node if failure_node is not None else error_node
        if node is None:
            continue

        suite = case.get("classname") or "unknown_suite"
        test_name = case.get("name") or "unknown_test"
        file_path = case.get("file") or suite.replace(".", "/") + ".py"
        error_type = node.get("type") or ("AssertionError" if failure_node is not None else "RuntimeError")
        message = (node.get("message") or "").strip() or "no message"
        traceback_excerpt = (node.text or "").strip()

        failures.append(
            FailureRecord(
                suite=suite,
                test_name=test_name,
                file_path=file_path,
                error_type=error_type,
                message=message,
                traceback_excerpt=traceback_excerpt,
            )
        )

    return failures


def parse_pytest_text(text: str) -> List[FailureRecord]:
    lines = text.splitlines()
    failures: List[FailureRecord] = []

    i = 0
    while i < len(lines):
        header = _PYTEST_FAILURE_HEADER_RE.match(lines[i].strip())
        if not header:
            i += 1
            continue

        test_id = header.group(1).strip()
        file_path = "unknown_file"
        error_type = "AssertionError"
        message = "no message"
        excerpt_lines = []

        i += 1
        while i < len(lines):
            current = lines[i]
            stripped = current.strip()

            # next failure block
            if _PYTEST_FAILURE_HEADER_RE.match(stripped):
                break
            if stripped.startswith("=") and "short test summary" in stripped.lower():
                break

            match = _PYTEST_FILE_LINE_RE.match(stripped)
            if match:
                file_path = match.group("path")
                error_type = match.group("error_type")
                message = stripped
            if stripped:
                excerpt_lines.append(stripped)
            i += 1

        suite, test_name = _split_pytest_nodeid(test_id)
        failures.append(
            FailureRecord(
                suite=suite,
                test_name=test_name,
                file_path=file_path,
                error_type=error_type,
                message=message,
                traceback_excerpt="\n".join(excerpt_lines[-8:]),
            )
        )

    return failures


def _split_pytest_nodeid(nodeid: str) -> Tuple[str, str]:
    cleaned = nodeid.strip()
    if "::" in cleaned:
        left, right = cleaned.rsplit("::", 1)
        return left, right
    return "unknown_suite", cleaned
