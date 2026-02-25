import argparse
import json
import subprocess
from pathlib import Path

from .test_analysis import analyze_unittest_output, summarize_trend


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate test analysis reports from unittest output.")
    parser.add_argument(
        "--command",
        default="python3 -m unittest -v",
        help="Test command to execute.",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Directory where the test command will run.",
    )
    parser.add_argument(
        "--report-dir",
        default="docs/reports",
        help="Directory to write markdown/json reports.",
    )
    parser.add_argument(
        "--basename",
        default="latest-test-analysis",
        help="Base filename for report artifacts (without extension).",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{args.basename}.json"
    md_path = report_dir / f"{args.basename}.md"

    previous = None
    if json_path.exists():
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = None

    proc = subprocess.run(
        args.command,
        shell=True,
        cwd=args.workdir,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    report = analyze_unittest_output(args.command, out, proc.returncode)
    report.trend_summary = summarize_trend(report, previous)

    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    print(str(md_path))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
