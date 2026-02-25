import argparse
import json
from pathlib import Path

from .analyzer import analyze_report_text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="test-analysis-assistant", description="Analyze test reports.")
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Analyze a test report file.")
    analyze.add_argument("--input", required=True, help="Path to junit xml or pytest text report.")
    analyze.add_argument(
        "--format",
        choices=["json", "pretty"],
        default="json",
        help="Output format: compact json or human-readable json.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "analyze":
        parser.print_help()
        return 2

    report_path = Path(args.input)
    if not report_path.exists():
        print(f"error: input file not found: {report_path}")
        return 1

    try:
        content = report_path.read_text(encoding="utf-8")
        result = analyze_report_text(content)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 1

    indent = 2 if args.format == "pretty" else None
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
