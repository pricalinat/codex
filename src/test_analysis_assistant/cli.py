import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .analyzer import analyze_report_text
from .rag_analyzer import RAGAnalyzer, rag_analyze


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test-analysis-assistant",
        description="AI-native test analysis assistant with RAG augmentation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Basic analyze command
    analyze = subparsers.add_parser("analyze", help="Analyze a test report file (basic).")
    analyze.add_argument("--input", required=True, help="Path to junit xml or pytest text report.")
    analyze.add_argument(
        "--format",
        choices=["json", "pretty"],
        default="json",
        help="Output format: compact json or human-readable json.",
    )

    # RAG-augmented analyze command
    rag = subparsers.add_parser("rag-analyze", help="Analyze with RAG augmentation.")
    rag.add_argument("--input", required=True, help="Path to junit xml or pytest text report.")
    rag.add_argument(
        "--repo",
        help="Path to repository for context ingestion.",
    )
    rag.add_argument(
        "--requirements",
        action="append",
        dest="requirements",
        help="Path to requirements markdown file(s). Can be specified multiple times.",
    )
    rag.add_argument(
        "--requirements-text",
        help="Inline requirements text (markdown format).",
    )
    rag.add_argument(
        "--knowledge",
        action="append",
        dest="knowledge",
        help="Path to knowledge document(s). Can be specified multiple times.",
    )
    rag.add_argument(
        "--query",
        help="Additional query for retrieval context.",
    )
    rag.add_argument(
        "--format",
        choices=["json", "pretty"],
        default="json",
        help="Output format: compact json or human-readable json.",
    )
    rag.add_argument(
        "--prompt-only",
        action="store_true",
        help="Only output the augmented prompt, not the full analysis.",
    )

    # Ingest command - index documents without analysis
    ingest = subparsers.add_parser("ingest", help="Ingest documents into the corpus.")
    ingest.add_argument(
        "--repo",
        help="Path to repository for ingestion.",
    )
    ingest.add_argument(
        "--requirements",
        action="append",
        dest="requirements",
        help="Path to requirements markdown file(s).",
    )
    ingest.add_argument(
        "--requirements-text",
        help="Inline requirements text (markdown format).",
    )
    ingest.add_argument(
        "--knowledge",
        action="append",
        dest="knowledge",
        help="Path to knowledge document(s).",
    )

    return parser


def _load_docs_from_paths(paths: Optional[Sequence[str]]) -> Sequence[tuple]:
    """Load documents from file paths."""
    if not paths:
        return []
    docs = []
    for path_str in paths:
        path = Path(path_str)
        if path.exists():
            content = path.read_text(encoding="utf-8")
            docs.append((f"doc:{path.name}", content))
    return docs


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        return _handle_basic_analyze(args)
    elif args.command == "rag-analyze":
        return _handle_rag_analyze(args)
    elif args.command == "ingest":
        return _handle_ingest(args)
    else:
        parser.print_help()
        return 2


def _handle_basic_analyze(args) -> int:
    """Handle basic analysis command."""
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


def _handle_rag_analyze(args) -> int:
    """Handle RAG-augmented analysis command."""
    report_path = Path(args.input)
    if not report_path.exists():
        print(f"error: input file not found: {report_path}")
        return 1

    try:
        content = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: reading report: {exc}")
        return 1

    # Load documents
    requirements_docs = []
    if args.requirements:
        requirements_docs.extend(_load_docs_from_paths(args.requirements))
    if args.requirements_text:
        requirements_docs.append(("inline:requirements", args.requirements_text))

    knowledge_docs = []
    if args.knowledge:
        knowledge_docs.extend(_load_docs_from_paths(args.knowledge))

    # Run RAG analysis
    try:
        result = rag_analyze(
            test_report_content=content,
            repo_path=args.repo,
            requirements_docs=requirements_docs if requirements_docs else None,
            query=args.query,
        )
    except Exception as exc:
        print(f"error: RAG analysis failed: {exc}")
        return 1

    if args.prompt_only:
        print(result.augmented_prompt)
        return 0

    indent = 2 if args.format == "pretty" else None
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=indent))
    return 0


def _handle_ingest(args) -> int:
    """Handle document ingestion command."""
    analyzer = RAGAnalyzer()

    requirements_docs = []
    if args.requirements:
        requirements_docs.extend(_load_docs_from_paths(args.requirements))
    if args.requirements_text:
        requirements_docs.append(("inline:requirements", args.requirements_text))

    knowledge_docs = []
    if args.knowledge:
        knowledge_docs.extend(_load_docs_from_paths(args.knowledge))

    try:
        total = analyzer.initialize_corpus(
            repo_path=args.repo,
            requirements_docs=requirements_docs if requirements_docs else None,
            knowledge_docs=knowledge_docs if knowledge_docs else None,
        )
        print(f"Indexed {total} chunks")
    except Exception as exc:
        print(f"error: ingestion failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
