"""Smart content type detection for improved multimodal ingestion routing.

This module provides content-aware detection that goes beyond file extensions
to analyze actual content and determine appropriate processing strategies.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class ContentCategory(str, Enum):
    """High-level content categories."""

    CODE = "code"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    STRUCTURED_DATA = "structured_data"  # JSON, YAML, TOML
    DOCUMENTATION = "documentation"
    REQUIREMENTS = "requirements"
    SYSTEM_ANALYSIS = "system_analysis"
    UNKNOWN = "unknown"


class CodeLanguage(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    SQL = "sql"
    BASH = "bash"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class DocumentType(str, Enum):
    """Document types for non-code content."""

    REQUIREMENTS = "requirements"
    SPECIFICATION = "specification"
    ARCHITECTURE = "architecture"
    API_DOC = "api_doc"
    CHANGELOG = "changelog"
    README = "readme"
    INCIDENT_REPORT = "incident_report"
    TEST_PLAN = "test_plan"
    USER_GUIDE = "user_guide"
    UNKNOWN = "unknown"


@dataclass
class ContentDetectionResult:
    """Result of content type detection."""

    category: ContentCategory
    confidence: float
    detected_language: Optional[CodeLanguage] = None
    detected_doc_type: Optional[DocumentType] = None
    indicators: Dict[str, float] = field(default_factory=dict)
    recommended_chunker: str = "basic"  # basic, code_aware
    recommended_source_type: str = "knowledge"


# Language detection patterns
LANGUAGE_PATTERNS: Tuple[Tuple[CodeLanguage, List[Tuple[str, float]]], ...] = (
    (CodeLanguage.PYTHON, [
        (r'^import\s+\w+', 0.8),
        (r'^from\s+\w+\s+import', 0.8),
        (r'def\s+\w+\s*\(', 0.9),
        (r'class\s+\w+.*:', 0.7),
        (r'if\s+__name__\s*==', 0.9),
        (r'@pytest\.', 0.8),
        (r'self\.\w+', 0.6),
        (r'TypeAlias', 0.8),
        (r'->\s*\w+:', 0.7),
    ]),
    (CodeLanguage.JAVASCRIPT, [
        (r'^const\s+\w+\s*=', 0.8),
        (r'^let\s+\w+\s*=', 0.7),
        (r'^function\s+\w+\s*\(', 0.8),
        (r'=>\s*{', 0.7),
        (r'console\.log', 0.8),
        (r'require\s*\(', 0.7),
        (r'module\.exports', 0.9),
        (r'\.then\s*\(', 0.6),
        (r'async\s+function', 0.7),
    ]),
    (CodeLanguage.TYPESCRIPT, [
        (r':\s*(string|number|boolean|any|void|never)\b', 0.9),
        (r'interface\s+\w+\s*{', 0.9),
        (r'type\s+\w+\s*=', 0.8),
        (r'<\w+>', 0.7),  # Generics
        (r'as\s+(string|number|boolean|any)', 0.8),
        (r':\s*\w+\[\]', 0.7),  # Array types
        (r'enum\s+\w+', 0.8),
    ]),
    (CodeLanguage.JAVA, [
        (r'^package\s+[\w.]+;', 0.9),
        (r'^import\s+java\.', 0.9),
        (r'public\s+class\s+\w+', 0.9),
        (r'public\s+static\s+void\s+main', 0.9),
        (r'System\.out\.print', 0.8),
        (r'@Override', 0.9),
        (r'@Test', 0.8),
    ]),
    (CodeLanguage.GO, [
        (r'^package\s+\w+', 0.9),
        (r'^import\s+\(', 0.8),
        (r'func\s+\w+\s*\(', 0.9),
        (r'func\s+\(\w+\s+\*?\w+\)', 0.8),
        (r':=', 0.8),
        (r'go\s+func', 0.7),
        (r'defer\s+', 0.8),
    ]),
    (CodeLanguage.RUST, [
        (r'^use\s+\w+::', 0.8),
        (r'^fn\s+\w+\s*\(', 0.9),
        (r'impl\s+\w+', 0.8),
        (r'let\s+mut\s+', 0.8),
        (r'->\s*\w+', 0.7),
        (r'println!', 0.8),
        (r'#\[test\]', 0.8),
        (r'pub\s+fn', 0.7),
    ]),
    (CodeLanguage.C, [
        (r'#include\s*<', 0.9),
        (r'^int\s+main\s*\(', 0.9),
        (r'printf\s*\(', 0.8),
        (r'malloc\s*\(', 0.8),
        (r'struct\s+\w+\s*{', 0.8),
    ]),
    (CodeLanguage.CPP, [
        (r'#include\s*<iostream>', 0.9),
        (r'std::', 0.9),
        (r'cout\s*<<', 0.8),
        (r'class\s+\w+\s*{', 0.7),
        (r'template\s*<', 0.8),
        (r'namespace\s+\w+', 0.7),
    ]),
    (CodeLanguage.CSHARP, [
        (r'^using\s+System', 0.8),
        (r'namespace\s+\w+', 0.8),
        (r'public\s+class\s+\w+', 0.9),
        (r'Console\.Write', 0.8),
        (r'\[Test\]', 0.8),
    ]),
    (CodeLanguage.RUBY, [
        (r'^require\s+[\'"]', 0.8),
        (r'^def\s+\w+', 0.9),
        (r'^class\s+\w+', 0.8),
        (r'puts\s+', 0.8),
        (r'end\s*$', 0.7),
    ]),
    (CodeLanguage.PHP, [
        (r'<\?php', 0.9),
        (r'\$\w+\s*=', 0.8),
        (r'function\s+\w+\s*\(', 0.7),
        (r'echo\s+', 0.8),
    ]),
    (CodeLanguage.SWIFT, [
        (r'^import\s+(Foundation|UIKit|SwiftUI)', 0.9),
        (r'func\s+\w+\s*\(', 0.8),
        (r'var\s+\w+:\s*\w+', 0.7),
        (r'let\s+\w+:\s*\w+', 0.7),
        (r'class\s+\w+:', 0.8),
    ]),
    (CodeLanguage.KOTLIN, [
        (r'^package\s+[\w.]+', 0.9),
        (r'fun\s+\w+\s*\(', 0.9),
        (r'val\s+\w+\s*[:=]', 0.8),
        (r'var\s+\w+\s*[:=]', 0.7),
        (r'class\s+\w+\(', 0.8),
    ]),
    (CodeLanguage.SCALA, [
        (r'^package\s+[\w.]+', 0.9),
        (r'def\s+\w+\s*\(', 0.8),
        (r'val\s+\w+\s*[:=]', 0.7),
        (r'class\s+\w+', 0.7),
        (r'object\s+\w+', 0.8),
    ]),
    (CodeLanguage.SQL, [
        (r'^SELECT\s+', 0.9),
        (r'^INSERT\s+INTO', 0.9),
        (r'^UPDATE\s+\w+', 0.9),
        (r'^CREATE\s+(TABLE|INDEX|VIEW)', 0.9),
        (r'FROM\s+\w+\s+WHERE', 0.8),
    ]),
    (CodeLanguage.BASH, [
        (r'^#!/bin/(ba)?sh', 0.9),
        (r'\$\{?\w+\}?', 0.7),
        (r'^if\s+\[', 0.8),
        (r'echo\s+', 0.7),
        (r'\|\s*grep', 0.7),
    ]),
    (CodeLanguage.YAML, [
        (r'^\w+:\s*$', 0.7),
        (r'^\s+-\s+\w+:', 0.8),
        (r'^\s+\w+:\s+\|', 0.8),
    ]),
    (CodeLanguage.JSON, [
        (r'^\s*{', 0.8),
        (r'"\w+":\s*["{\[\d]', 0.9),
    ]),
)

# Document type patterns
DOCUMENT_PATTERNS: Tuple[Tuple[DocumentType, List[Tuple[str, float]]], ...] = (
    (DocumentType.REQUIREMENTS, [
        (r'shall\s+', 0.8),
        (r'should\s+', 0.7),
        (r'must\s+', 0.8),
        (r'requirement', 0.9),
        (r'user\s+story', 0.8),
        (r'acceptance\s+criteria', 0.9),
        (r'feature\s+description', 0.7),
        (r'functional\s+requirement', 0.9),
        (r'non-functional', 0.8),
    ]),
    (DocumentType.SPECIFICATION, [
        (r'specification', 0.9),
        (r'api\s+spec', 0.8),
        (r'interface\s+definition', 0.7),
        (r'data\s+model', 0.7),
        (r'schema', 0.6),
    ]),
    (DocumentType.ARCHITECTURE, [
        (r'architecture', 0.9),
        (r'system\s+design', 0.8),
        (r'microservice', 0.8),
        (r'component\s+diagram', 0.7),
        (r'high-level', 0.7),
    ]),
    (DocumentType.API_DOC, [
        (r'endpoint', 0.8),
        (r'http\s+(get|post|put|delete|patch)', 0.9),
        (r'request\s+body', 0.8),
        (r'response\s+code', 0.8),
        (r'api', 0.7),
        (r'openapi', 0.9),
        (r'swagger', 0.9),
    ]),
    (DocumentType.CHANGELOG, [
        (r'^#\s*changelog', 0.9),
        (r'^##\s*\d+\.\d+', 0.8),
        (r'\[breaking\]', 0.8),
        (r'\[feature\]', 0.8),
        (r'\[fix\]', 0.8),
    ]),
    (DocumentType.README, [
        (r'^#\s*\w+', 0.7),
        (r'^##\s*install', 0.8),
        (r'^##\s*usage', 0.8),
        (r'^##\s*contribut', 0.8),
    ]),
    (DocumentType.INCIDENT_REPORT, [
        (r'incident', 0.9),
        (r'root\s+cause', 0.9),
        (r'timeline', 0.8),
        (r'impact', 0.7),
        (r'mitigation', 0.8),
        (r'resolution', 0.7),
    ]),
    (DocumentType.TEST_PLAN, [
        (r'test\s+plan', 0.9),
        (r'test\s+case', 0.8),
        (r'test\s+scenario', 0.8),
        (r'testing\s+strategy', 0.8),
        (r'coverage', 0.6),
    ]),
    (DocumentType.USER_GUIDE, [
        (r'user\s+guide', 0.9),
        (r'tutorial', 0.8),
        (r'how\s+to', 0.8),
        (r'getting\s+started', 0.8),
        (r'step\s+by\s+step', 0.7),
    ]),
)

# Code indicators
CODE_INDICATORS = [
    (r'^#!/', 0.9),  # Shebang - strong indicator of code
    (r'^SELECT\s+', 0.8),  # SQL keyword
    (r'^INSERT\s+', 0.8),  # SQL keyword
    (r'^UPDATE\s+', 0.8),  # SQL keyword
    (r'^DELETE\s+', 0.8),  # SQL keyword
    (r'^CREATE\s+', 0.8),  # SQL keyword
    (r'^DROP\s+', 0.8),  # SQL keyword
    (r'^ALTER\s+', 0.8),  # SQL keyword
    (r'^import\s+', 0.6),
    (r'^from\s+\w+\s+import', 0.7),
    (r'^export\s+', 0.7),
    (r'^require\s*\(', 0.6),
    (r'def\s+\w+\s*\(', 0.9),
    (r'function\s+\w+\s*\(', 0.8),
    (r'fn\s+\w+\s*\(', 0.9),
    (r'class\s+\w+', 0.7),
    (r'interface\s+\w+', 0.7),
    (r'impl\s+\w+', 0.8),
    (r'public\s+', 0.5),
    (r'private\s+', 0.5),
    (r'const\s+', 0.6),
    (r'var\s+', 0.5),
    (r'let\s+', 0.5),
    (r'if\s+\(.+\)\s*{', 0.6),
    (r'for\s*\(.+\)\s*{', 0.6),
    (r'while\s*\(.+\)\s*{', 0.6),
    (r'return\s+', 0.5),
    (r'throw\s+', 0.6),
    (r'try\s*{', 0.6),
    (r'catch\s*\(', 0.6),
]

# Markdown indicators
MARKDOWN_INDICATORS = [
    (r'^#+\s+', 0.9),
    (r'\[.+\]\(.+\)', 0.8),  # Links
    (r'\*\*[^*]+\*\*', 0.7),  # Bold
    (r'\*[^*]+\*', 0.7),  # Italic
    (r'^```', 0.9),  # Code blocks
    (r'^-\s+', 0.7),  # Lists
    (r'^\d+\.\s+', 0.7),  # Numbered lists
    (r'\|.+\|.+\|', 0.8),  # Tables
]

# Documentation vs plain text indicators
DOCUMENTATION_INDICATORS = [
    (r'\b\w{50,}\b', 0.3),  # Long words suggest prose
    (r'\b(the|a|an|and|or|but|however|therefore)\b', 0.4),  # Prose words
]


def detect_content_type(
    content: str,
    source_hint: Optional[str] = None,
) -> ContentDetectionResult:
    """Detect content type from raw content.

    This goes beyond file extension detection to analyze actual content
    and determine appropriate processing strategies.

    Args:
        content: Raw content to analyze
        source_hint: Optional hint about the source (file path, source_id)

    Returns:
        ContentDetectionResult with detected type and confidence
    """
    if not content or not content.strip():
        return ContentDetectionResult(
            category=ContentCategory.UNKNOWN,
            confidence=0.0,
            recommended_chunker="basic",
            recommended_source_type="knowledge",
        )

    lines = content.split('\n')
    first_lines = '\n'.join(lines[:10])
    first_line = lines[0].strip() if lines else ""
    last_lines = '\n'.join(lines[-5:]) if len(lines) > 5 else first_lines

    # Check for code
    code_indicators: Dict[str, float] = {}
    for pattern, weight in CODE_INDICATORS:
        # Handle leading whitespace for line-starting patterns
        adjusted_pattern = pattern
        if pattern.startswith('^'):
            # Handle ^# patterns specially to avoid double ^
            if pattern.startswith('^#'):
                adjusted_pattern = r'^\s*' + pattern[1:]
            else:
                adjusted_pattern = r'^\s*' + pattern[1:]
        matches = len(re.findall(adjusted_pattern, content, re.MULTILINE | re.IGNORECASE))
        if matches > 0:
            code_indicators[pattern] = min(1.0, matches * weight / 3)

    code_score = sum(code_indicators.values())

    # Check for specific language
    language_scores: Dict[CodeLanguage, float] = {}
    for language, patterns in LANGUAGE_PATTERNS:
        lang_score = 0.0
        for pattern, weight in patterns:
            # Use MULTILINE to match at start of each line, and handle leading whitespace
            adjusted_pattern = pattern
            if pattern.startswith('^'):
                # Handle ^# patterns specially to avoid double ^
                if pattern.startswith('^#'):
                    adjusted_pattern = r'^\s*' + pattern[1:]
                else:
                    adjusted_pattern = r'^\s*' + pattern[1:]
            matches = len(re.findall(adjusted_pattern, content, re.MULTILINE | re.IGNORECASE))
            if matches > 0:
                lang_score += weight * min(1.0, matches / 2)
        if lang_score > 0:
            language_scores[language] = lang_score

    # Check for markdown
    md_indicators: Dict[str, float] = {}
    for pattern, weight in MARKDOWN_INDICATORS:
        matches = len(re.findall(pattern, content, re.MULTILINE))
        if matches > 0:
            md_indicators[pattern] = min(1.0, matches * weight / 3)

    md_score = sum(md_indicators.values())

    # Check for structured data (JSON/YAML)
    json_score = 0.0
    if re.match(r'^\s*[{[]', content):
        json_score = 0.9

    yaml_score = 0.0
    if re.match(r'^\w+:\s*$', content, re.MULTILINE):
        yaml_score = 0.7

    # Check for document type
    doc_scores: Dict[DocumentType, float] = {}
    for doc_type, patterns in DOCUMENT_PATTERNS:
        doc_score = 0.0
        for pattern, weight in patterns:
            matches = len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
            if matches > 0:
                doc_score += weight * min(1.0, matches / 2)
        if doc_score > 0:
            doc_scores[doc_type] = doc_score

    # Determine category
    category = ContentCategory.UNKNOWN
    detected_language = None
    detected_doc_type = None
    confidence = 0.5

    if code_score > 0.25 and code_score > md_score:
        category = ContentCategory.CODE
        if language_scores:
            detected_language = max(language_scores, key=language_scores.get)
            confidence = min(1.0, language_scores[detected_language] / 5)
        else:
            confidence = min(1.0, code_score / 3)
    elif md_score > 1.0:
        category = ContentCategory.MARKDOWN
        confidence = min(1.0, md_score / 3)
        # Check if it's a specific doc type
        if doc_scores:
            detected_doc_type = max(doc_scores, key=doc_scores.get)
            confidence = min(1.0, (md_score + doc_scores[detected_doc_type]) / 4)
    elif json_score > 0.5:
        category = ContentCategory.STRUCTURED_DATA
        detected_language = CodeLanguage.JSON
        confidence = json_score
    elif yaml_score > 0.5:
        category = ContentCategory.STRUCTURED_DATA
        detected_language = CodeLanguage.YAML
        confidence = yaml_score
    elif doc_scores:
        category = ContentCategory.DOCUMENTATION
        detected_doc_type = max(doc_scores, key=doc_scores.get)
        confidence = min(1.0, doc_scores[detected_doc_type] / 3)
    else:
        # Check line length to distinguish prose from short text
        avg_line_length = sum(len(l) for l in lines) / max(len(lines), 1)
        if avg_line_length > 50:
            category = ContentCategory.PLAIN_TEXT
            confidence = 0.6
        else:
            category = ContentCategory.UNKNOWN
            confidence = 0.3

    # Determine recommended chunker and source type based on category
    recommended_chunker = "basic"
    recommended_source_type = "knowledge"

    if category == ContentCategory.CODE:
        recommended_chunker = "code_aware"
        if source_hint and any(ext in source_hint.lower() for ext in ['test_', 'tests/', '_test.py', '.test.']):
            recommended_source_type = "code_snippet"
        else:
            recommended_source_type = "repository"
    elif detected_doc_type == DocumentType.REQUIREMENTS:
        recommended_source_type = "requirements"
    elif detected_doc_type in (DocumentType.ARCHITECTURE, DocumentType.SPECIFICATION, DocumentType.INCIDENT_REPORT):
        recommended_source_type = "system_analysis"
    elif category == ContentCategory.MARKDOWN:
        if detected_doc_type == DocumentType.README:
            recommended_source_type = "repository"
        elif detected_doc_type:
            recommended_source_type = "knowledge"

    return ContentDetectionResult(
        category=category,
        confidence=round(confidence, 3),
        detected_language=detected_language,
        detected_doc_type=detected_doc_type,
        indicators={
            "code_score": round(code_score, 3),
            "md_score": round(md_score, 3),
            "json_score": round(json_score, 3),
            "yaml_score": round(yaml_score, 3),
        },
        recommended_chunker=recommended_chunker,
        recommended_source_type=recommended_source_type,
    )


def detect_code_language(content: str) -> Optional[CodeLanguage]:
    """Detect programming content.

    Args:
        content: Code content to analyze

    Returns:
        Detected language language from code or None
    """
    if not content:
        return None

    result = detect_content_type(content)
    return result.detected_language


def suggest_ingestion_strategy(
    content: str,
    source_hint: Optional[str] = None,
) -> Tuple[str, str]:
    """Suggest ingestion strategy based on content analysis.

    Args:
        content: Content to analyze
        source_hint: Optional source hint (file path)

    Returns:
        Tuple of (chunker_type, source_type) recommendations
    """
    result = detect_content_type(content, source_hint)
    return result.recommended_chunker, result.recommended_source_type
