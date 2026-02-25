"""Code-aware chunking for improved repository ingestion.

This module provides code-structure-aware text chunking that respects
function, class, and module boundaries when processing code repositories.
It integrates with the existing retrieval pipeline to provide better
code context for test analysis.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence


class CodeLanguage(str, Enum):
    """Supported programming languages for code-aware chunking."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


@dataclass
class CodeUnit:
    """A unit of code with structure awareness."""
    unit_type: str  # module, class, function, method, constant
    name: str
    content: str
    start_line: int
    end_line: int
    language: CodeLanguage
    parent: Optional[str] = None  # Parent unit name (e.g., class for method)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeChunk:
    """A chunk of code with code-specific metadata."""
    chunk_id: str
    source_id: str
    code_unit: Optional[CodeUnit]  # The source code unit
    text: str
    token_count: int
    start_line: int
    end_line: int
    language: CodeLanguage
    chunk_type: str  # full_unit, partial_unit, standalone
    metadata: Dict[str, Any] = field(default_factory=dict)


# Pattern definitions for Python code structure
_PYTHON_FUNCTION_PATTERN = re.compile(
    r'^(?:async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
    re.MULTILINE
)
_PYTHON_CLASS_PATTERN = re.compile(
    r'^class\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\([^)]*\))?:',
    re.MULTILINE
)
_PYTHON_IMPORT_PATTERN = re.compile(
    r'^(?:from\s+[\w.]+\s+)?import\s+([\w.,\s]+)',
    re.MULTILINE
)
_PYTHON_CONSTANT_PATTERN = re.compile(
    r'^([A-Z][A-Z0-9_]*)\s*=',
    re.MULTILINE
)

# Pattern definitions for JavaScript/TypeScript
_JS_FUNCTION_PATTERN = re.compile(
    r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)|'
    r'(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s+)?\(|'
    r'(?:async\s+)?\(([^)]*)\)\s*=>)',
    re.MULTILINE
)
_JS_CLASS_PATTERN = re.compile(
    r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+\w+)?\s*\{',
    re.MULTILINE
)


def detect_language(source_id: str, content: str) -> CodeLanguage:
    """Detect programming language from file extension or content.

    Args:
        source_id: Source identifier (file path)
        content: Code content

    Returns:
        Detected CodeLanguage
    """
    # Check file extension
    if isinstance(source_id, str):
        ext = source_id.lower().split('.')[-1] if '.' in source_id else ''
        lang_map = {
            'py': CodeLanguage.PYTHON,
            'js': CodeLanguage.JAVASCRIPT,
            'ts': CodeLanguage.TYPESCRIPT,
            'jsx': CodeLanguage.JAVASCRIPT,
            'tsx': CodeLanguage.TYPESCRIPT,
            'java': CodeLanguage.JAVA,
            'go': CodeLanguage.GO,
            'rs': CodeLanguage.RUST,
        }
        if ext in lang_map:
            return lang_map[ext]

    # Fallback: content-based detection
    if 'def ' in content and ':' in content:
        return CodeLanguage.PYTHON
    if 'function ' in content or '=>' in content:
        return CodeLanguage.JAVASCRIPT
    if 'interface ' in content and '{' in content:
        return CodeLanguage.TYPESCRIPT

    return CodeLanguage.UNKNOWN


def extract_python_units(content: str) -> List[CodeUnit]:
    """Extract Python code units (classes, functions, imports).

    Args:
        content: Python source code

    Returns:
        List of CodeUnit objects
    """
    lines = content.splitlines()
    units: List[CodeUnit] = []

    # Track indentation to determine scope
    current_class: Optional[str] = None

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # Check for class definition
        class_match = _PYTHON_CLASS_PATTERN.match(stripped)
        if class_match:
            current_class = class_match.group(1)
            units.append(CodeUnit(
                unit_type='class',
                name=current_class,
                content=stripped,
                start_line=i + 1,
                end_line=i + 1,
                language=CodeLanguage.PYTHON,
                metadata={'is_public': not stripped.startswith('_')}
            ))
            continue

        # Check for function/method definition
        func_match = _PYTHON_FUNCTION_PATTERN.match(stripped)
        if func_match:
            func_name = func_match.group(1)
            # Determine if it's a method (inside a class)
            unit_type = 'method' if current_class else 'function'
            units.append(CodeUnit(
                unit_type=unit_type,
                name=func_name,
                content=stripped,
                start_line=i + 1,
                end_line=i + 1,
                language=CodeLanguage.PYTHON,
                parent=current_class,
                metadata={
                    'is_public': not func_name.startswith('_'),
                    'is_async': 'async' in stripped
                }
            ))
            continue

        # Check for imports
        import_match = _PYTHON_IMPORT_PATTERN.match(stripped)
        if import_match and not stripped.startswith('#'):
            units.append(CodeUnit(
                unit_type='import',
                name=import_match.group(1).strip(),
                content=stripped,
                start_line=i + 1,
                end_line=i + 1,
                language=CodeLanguage.PYTHON,
            ))
            continue

        # Check for constants
        const_match = _PYTHON_CONSTANT_PATTERN.match(stripped)
        if const_match:
            units.append(CodeUnit(
                unit_type='constant',
                name=const_match.group(1),
                content=stripped,
                start_line=i + 1,
                end_line=i + 1,
                language=CodeLanguage.PYTHON,
            ))

    return units


def extract_javascript_units(content: str) -> List[CodeUnit]:
    """Extract JavaScript/TypeScript code units.

    Args:
        content: JavaScript/TypeScript source code

    Returns:
        List of CodeUnit objects
    """
    lines = content.splitlines()
    units: List[CodeUnit] = []
    current_class: Optional[str] = None

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # Check for class
        class_match = _JS_CLASS_PATTERN.match(stripped)
        if class_match:
            current_class = class_match.group(1)
            units.append(CodeUnit(
                unit_type='class',
                name=current_class,
                content=stripped,
                start_line=i + 1,
                end_line=i + 1,
                language=CodeLanguage.JAVASCRIPT,
            ))
            continue

        # Check for function
        func_match = _JS_FUNCTION_PATTERN.search(stripped)
        if func_match:
            func_name = func_match.group(1) or func_match.group(2) or 'anonymous'
            unit_type = 'method' if current_class else 'function'
            units.append(CodeUnit(
                unit_type=unit_type,
                name=func_name,
                content=stripped,
                start_line=i + 1,
                end_line=i + 1,
                language=CodeLanguage.JAVASCRIPT,
                parent=current_class,
            ))

    return units


def extract_code_units(content: str, language: CodeLanguage) -> List[CodeUnit]:
    """Extract code units based on detected language.

    Args:
        content: Source code content
        language: Programming language

    Returns:
        List of CodeUnit objects
    """
    if language == CodeLanguage.PYTHON:
        return extract_python_units(content)
    elif language in (CodeLanguage.JAVASCRIPT, CodeLanguage.TYPESCRIPT):
        return extract_javascript_units(content)
    else:
        # Return entire content as single module unit
        return [CodeUnit(
            unit_type='module',
            name='main',
            content=content,
            start_line=1,
            end_line=len(content.splitlines()),
            language=language,
        )]


def chunk_code_by_structure(
    content: str,
    source_id: str,
    language: CodeLanguage,
    max_chunk_tokens: int = 360,
    overlap_tokens: int = 40,
) -> List[CodeChunk]:
    """Chunk code respecting structure boundaries.

    This function creates chunks that respect code structure (functions, classes)
    while still allowing chunks to be split if they exceed the token limit.

    Args:
        content: Source code content
        source_id: Source identifier
        language: Programming language
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap for partial chunks

    Returns:
        List of CodeChunk objects
    """
    units = extract_code_units(content, language)
    if not units:
        # Fallback: treat as plain text
        return _chunk_plain_text(content, source_id, language, max_chunk_tokens, overlap_tokens)

    chunks: List[CodeChunk] = []
    lines = content.splitlines()

    # Group units by their line positions
    # For each unit, get its content from the full source
    for unit in units:
        # Get unit content from full source (lines from start_line-1 to end_line-1)
        start_idx = max(0, unit.start_line - 1)
        end_idx = min(len(lines), unit.end_line)
        unit_content = '\n'.join(lines[start_idx:end_idx])

        tokens = _tokenize_code(unit_content)

        if len(tokens) <= max_chunk_tokens:
            # Unit fits in single chunk
            chunks.append(CodeChunk(
                chunk_id=_make_chunk_id(source_id, unit.start_line, unit.name),
                source_id=source_id,
                code_unit=unit,
                text=unit_content,
                token_count=len(tokens),
                start_line=unit.start_line,
                end_line=unit.end_line,
                language=language,
                chunk_type='full_unit',
                metadata={
                    'unit_name': unit.name,
                    'unit_type': unit.unit_type,
                    'parent': unit.parent,
                }
            ))
        else:
            # Unit too large, chunk it further using text-based splitting
            sub_chunks = _chunk_plain_text(
                unit_content,
                source_id,
                language,
                max_chunk_tokens,
                overlap_tokens,
                offset_line=unit.start_line
            )
            for sc in sub_chunks:
                sc.chunk_type = 'partial_unit'
                sc.metadata.update({
                    'unit_name': unit.name,
                    'unit_type': unit.unit_type,
                    'parent': unit.parent,
                })
            chunks.extend(sub_chunks)

    # Add standalone code sections (between defined units)
    if len(units) > 1:
        for i in range(len(units) - 1):
            current = units[i]
            next_unit = units[i + 1]
            if current.end_line < next_unit.start_line - 1:
                # There's code between these units
                between_lines = lines[current.end_line:next_unit.start_line - 1]
                between_content = '\n'.join(between_lines).strip()
                if between_content:
                    tokens = _tokenize_code(between_content)
                    if tokens:
                        chunks.append(CodeChunk(
                            chunk_id=_make_chunk_id(source_id, current.end_line + 1, 'standalone'),
                            source_id=source_id,
                            code_unit=None,
                            text=between_content,
                            token_count=len(tokens),
                            start_line=current.end_line + 1,
                            end_line=next_unit.start_line - 1,
                            language=language,
                            chunk_type='standalone',
                            metadata={}
                        ))

    return chunks


def _chunk_plain_text(
    content: str,
    source_id: str,
    language: CodeLanguage,
    max_chunk_tokens: int,
    overlap_tokens: int,
    offset_line: int = 1,
) -> List[CodeChunk]:
    """Fallback chunking using plain text approach."""
    chunks: List[CodeChunk] = []
    tokens = _tokenize_code(content)

    if not tokens:
        return chunks

    if len(tokens) <= max_chunk_tokens:
        return [CodeChunk(
            chunk_id=_make_chunk_id(source_id, offset_line, 'full'),
            source_id=source_id,
            code_unit=None,
            text=content,
            token_count=len(tokens),
            start_line=offset_line,
            end_line=offset_line + len(content.splitlines()) - 1,
            language=language,
            chunk_type='standalone',
            metadata={}
        )]

    # Split into smaller chunks
    lines = content.splitlines()
    current_lines: List[str] = []
    current_tokens: List[str] = []
    start_line = offset_line

    for i, line in enumerate(lines):
        line_tokens = _tokenize_code(line)
        if len(current_tokens) + len(line_tokens) > max_chunk_tokens and current_lines:
            # Emit current chunk
            chunks.append(CodeChunk(
                chunk_id=_make_chunk_id(source_id, start_line, f'chunk{len(chunks)}'),
                source_id=source_id,
                code_unit=None,
                text='\n'.join(current_lines),
                token_count=len(current_tokens),
                start_line=start_line,
                end_line=start_line + len(current_lines) - 1,
                language=language,
                chunk_type='standalone',
                metadata={}
            ))

            # Start new chunk with overlap
            overlap_lines = current_lines[-overlap_tokens:] if len(current_lines) > overlap_tokens else current_lines
            current_lines = overlap_lines + [line]
            current_tokens = _tokenize_code('\n'.join(current_lines))
            start_line = start_line + len(current_lines) - len(overlap_lines) - 1
        else:
            current_lines.append(line)
            current_tokens.extend(line_tokens)

    # Emit final chunk
    if current_lines:
        chunks.append(CodeChunk(
            chunk_id=_make_chunk_id(source_id, start_line, f'chunk{len(chunks)}'),
            source_id=source_id,
            code_unit=None,
            text='\n'.join(current_lines),
            token_count=len(current_tokens),
            start_line=start_line,
            end_line=offset_line + len(lines) - 1,
            language=language,
            chunk_type='standalone',
            metadata={}
        ))

    return chunks


def _tokenize_code(text: str) -> List[str]:
    """Tokenize code text, handling identifiers specially.

    This tokenizer preserves code identifiers as single tokens,
    which helps with semantic matching of function/class names.

    Args:
        text: Code text to tokenize

    Returns:
        List of tokens
    """
    # Split on whitespace and operators, but keep identifiers intact
    # Pattern matches: identifiers, numbers, operators, punctuation
    pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+(?:\.[0-9]+)?|[^\s\w]')
    return pattern.findall(text)


def _make_chunk_id(source_id: str, line: int, name: str) -> str:
    """Create a unique chunk ID."""
    import hashlib
    h = hashlib.sha1()
    h.update(f"{source_id}:{line}:{name}".encode())
    return f"{source_id.split('/')[-1].split('.')[0]}_{line}_{name}"[:40]


class CodeAwareChunker:
    """Code-structure-aware chunker that integrates with retrieval pipeline.

    This chunker extracts code structure (functions, classes) and creates
    chunks that respect those boundaries while still allowing splitting
    for large units.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 360,
        overlap_tokens: int = 40,
    ) -> None:
        self._max_tokens = max_chunk_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(
        self,
        content: str,
        source_id: str,
    ) -> List[CodeChunk]:
        """Chunk code with structure awareness.

        Args:
            content: Source code content
            source_id: Source identifier (file path)

        Returns:
            List of CodeChunk objects
        """
        language = detect_language(source_id, content)
        return chunk_code_by_structure(
            content=content,
            source_id=source_id,
            language=language,
            max_chunk_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
        )

    def detect_language(self, source_id: str, content: str) -> CodeLanguage:
        """Detect the programming language."""
        return detect_language(source_id, content)
