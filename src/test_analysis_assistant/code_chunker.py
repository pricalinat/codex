"""Semantic chunking implementation for improved retrieval.

This module provides semantic chunking that creates more coherent chunks
by respecting sentence boundaries and topic shifts. It complements the
existing code-aware and test-aware chunkers.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence


# Sentence boundary patterns for different content types
_SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')
_PARAGRAPH_BREAKS = re.compile(r'\n\s*\n')
_CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```|`[^`]+`')
_MARKDOWN_HEADERS = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)
_LIST_MARKERS = re.compile(r'^[\s]*[-*+]\s+|^[\s]*\d+\.\s+', re.MULTILINE)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while handling common edge cases.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Handle code blocks first - don't split inside them
    code_blocks: List[str] = []
    def replace_code(match: str) -> str:
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{idx}__"

    text_without_code = _CODE_BLOCK_PATTERN.sub(replace_code, text)

    # Split on sentence endings
    sentences = _SENTENCE_ENDINGS.split(text_without_code)

    # Restore code blocks
    result: List[str] = []
    for sent in sentences:
        for idx, block in enumerate(code_blocks):
            sent = sent.replace(f"__CODE_BLOCK_{idx}__", block)
        if sent.strip():
            result.append(sent)

    return result


def detect_paragraph_boundaries(text: str) -> List[int]:
    """Detect paragraph boundary positions.

    Args:
        text: Input text

    Returns:
        List of line indices where paragraphs start
    """
    lines = text.split('\n')
    boundaries = [0]  # Start with first line

    for i, line in enumerate(lines):
        stripped = line.strip()
        # New paragraph indicators
        if not stripped:
            # Empty line - next non-empty line starts new paragraph
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    boundaries.append(j)
                    break
        elif _MARKDOWN_HEADERS.match(stripped):
            # Header starts new section
            boundaries.append(i)
        elif _LIST_MARKERS.match(stripped) and i > 0:
            # List continuation or new list
            prev_stripped = lines[i - 1].strip()
            if not prev_stripped or not _LIST_MARKERS.match(prev_stripped):
                boundaries.append(i)

    return sorted(set(boundaries))


def group_into_chunks(
    sentences: List[str],
    max_tokens: int,
    overlap_sentences: int = 1,
) -> List[str]:
    """Group sentences into chunks respecting token limits.

    Args:
        sentences: List of sentences to group
        max_tokens: Maximum tokens per chunk
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of text chunks
    """
    if not sentences:
        return []

    # Estimate tokens (simple word-based estimate)
    def estimate_tokens(text: str) -> int:
        return len(text.split())

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        sent_tokens = estimate_tokens(sentence)

        # Check if adding this sentence would exceed limit
        if current_tokens + sent_tokens > max_tokens and current_chunk:
            # Emit current chunk
            chunks.append(' '.join(current_chunk))

            # Start new chunk with overlap
            overlap_start = max(0, len(current_chunk) - overlap_sentences)
            current_chunk = current_chunk[overlap_start:]
            current_tokens = sum(estimate_tokens(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_tokens += sent_tokens

    # Emit final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def semantic_chunk_text(
    content: str,
    max_chunk_tokens: int = 360,
    overlap_sentences: int = 1,
    respect_paragraphs: bool = True,
) -> List[str]:
    """Create semantically coherent chunks from text.

    This function creates chunks that:
    1. Respect sentence boundaries
    2. Group related sentences together
    3. Respect paragraph breaks when possible
    4. Allow configurable overlap for context preservation

    Args:
        content: Input text content
        max_chunk_tokens: Maximum tokens per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        respect_paragraphs: Whether to prioritize paragraph boundaries

    Returns:
        List of semantically coherent text chunks
    """
    if not content or not content.strip():
        return []

    # For code-heavy content, fall back to code-aware chunking
    code_indicators = content.count('def ') + content.count('class ') + content.count('function ')
    if code_indicators > 5:
        # Likely code - use line-based chunking
        return _chunk_by_lines(content, max_chunk_tokens)

    if respect_paragraphs:
        # Split by paragraphs first
        paragraphs = _PARAGRAPH_BREAKS.split(content)
        sentences: List[str] = []

        for para in paragraphs:
            if para.strip():
                para_sentences = split_into_sentences(para)
                sentences.extend(para_sentences)
    else:
        sentences = split_into_sentences(content)

    # Remove very short sentences that are likely noise
    sentences = [s for s in sentences if len(s.split()) >= 3]

    return group_into_chunks(sentences, max_chunk_tokens, overlap_sentences)


def _chunk_by_lines(content: str, max_tokens: int) -> List[str]:
    """Fallback chunking by lines for code-heavy content."""
    lines = content.split('\n')
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(line.split())
        if current_tokens + line_tokens > max_tokens and current:
            chunks.append('\n'.join(current))
            current = [line]
            current_tokens = line_tokens
        else:
            current.append(line)
            current_tokens += line_tokens

    if current:
        chunks.append('\n'.join(current))

    return chunks


@dataclass
class SemanticChunk:
    """A semantically coherent chunk with metadata."""
    chunk_id: str
    text: str
    token_count: int
    start_index: int
    end_index: int
    chunk_type: str = "semantic"
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticChunker:
    """Semantic chunker that creates coherent text chunks.

    This chunker analyzes text structure and creates chunks that
    respect semantic boundaries (sentences, paragraphs, topics)
    rather than arbitrary size limits.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 360,
        overlap_sentences: int = 1,
        respect_paragraphs: bool = True,
    ) -> None:
        self._max_tokens = max_chunk_tokens
        self._overlap = overlap_sentences
        self._respect_paragraphs = respect_paragraphs

    def chunk(
        self,
        content: str,
        source_id: str,
    ) -> List[SemanticChunk]:
        """Chunk content with semantic awareness.

        Args:
            content: Text content to chunk
            source_id: Source identifier for chunk IDs

        Returns:
            List of SemanticChunk objects
        """
        raw_chunks = semantic_chunk_text(
            content,
            max_chunk_tokens=self._max_tokens,
            overlap_sentences=self._overlap,
            respect_paragraphs=self._respect_paragraphs,
        )

        chunks: List[SemanticChunk] = []
        current_idx = 0

        for i, text in enumerate(raw_chunks):
            chunks.append(SemanticChunk(
                chunk_id=f"{source_id.split('/')[-1].split('.')[0]}_sem_{i}",
                text=text,
                token_count=len(text.split()),
                start_index=current_idx,
                end_index=current_idx + len(text),
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                }
            ))
            current_idx += len(text)

        return chunks

    def chunk_with_boundaries(
        self,
        content: str,
        source_id: str,
    ) -> List[SemanticChunk]:
        """Chunk content while preserving explicit boundaries.

        This method prioritizes preserving paragraph and header boundaries
        over strict token limits.

        Args:
            content: Text content to chunk
            source_id: Source identifier for chunk IDs

        Returns:
            List of SemanticChunk objects
        """
        # Get paragraph boundaries
        para_boundaries = detect_paragraph_boundaries(content)
        lines = content.split('\n')

        chunks: List[SemanticChunk] = []
        current_lines: List[str] = []
        current_start = 0
        current_tokens = 0
        chunk_idx = 0

        for i, line in enumerate(lines):
            is_boundary = i in para_boundaries
            line_tokens = len(line.split())

            # Check if adding this line would exceed limit
            if current_tokens + line_tokens > self._max_tokens and current_lines:
                # Emit current chunk
                chunks.append(SemanticChunk(
                    chunk_id=f"{source_id.split('/')[-1].split('.')[0]}_bound_{chunk_idx}",
                    text='\n'.join(current_lines),
                    token_count=current_tokens,
                    start_index=current_start,
                    end_index=current_start + len('\n'.join(current_lines)),
                    chunk_type="boundary_preserved",
                    metadata={
                        "chunk_index": chunk_idx,
                        "preserved_boundaries": True,
                    }
                ))

                # Start new chunk
                chunk_idx += 1
                current_lines = [line]
                current_start += len('\n'.join(current_lines[:-1])) + 1
                current_tokens = line_tokens
            elif is_boundary and current_lines:
                # Emit chunk at paragraph boundary
                chunks.append(SemanticChunk(
                    chunk_id=f"{source_id.split('/')[-1].split('.')[0]}_bound_{chunk_idx}",
                    text='\n'.join(current_lines),
                    token_count=current_tokens,
                    start_index=current_start,
                    end_index=current_start + len('\n'.join(current_lines)),
                    chunk_type="boundary_preserved",
                    metadata={
                        "chunk_index": chunk_idx,
                        "preserved_boundaries": True,
                    }
                ))
                chunk_idx += 1
                current_lines = [line]
                current_start += len('\n'.join(current_lines[:-1])) + 1 if len(current_lines) > 1 else 0
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        # Emit final chunk
        if current_lines:
            chunks.append(SemanticChunk(
                chunk_id=f"{source_id.split('/')[-1].split('.')[0]}_bound_{chunk_idx}",
                text='\n'.join(current_lines),
                token_count=current_tokens,
                start_index=current_start,
                end_index=current_start + len('\n'.join(current_lines)),
                chunk_type="boundary_preserved",
                metadata={
                    "chunk_index": chunk_idx,
                    "preserved_boundaries": True,
                }
            ))

        return chunks


def create_semantic_chunker(
    max_chunk_tokens: int = 360,
    overlap_sentences: int = 1,
    respect_paragraphs: bool = True,
) -> SemanticChunker:
    """Factory function to create a semantic chunker.

    Args:
        max_chunk_tokens: Maximum tokens per chunk
        overlap_sentences: Number of sentences to overlap
        respect_paragraphs: Whether to respect paragraph boundaries

    Returns:
        Configured SemanticChunker instance
    """
    return SemanticChunker(
        max_chunk_tokens=max_chunk_tokens,
        overlap_sentences=overlap_sentences,
        respect_paragraphs=respect_paragraphs,
    )


# Backward compatibility - import these from the original module
class CodeLanguage(str, Enum):
    """Supported programming languages for code-aware chunking."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    UNKNOWN = "unknown"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    BASIC = "basic"
    CODE_AWARE = "code_aware"
    TEST_AWARE = "test_aware"  # Prioritizes test-related code
    SEMANTIC = "semantic"  # Uses semantic relevance


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


# Test-aware chunking patterns
_PYTHON_TEST_PATTERNS = [
    # Test functions (def test_*)
    (re.compile(r'^(\s*)def (test_\w+)\(', re.MULTILINE), 'test_function'),
    # Test classes (class Test*)
    (re.compile(r'^class (Test\w+)', re.MULTILINE), 'test_class'),
    # Fixtures (@pytest.fixture, @fixture)
    (re.compile(r'^(\s*)@(?:pytest\.)?fixture', re.MULTILINE), 'fixture'),
    # Setups/teardowns
    (re.compile(r'^(\s*)def (setup|teardown|setUp|tearDown)\(', re.MULTILINE), 'setup_teardown'),
    # Parametrized tests
    (re.compile(r'^(\s*)@pytest\.parametrize', re.MULTILINE), 'parametrized'),
    # Assertions (assert statements - approximate)
    (re.compile(r'\bassert\b', re.MULTILINE), 'assertion'),
    # Test helpers
    (re.compile(r'^(\s*)def (\w*helper\w*)\(', re.MULTILINE), 'test_helper'),
]

# Priority weights for test-aware chunking
_TEST_PRIORITY_WEIGHTS = {
    'test_function': 1.0,
    'test_class': 0.95,
    'fixture': 0.9,
    'setup_teardown': 0.85,
    'parametrized': 0.8,
    'assertion': 0.7,
    'test_helper': 0.6,
    'function': 0.4,
    'class': 0.3,
    'import': 0.2,
    'constant': 0.1,
    'standalone': 0.05,
}


def compute_test_relevance_score(unit: CodeUnit, source_id: str) -> float:
    """Compute a test relevance score for a code unit.

    Args:
        unit: The code unit to score
        source_id: Source identifier (file path)

    Returns:
        Score between 0.0 and 1.0 indicating test relevance
    """
    score = 0.0

    # Check source_id for test indicators
    source_lower = source_id.lower()
    if 'test' in source_lower:
        score += 0.3
    if source_lower.endswith('_test.py') or source_lower.endswith('test_.py'):
        score += 0.2
    if '/tests/' in source_lower or '\\tests\\' in source_lower:
        score += 0.2

    # Check unit type
    unit_type = unit.unit_type.lower()
    if 'test' in unit_type:
        score += 0.4
    if unit_type == 'fixture':
        score += 0.35
    if unit_type in ('function', 'method'):
        # Check if it's a test function
        if unit.name.startswith('test_') or unit.name.endswith('_test'):
            score += 0.5
        # Check for common test patterns
        if any(pat in unit.name.lower() for pat in ['assert', 'verify', 'check', 'should']):
            score += 0.3

    return min(score, 1.0)


def chunk_test_aware(
    content: str,
    source_id: str,
    max_chunk_tokens: int = 360,
    overlap_tokens: int = 40,
) -> List[CodeChunk]:
    """Chunk code with test-awareness for improved test analysis.

    This function prioritizes test-related code sections and creates
    chunks that are more relevant for test failure analysis.

    Args:
        content: Source code content
        source_id: Source identifier
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap for partial chunks

    Returns:
        List of CodeChunk objects with test relevance metadata
    """
    language = detect_language(source_id, content)

    # First, get structure-aware chunks
    struct_chunks = chunk_code_by_structure(
        content=content,
        source_id=source_id,
        language=language,
        max_chunk_tokens=max_chunk_tokens,
        overlap_tokens=overlap_tokens,
    )

    # Add test relevance scores
    for chunk in struct_chunks:
        # Create a temporary unit if needed for scoring
        unit = chunk.code_unit
        if unit is None:
            unit = CodeUnit(
                unit_type=chunk.chunk_type,
                name=chunk.metadata.get('unit_name', 'unknown'),
                content=chunk.text,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                language=chunk.language,
            )

        relevance = compute_test_relevance_score(unit, source_id)
        chunk.metadata['test_relevance_score'] = relevance
        chunk.metadata['is_test_related'] = relevance > 0.3

    # Sort by test relevance (most relevant first)
    struct_chunks.sort(
        key=lambda c: c.metadata.get('test_relevance_score', 0.0),
        reverse=True,
    )

    return struct_chunks


class TestAwareChunker:
    """Test-aware chunker that prioritizes test-related code sections.

    This chunker extends CodeAwareChunker with test-specific heuristics
    to improve retrieval relevance for test failure analysis.
    """

    def __init__(
        self,
        max_chunk_tokens: int = 360,
        overlap_tokens: int = 40,
    ) -> None:
        self._max_tokens = max_chunk_tokens
        self._overlap_tokens = overlap_tokens
        self._base_chunker = CodeAwareChunker(
            max_chunk_tokens=max_chunk_tokens,
            overlap_tokens=overlap_tokens,
        )

    def chunk(
        self,
        content: str,
        source_id: str,
    ) -> List[CodeChunk]:
        """Chunk code with test-awareness.

        Args:
            content: Source code content
            source_id: Source identifier (file path)

        Returns:
            List of CodeChunk objects with test relevance scores
        """
        return chunk_test_aware(
            content=content,
            source_id=source_id,
            max_chunk_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
        )

    def detect_language(self, source_id: str, content: str) -> CodeLanguage:
        """Detect the programming language."""
        return detect_language(source_id, content)

    def get_test_priority_weight(self, chunk: CodeChunk) -> float:
        """Get priority weight for a chunk based on test relevance.

        Args:
            chunk: Code chunk to evaluate

        Returns:
            Priority weight between 0.0 and 1.0
        """
        return chunk.metadata.get('test_relevance_score', 0.0)


def create_chunker(
    strategy: ChunkingStrategy = ChunkingStrategy.CODE_AWARE,
    max_chunk_tokens: int = 360,
    overlap_tokens: int = 40,
) -> Any:
    """Factory function to create a chunker based on strategy.

    Args:
        strategy: The chunking strategy to use
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap for partial chunks

    Returns:
        An appropriate chunker instance
    """
    if strategy == ChunkingStrategy.TEST_AWARE:
        return TestAwareChunker(max_chunk_tokens, overlap_tokens)
    elif strategy == ChunkingStrategy.CODE_AWARE:
        return CodeAwareChunker(max_chunk_tokens, overlap_tokens)
    elif strategy == ChunkingStrategy.SEMANTIC:
        return SemanticChunker(
            max_chunk_tokens=max_chunk_tokens,
            overlap_sentences=overlap_tokens // 40,  # Convert line overlap to sentence overlap
        )
    else:
        # Default to code-aware
        return CodeAwareChunker(max_chunk_tokens, overlap_tokens)
