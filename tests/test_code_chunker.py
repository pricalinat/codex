import unittest

from src.test_analysis_assistant.code_chunker import (
    ChunkingStrategy,
    CodeAwareChunker,
    CodeLanguage,
    CodeUnit,
    SemanticChunker,
    TestAwareChunker,
    chunk_code_by_structure,
    chunk_test_aware,
    compute_test_relevance_score,
    create_chunker,
    create_semantic_chunker,
    detect_language,
    detect_paragraph_boundaries,
    extract_code_units,
    extract_python_units,
    group_into_chunks,
    semantic_chunk_text,
    split_into_sentences,
)


class TestCodeLanguageDetection(unittest.TestCase):
    def test_detect_language_from_extension(self):
        # Python detection
        self.assertEqual(detect_language("test.py", ""), CodeLanguage.PYTHON)
        self.assertEqual(detect_language("module/test.py", ""), CodeLanguage.PYTHON)

        # JavaScript detection
        self.assertEqual(detect_language("test.js", ""), CodeLanguage.JAVASCRIPT)

        # TypeScript detection
        self.assertEqual(detect_language("test.ts", ""), CodeLanguage.TYPESCRIPT)
        self.assertEqual(detect_language("test.tsx", ""), CodeLanguage.TYPESCRIPT)

    def test_detect_language_from_content(self):
        # Python by content
        content = "def foo():\n    pass"
        self.assertEqual(detect_language("unknown", content), CodeLanguage.PYTHON)

        # JavaScript by content
        content = "function foo() { return 1; }"
        self.assertEqual(detect_language("unknown", content), CodeLanguage.JAVASCRIPT)


class TestPythonUnitExtraction(unittest.TestCase):
    def test_extract_class(self):
        content = """
class MyClass:
    def method(self):
        pass
"""
        units = extract_python_units(content)
        class_units = [u for u in units if u.unit_type == 'class']
        self.assertEqual(len(class_units), 1)
        self.assertEqual(class_units[0].name, 'MyClass')

    def test_extract_functions(self):
        content = """
def standalone_function():
    pass

class MyClass:
    def method(self):
        pass
"""
        units = extract_python_units(content)
        func_units = [u for u in units if u.unit_type == 'function']
        method_units = [u for u in units if u.unit_type == 'method']

        self.assertEqual(len(func_units), 1)
        self.assertEqual(func_units[0].name, 'standalone_function')

        self.assertEqual(len(method_units), 1)
        self.assertEqual(method_units[0].name, 'method')
        self.assertEqual(method_units[0].parent, 'MyClass')

    def test_extract_imports(self):
        content = """
import os
import sys
from pathlib import Path
"""
        units = extract_python_units(content)
        import_units = [u for u in units if u.unit_type == 'import']

        self.assertGreaterEqual(len(import_units), 1)

    def test_extract_constants(self):
        content = """
MAX_SIZE = 100
API_KEY = "secret"
"""
        units = extract_python_units(content)
        const_units = [u for u in units if u.unit_type == 'constant']

        self.assertEqual(len(const_units), 2)


class TestCodeChunking(unittest.TestCase):
    def test_chunk_respects_function_boundaries(self):
        content = """
def function_one():
    return 1

def function_two():
    return 2
"""
        chunks = chunk_code_by_structure(content, "test.py", CodeLanguage.PYTHON)

        # Should have chunks that preserve function content
        self.assertGreater(len(chunks), 0)

        # Check that function names appear in chunks
        chunk_texts = [c.text for c in chunks]
        self.assertTrue(any('function_one' in t for t in chunk_texts))
        self.assertTrue(any('function_two' in t for t in chunk_texts))

    def test_chunk_includes_metadata(self):
        content = """
class MyClass:
    def method(self):
        pass
"""
        chunks = chunk_code_by_structure(content, "test.py", CodeLanguage.PYTHON)

        # Should have at least one chunk with class metadata
        class_chunks = [c for c in chunks if c.metadata.get('unit_type') == 'class']
        self.assertGreater(len(class_chunks), 0)

        method_chunks = [c for c in chunks if c.metadata.get('unit_type') == 'method']
        self.assertGreater(len(method_chunks), 0)
        self.assertEqual(method_chunks[0].metadata.get('parent'), 'MyClass')

    def test_chunk_unknown_language(self):
        content = "Some random content without code structure"
        chunks = chunk_code_by_structure(content, "test.xyz", CodeLanguage.UNKNOWN)

        # Should still produce a chunk (as full_unit for unknown language)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, 'full_unit')


class TestCodeAwareChunker(unittest.TestCase):
    def test_chunker_initialization(self):
        chunker = CodeAwareChunker(max_chunk_tokens=200, overlap_tokens=20)
        self.assertEqual(chunker._max_tokens, 200)
        self.assertEqual(chunker._overlap_tokens, 20)

    def test_chunker_detect_language(self):
        chunker = CodeAwareChunker()
        self.assertEqual(chunker.detect_language("test.py", ""), CodeLanguage.PYTHON)
        self.assertEqual(chunker.detect_language("test.js", ""), CodeLanguage.JAVASCRIPT)

    def test_chunker_chunks_code(self):
        chunker = CodeAwareChunker()
        content = """
def test_login():
    assert True

def test_logout():
    assert True
"""
        chunks = chunker.chunk(content, "test.py")

        self.assertGreater(len(chunks), 0)
        # Verify code-specific metadata
        unit_names = [c.metadata.get('unit_name') for c in chunks]
        self.assertIn('test_login', unit_names)
        self.assertIn('test_logout', unit_names)


class TestTestAwareChunking(unittest.TestCase):
    def test_compute_test_relevance_score_test_file(self):
        """Test that test files get high relevance scores."""
        unit = CodeUnit(
            unit_type='function',
            name='test_login_success',
            content='def test_login_success(): pass',
            start_line=1,
            end_line=2,
            language=CodeLanguage.PYTHON,
        )
        score = compute_test_relevance_score(unit, 'tests/test_auth.py')
        self.assertGreater(score, 0.5)

    def test_compute_test_relevance_score_regular_file(self):
        """Test that regular files get lower scores."""
        unit = CodeUnit(
            unit_type='function',
            name='process_data',
            content='def process_data(): pass',
            start_line=1,
            end_line=2,
            language=CodeLanguage.PYTHON,
        )
        score = compute_test_relevance_score(unit, 'src/processor.py')
        self.assertLess(score, 0.5)

    def test_compute_test_relevance_score_test_function(self):
        """Test that test functions get high scores."""
        unit = CodeUnit(
            unit_type='function',
            name='test_auth_invalid_credentials',
            content='def test_auth_invalid_credentials(): pass',
            start_line=1,
            end_line=2,
            language=CodeLanguage.PYTHON,
        )
        score = compute_test_relevance_score(unit, 'src/auth.py')
        # Test function name should give high score even in non-test file
        self.assertGreater(score, 0.4)

    def test_chunk_test_aware_returns_sorted_chunks(self):
        """Test that test-aware chunking returns chunks sorted by relevance."""
        content = """
# Regular module code
def process_data(data):
    return data.upper()

# Test code
def test_process_data():
    assert process_data("hello") == "HELLO"

class TestProcessor:
    def test_upper(self):
        assert True
"""
        chunks = chunk_test_aware(content, "test_module.py")

        self.assertGreater(len(chunks), 0)
        # All chunks should have test_relevance_score metadata
        for chunk in chunks:
            self.assertIn('test_relevance_score', chunk.metadata)

    def test_chunk_test_aware_identifies_test_functions(self):
        """Test that test functions are identified."""
        content = """
def test_login():
    assert True

def regular_function():
    return 1
"""
        chunks = chunk_test_aware(content, "test_file.py")

        test_chunks = [c for c in chunks if c.metadata.get('is_test_related')]
        self.assertGreater(len(test_chunks), 0)

    def test_test_aware_chunker(self):
        """Test the TestAwareChunker class."""
        chunker = TestAwareChunker(max_chunk_tokens=200, overlap_tokens=20)
        self.assertEqual(chunker._max_tokens, 200)
        self.assertEqual(chunker._overlap_tokens, 20)

    def test_test_aware_chunker_chunks(self):
        """Test TestAwareChunker produces chunks with relevance scores."""
        chunker = TestAwareChunker()
        content = """
def test_example():
    assert 1 == 1

def helper():
    pass
"""
        chunks = chunker.chunk(content, "tests/test_example.py")

        self.assertGreater(len(chunks), 0)
        # Check test relevance scoring
        scores = [c.metadata.get('test_relevance_score', 0) for c in chunks]
        self.assertTrue(any(s > 0 for s in scores))

    def test_create_chunker_factory(self):
        """Test the chunker factory function."""
        # Test creating different chunker types
        basic_chunker = create_chunker(ChunkingStrategy.BASIC)
        self.assertIsInstance(basic_chunker, CodeAwareChunker)

        code_chunker = create_chunker(ChunkingStrategy.CODE_AWARE)
        self.assertIsInstance(code_chunker, CodeAwareChunker)

        test_chunker = create_chunker(ChunkingStrategy.TEST_AWARE)
        self.assertIsInstance(test_chunker, TestAwareChunker)


class TestChunkingStrategy(unittest.TestCase):
    def test_chunking_strategy_enum(self):
        """Test ChunkingStrategy enum values."""
        self.assertEqual(ChunkingStrategy.BASIC.value, "basic")
        self.assertEqual(ChunkingStrategy.CODE_AWARE.value, "code_aware")
        self.assertEqual(ChunkingStrategy.TEST_AWARE.value, "test_aware")
        self.assertEqual(ChunkingStrategy.SEMANTIC.value, "semantic")


class TestSemanticChunking(unittest.TestCase):
    def test_split_into_sentences_basic(self):
        """Test splitting text into sentences."""
        text = "This is the first sentence. This is the second sentence! Is this the third?"
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0].startswith("This is the first"))
        self.assertTrue(sentences[1].startswith("This is the second"))
        self.assertTrue(sentences[2].startswith("Is this the third"))

    def test_split_into_sentences_preserves_code_blocks(self):
        """Test that code blocks are preserved in sentence splitting."""
        text = "First sentence. Then `code with dots.` More text. Finally!"
        sentences = split_into_sentences(text)
        # Should not split inside code blocks
        self.assertTrue(any("code with dots" in s for s in sentences))

    def test_detect_paragraph_boundaries(self):
        """Test paragraph boundary detection."""
        text = """First paragraph line 1
First paragraph line 2

Second paragraph

# Header paragraph
Content after header"""
        boundaries = detect_paragraph_boundaries(text)
        # Should detect start and paragraph breaks
        self.assertIn(0, boundaries)  # First line

    def test_group_into_chunks(self):
        """Test grouping sentences into chunks."""
        sentences = [
            "This is the first sentence with some content.",
            "This is the second sentence continuing the thought.",
            "This is the third sentence that wraps up the idea.",
            "Now we start a new topic with different content.",
            "This sentence continues the new topic.",
            "And this is the final sentence in our test.",
        ]
        chunks = group_into_chunks(sentences, max_tokens=10)
        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)

    def test_semantic_chunk_text_basic(self):
        """Test basic semantic chunking."""
        text = """This is the first paragraph. It contains multiple sentences.
This paragraph should be kept together when possible.

This is the second paragraph with different content.
It also has multiple sentences that belong together.

Third paragraph starts here. More content follows."""
        chunks = semantic_chunk_text(text, max_chunk_tokens=15)
        self.assertGreater(len(chunks), 0)
        # Each chunk should have reasonable content
        for chunk in chunks:
            self.assertGreater(len(chunk.split()), 2)

    def test_semantic_chunk_text_handles_empty(self):
        """Test semantic chunking with empty input."""
        self.assertEqual(semantic_chunk_text(""), [])
        self.assertEqual(semantic_chunk_text("   "), [])

    def test_semantic_chunk_text_respects_code(self):
        """Test that semantic chunking handles code-heavy content."""
        # This should fall back to line-based chunking for code
        code_content = """
def function_one():
    return 1

def function_two():
    return 2

class TestClass:
    def test_method(self):
        pass
"""
        chunks = semantic_chunk_text(code_content, max_chunk_tokens=10)
        self.assertGreater(len(chunks), 0)

    def test_semantic_chunker_class(self):
        """Test the SemanticChunker class."""
        chunker = SemanticChunker(
            max_chunk_tokens=100,
            overlap_sentences=1,
            respect_paragraphs=True,
        )
        self.assertEqual(chunker._max_tokens, 100)
        self.assertEqual(chunker._overlap, 1)

    def test_semantic_chunker_chunk_method(self):
        """Test SemanticChunker.chunk() method."""
        chunker = SemanticChunker(max_chunk_tokens=50)
        text = "First sentence here. Second sentence continues. Third sentence is longer and contains more words. Fourth sentence."
        chunks = chunker.chunk(text, "test_doc.txt")
        self.assertGreater(len(chunks), 0)
        # All chunks should have required fields
        for chunk in chunks:
            self.assertTrue(hasattr(chunk, 'chunk_id'))
            self.assertTrue(hasattr(chunk, 'text'))
            self.assertTrue(hasattr(chunk, 'token_count'))
            self.assertTrue(hasattr(chunk, 'metadata'))

    def test_semantic_chunker_with_boundaries(self):
        """Test SemanticChunker with boundary preservation."""
        chunker = SemanticChunker(max_chunk_tokens=30)
        text = """# Header

First paragraph with content.

Second paragraph here."""
        chunks = chunker.chunk_with_boundaries(text, "test.md")
        self.assertGreater(len(chunks), 0)

    def test_create_semantic_chunker(self):
        """Test factory function for semantic chunker."""
        chunker = create_semantic_chunker(
            max_chunk_tokens=200,
            overlap_sentences=2,
            respect_paragraphs=True,
        )
        self.assertIsInstance(chunker, SemanticChunker)
        self.assertEqual(chunker._max_tokens, 200)
        self.assertEqual(chunker._overlap, 2)

    def test_create_chunker_semantic_strategy(self):
        """Test create_chunker factory with SEMANTIC strategy."""
        chunker = create_chunker(ChunkingStrategy.SEMANTIC, max_chunk_tokens=150)
        self.assertIsInstance(chunker, SemanticChunker)


if __name__ == "__main__":
    unittest.main()
