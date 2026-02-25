import unittest

from src.test_analysis_assistant.code_chunker import (
    ChunkingStrategy,
    CodeAwareChunker,
    CodeLanguage,
    CodeUnit,
    TestAwareChunker,
    chunk_code_by_structure,
    chunk_test_aware,
    compute_test_relevance_score,
    create_chunker,
    detect_language,
    extract_code_units,
    extract_python_units,
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


if __name__ == "__main__":
    unittest.main()
