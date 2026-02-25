import unittest

from src.test_analysis_assistant.code_chunker import (
    CodeAwareChunker,
    CodeLanguage,
    CodeUnit,
    chunk_code_by_structure,
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


if __name__ == "__main__":
    unittest.main()
