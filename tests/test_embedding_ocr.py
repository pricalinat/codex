"""Tests for embedding providers and OCR processors."""

import unittest
import tempfile
import os
from pathlib import Path


class TestSentenceTransformerEmbeddingProvider(unittest.TestCase):
    """Tests for SentenceTransformerEmbeddingProvider."""

    def test_provider_creation(self):
        """Test that provider can be created."""
        from src.test_analysis_assistant.retrieval import (
            SentenceTransformerEmbeddingProvider,
        )
        # Provider should be created even without sentence-transformers
        provider = SentenceTransformerEmbeddingProvider()
        self.assertIsNotNone(provider)

    def test_provider_has_is_available_property(self):
        """Test that provider has is_available property."""
        from src.test_analysis_assistant.retrieval import (
            SentenceTransformerEmbeddingProvider,
        )
        provider = SentenceTransformerEmbeddingProvider()
        self.assertTrue(hasattr(provider, "is_available"))

    def test_provider_encode_returns_vectors(self):
        """Test that encode returns embedding vectors."""
        from src.test_analysis_assistant.retrieval import (
            SentenceTransformerEmbeddingProvider,
        )
        provider = SentenceTransformerEmbeddingProvider()
        texts = ["hello world", "test embedding"]
        vectors = provider.encode(texts)
        # Should return fallback vectors when model unavailable
        self.assertIsInstance(vectors, list)
        self.assertEqual(len(texts), len(vectors))

    def test_provider_encode_query(self):
        """Test query encoding."""
        from src.test_analysis_assistant.retrieval import (
            SentenceTransformerEmbeddingProvider,
        )
        provider = SentenceTransformerEmbeddingProvider()
        query_vec = provider.encode_query("test query")
        self.assertIsInstance(query_vec, list)

    def test_provider_prepare_corpus(self):
        """Test corpus preparation."""
        from src.test_analysis_assistant.retrieval import (
            SentenceTransformerEmbeddingProvider,
        )
        provider = SentenceTransformerEmbeddingProvider()
        texts = ["doc one", "doc two", "doc three"]
        vectors = provider.prepare_corpus(texts)
        self.assertEqual(len(texts), len(vectors))


class TestCreateEmbeddingProvider(unittest.TestCase):
    """Tests for create_embedding_provider factory function."""

    def test_create_dummy_provider(self):
        """Test creating dummy provider."""
        from src.test_analysis_assistant.retrieval import (
            create_embedding_provider,
            DummyEmbeddingProvider,
        )
        provider = create_embedding_provider("dummy")
        self.assertIsInstance(provider, DummyEmbeddingProvider)

    def test_create_tfidf_provider(self):
        """Test creating TF-IDF provider."""
        from src.test_analysis_assistant.retrieval import (
            create_embedding_provider,
            TFIDFEmbeddingProvider,
        )
        provider = create_embedding_provider("tfidf")
        self.assertIsInstance(provider, TFIDFEmbeddingProvider)

    def test_create_sentence_transformer_provider(self):
        """Test creating sentence-transformer provider."""
        from src.test_analysis_assistant.retrieval import (
            create_embedding_provider,
            SentenceTransformerEmbeddingProvider,
        )
        provider = create_embedding_provider("sentence-transformer")
        self.assertIsInstance(provider, SentenceTransformerEmbeddingProvider)

    def test_create_auto_provider(self):
        """Test auto provider selection."""
        from src.test_analysis_assistant.retrieval import create_embedding_provider
        # Auto should return some provider
        provider = create_embedding_provider("auto")
        self.assertIsNotNone(provider)

    def test_create_with_custom_params(self):
        """Test creating provider with custom parameters."""
        from src.test_analysis_assistant.retrieval import (
            create_embedding_provider,
            TFIDFEmbeddingProvider,
        )
        provider = create_embedding_provider("tfidf", max_features=128)
        self.assertIsInstance(provider, TFIDFEmbeddingProvider)

    def test_invalid_provider_type_raises(self):
        """Test that invalid provider type raises ValueError."""
        from src.test_analysis_assistant.retrieval import create_embedding_provider
        with self.assertRaises(ValueError):
            create_embedding_provider("invalid_type")


class TestTesseractOCRProcessor(unittest.TestCase):
    """Tests for TesseractOCRProcessor."""

    def test_processor_creation(self):
        """Test that processor can be created."""
        from src.test_analysis_assistant.multimodal import TesseractOCRProcessor
        processor = TesseractOCRProcessor()
        self.assertIsNotNone(processor)

    def test_processor_has_is_available_property(self):
        """Test that processor has is_available property."""
        from src.test_analysis_assistant.multimodal import TesseractOCRProcessor
        processor = TesseractOCRProcessor()
        self.assertTrue(hasattr(processor, "is_available"))

    def test_processor_has_supported_languages(self):
        """Test that processor has supported languages."""
        from src.test_analysis_assistant.multimodal import TesseractOCRProcessor
        processor = TesseractOCRProcessor()
        self.assertTrue(hasattr(processor, "supported_languages"))
        # Should have at least English
        self.assertIn("eng", processor.supported_languages)

    def test_processor_process_image_path_returns_ocr_result(self):
        """Test processing image from path returns OCRResult."""
        from src.test_analysis_assistant.multimodal import TesseractOCRProcessor
        processor = TesseractOCRProcessor()
        # Non-existent path should still return OCRResult (with fallback or error)
        result = processor.process_image_path("/nonexistent/image.png")
        self.assertTrue(hasattr(result, "text"))
        self.assertTrue(hasattr(result, "ocr_confidence"))

    def test_processor_process_image_bytes_returns_ocr_result(self):
        """Test processing image from bytes returns OCRResult."""
        from src.test_analysis_assistant.multimodal import TesseractOCRProcessor
        processor = TesseractOCRProcessor()
        # Empty bytes should still return OCRResult
        result = processor.process_image_bytes(b"")
        self.assertTrue(hasattr(result, "text"))
        self.assertTrue(hasattr(result, "ocr_confidence"))

    def test_processor_with_custom_psm(self):
        """Test creating processor with custom PSM."""
        from src.test_analysis_assistant.multimodal import TesseractOCRProcessor
        processor = TesseractOCRProcessor(psm=6)
        self.assertIsNotNone(processor)


class TestOCRIntegration(unittest.TestCase):
    """Integration tests for OCR with multimodal ingestor."""

    def test_multimodal_ingestor_with_tesseract(self):
        """Test MultimodalIngestor with TesseractOCRProcessor."""
        from src.test_analysis_assistant.multimodal import (
            MultimodalIngestor,
            TesseractOCRProcessor,
        )
        processor = TesseractOCRProcessor()
        ingestor = MultimodalIngestor(ocr_processor=processor)
        self.assertIsNotNone(ingestor)


if __name__ == "__main__":
    unittest.main()
