"""Tests for multimodal ingestion module."""

import unittest

from src.test_analysis_assistant.multimodal import (
    ExtractedTable,
    ImageOCRProcessor,
    ModalityType,
    MultimodalIngestor,
    OCRResult,
    ProcessedArtifact,
    TableExtractor,
)
from src.test_analysis_assistant.retrieval import ArtifactBundle, SourceType


class TestTableExtractor(unittest.TestCase):
    """Tests for table extraction functionality."""

    def test_extract_from_markdown_basic(self):
        """Test basic markdown table extraction."""
        extractor = TableExtractor()
        content = """| Name | Age | City |
|------|-----|------|
| Alice | 30 | NYC |
| Bob | 25 | LA |
"""
        tables = extractor.extract_from_markdown(content)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].headers, ["Name", "Age", "City"])
        self.assertEqual(len(tables[0].rows), 2)
        self.assertEqual(tables[0].rows[0], ["Alice", "30", "NYC"])

    def test_extract_from_markdown_multiple_tables(self):
        """Test extraction of multiple markdown tables."""
        extractor = TableExtractor()
        content = """| A | B |
| - | - |
| 1 | 2 |

| X | Y |
| - | - |
| 3 | 4 |
"""
        tables = extractor.extract_from_markdown(content)
        self.assertEqual(len(tables), 2)

    def test_extract_from_markdown_no_tables(self):
        """Test markdown with no tables returns empty list."""
        extractor = TableExtractor()
        content = """# Just regular markdown

Some text without tables.
"""
        tables = extractor.extract_from_markdown(content)
        self.assertEqual(len(tables), 0)

    def test_extract_from_csv(self):
        """Test CSV table extraction."""
        extractor = TableExtractor()
        content = "Name,Age,City\nAlice,30,NYC\nBob,25,LA"
        tables = extractor.extract_from_csv(content)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].headers, ["Name", "Age", "City"])
        self.assertEqual(len(tables[0].rows), 2)

    def test_extract_from_csv_empty(self):
        """Test empty CSV returns no tables."""
        extractor = TableExtractor()
        tables = extractor.extract_from_csv("")
        self.assertEqual(len(tables), 0)

    def test_extract_from_html(self):
        """Test HTML table extraction."""
        extractor = TableExtractor()
        content = """<table>
<tr><th>Name</th><th>Age</th></tr>
<tr><td>Alice</td><td>30</td></tr>
<tr><td>Bob</td><td>25</td></tr>
</table>"""
        tables = extractor.extract_from_html(content)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].headers, ["Name", "Age"])
        self.assertEqual(len(tables[0].rows), 2)

    def test_extract_auto_detect_markdown(self):
        """Test auto-detection of markdown format."""
        extractor = TableExtractor()
        content = "| A | B |\n| - | - |\n| 1 | 2 |"
        tables = extractor.extract(content, format_hint="auto")
        self.assertEqual(len(tables), 1)

    def test_extract_auto_detect_csv(self):
        """Test auto-detection of CSV format."""
        extractor = TableExtractor()
        content = "A,B\n1,2"
        tables = extractor.extract(content, format_hint="auto")
        self.assertEqual(len(tables), 1)


class TestImageOCRProcessor(unittest.TestCase):
    """Tests for OCR processing functionality."""

    def test_ocr_processor_init(self):
        """Test OCR processor initialization."""
        processor = ImageOCRProcessor(language="eng")
        self.assertEqual(processor._language, "eng")

    def test_process_image_bytes_returns_stub(self):
        """Test that image processing returns stub result."""
        processor = ImageOCRProcessor()
        result = processor.process_image_bytes(b"fake image data", "test_image")
        self.assertEqual(result.image_id, "test_image")
        self.assertIn("stub", result.metadata)

    def test_process_image_path_returns_stub(self):
        """Test that image path processing returns stub result."""
        processor = ImageOCRProcessor()
        result = processor.process_image_path("/path/to/image.png")
        self.assertEqual(result.image_id, "/path/to/image.png")
        self.assertIn("stub", result.metadata)

    def test_process_image_url_returns_stub(self):
        """Test that image URL processing returns stub result."""
        processor = ImageOCRProcessor()
        result = processor.process_image_url("https://example.com/image.png")
        self.assertEqual(result.image_id, "https://example.com/image.png")
        self.assertIn("stub", result.metadata)


class TestMultimodalIngestor(unittest.TestCase):
    """Tests for multimodal ingestion."""

    def test_multimodal_ingestor_init(self):
        """Test multimodal ingestor initialization."""
        ingestor = MultimodalIngestor()
        self.assertIsNotNone(ingestor._table_extractor)
        self.assertIsNotNone(ingestor._ocr_processor)

    def test_process_artifact_with_text(self):
        """Test processing artifact with text content."""
        ingestor = MultimodalIngestor()
        artifact = ArtifactBundle(
            source_id="doc:1",
            source_type=SourceType.REQUIREMENTS,
            text="This is some test content with multiple paragraphs.\n\nMore content here.",
        )
        result = ingestor.process_artifact_bundle(artifact)
        # Paragraphs are kept together if they fit in chunk size (default 500)
        self.assertGreaterEqual(len(result.text_chunks), 1)
        self.assertIn("test content", result.text_chunks[0])

    def test_process_artifact_with_tables(self):
        """Test processing artifact with markdown tables."""
        ingestor = MultimodalIngestor()
        artifact = ArtifactBundle(
            source_id="doc:2",
            source_type=SourceType.REQUIREMENTS,
            text="""| Name | Value |
| ---- | ----- |
| A | 1 |
| B | 2 |
""",
        )
        result = ingestor.process_artifact_bundle(artifact, include_tables=True)
        self.assertEqual(len(result.table_data), 1)
        self.assertEqual(result.table_data[0].headers, ["Name", "Value"])

    def test_process_artifact_with_images(self):
        """Test processing artifact with image references."""
        ingestor = MultimodalIngestor()
        artifact = ArtifactBundle(
            source_id="doc:3",
            source_type=SourceType.SYSTEM_ANALYSIS,
            text="Document with image",
            images=["image1.png", "image2.jpg"],
        )
        result = ingestor.process_artifact_bundle(artifact, include_images=True)
        self.assertEqual(len(result.ocr_results), 2)

    def test_chunk_text_small(self):
        """Test that small text is not chunked."""
        ingestor = MultimodalIngestor()
        text = "Short text"
        chunks = ingestor._chunk_text(text, max_chunk_size=500)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Short text")

    def test_chunk_text_large(self):
        """Test that large text is chunked."""
        ingestor = MultimodalIngestor()
        text = "A" * 600
        chunks = ingestor._chunk_text(text, max_chunk_size=500)
        self.assertEqual(len(chunks), 2)

    def test_determine_modality_text_only(self):
        """Test modality determination for text only."""
        artifact = ProcessedArtifact(
            source_id="test",
            source_type="test",
            text_chunks=["content"],
        )
        ingestor = MultimodalIngestor()
        modality = ingestor._determine_modality(artifact)
        self.assertEqual(modality, ModalityType.TEXT.value)

    def test_determine_modality_compound(self):
        """Test modality determination for compound content."""
        artifact = ProcessedArtifact(
            source_id="test",
            source_type="test",
            text_chunks=["content"],
            table_data=[ExtractedTable(table_id="t1", headers=[], rows=[])],
            ocr_results=[OCRResult(image_id="i1", text="ocr text", ocr_confidence=0.5)],
        )
        ingestor = MultimodalIngestor()
        modality = ingestor._determine_modality(artifact)
        self.assertEqual(modality, ModalityType.COMPOUND.value)

    def test_table_to_text(self):
        """Test conversion of table to text."""
        table = ExtractedTable(
            table_id="test",
            headers=["Name", "Age"],
            rows=[["Alice", "30"], ["Bob", "25"]],
        )
        ingestor = MultimodalIngestor()
        text = ingestor._table_to_text(table)
        self.assertIn("Name | Age", text)
        self.assertIn("Alice | 30", text)
        self.assertIn("Bob | 25", text)

    def test_extract_chunks_for_ingestion(self):
        """Test extraction of chunks ready for ingestion."""
        ingestor = MultimodalIngestor()
        artifact = ArtifactBundle(
            source_id="doc:4",
            source_type=SourceType.REQUIREMENTS,
            text="Simple content",
        )
        chunks = ingestor.extract_chunks_for_ingestion(artifact)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["source_id"], "doc:4")
        self.assertEqual(chunks[0]["modality"], "text")


class TestExtractedTable(unittest.TestCase):
    """Tests for ExtractedTable dataclass."""

    def test_extracted_table_creation(self):
        """Test creating an ExtractedTable."""
        table = ExtractedTable(
            table_id="t1",
            headers=["Col1", "Col2"],
            rows=[["a", "b"], ["c", "d"]],
            caption="Test table",
            extraction_confidence=0.9,
        )
        self.assertEqual(table.table_id, "t1")
        self.assertEqual(len(table.rows), 2)
        self.assertEqual(table.extraction_confidence, 0.9)


class TestOCRResult(unittest.TestCase):
    """Tests for OCRResult dataclass."""

    def test_ocr_result_creation(self):
        """Test creating an OCRResult."""
        result = OCRResult(
            image_id="img1",
            text="Extracted text",
            language="en",
            ocr_confidence=0.85,
        )
        self.assertEqual(result.image_id, "img1")
        self.assertEqual(result.text, "Extracted text")
        self.assertEqual(result.ocr_confidence, 0.85)


class TestModalityType(unittest.TestCase):
    """Tests for ModalityType enum."""

    def test_modality_types(self):
        """Test all modality types are defined."""
        self.assertEqual(ModalityType.TEXT.value, "text")
        self.assertEqual(ModalityType.TABLE.value, "table")
        self.assertEqual(ModalityType.IMAGE.value, "image")
        self.assertEqual(ModalityType.COMPOUND.value, "compound")


if __name__ == "__main__":
    unittest.main()
