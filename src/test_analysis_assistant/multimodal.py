"""Multimodal ingestion support for processing mixed content (tables, images).

This module provides stubs and interfaces for:
- Table extraction from various document formats
- Image OCR processing
- Combined multimodal content ingestion
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class ModalityType(str, Enum):
    """Types of content modalities."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    COMPOUND = "compound"


@dataclass
class ExtractedTable:
    """Extracted table data with metadata."""

    table_id: str
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    source_location: Optional[str] = None
    extraction_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRResult:
    """OCR processing result for images."""

    image_id: str
    text: str
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    ocr_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedArtifact:
    """Processed artifact with extracted content from all modalities."""

    source_id: str
    source_type: str
    text_chunks: List[str] = field(default_factory=list)
    table_data: List[ExtractedTable] = field(default_factory=list)
    ocr_results: List[OCRResult] = field(default_factory=list)
    processing_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TableExtractor:
    """Extracts tables from various document formats.

    This is a stub implementation that provides basic table extraction.
    In production, this would integrate with libraries like:
    - tabula-py for PDFs
    - openpyxl for Excel
    - python-docx for Word documents
    """

    SUPPORTED_FORMATS = {"markdown", "csv", "html", "excel", "pdf", "docx"}

    def extract_from_markdown(self, content: str) -> List[ExtractedTable]:
        """Extract tables from markdown content.

        Args:
            content: Markdown text potentially containing tables

        Returns:
            List of extracted tables
        """
        tables = []
        lines = content.split("\n")
        in_table = False
        headers: List[str] = []
        rows: List[List[str]] = []
        table_id = 0

        for line in lines:
            stripped = line.strip()
            # Check for table markdown format: | header | header |
            if stripped.startswith("|") and "|" in stripped[1:]:
                parts = [p.strip() for p in stripped.split("|")[1:-1]]
                # Check if it's a separator line (all dashes and/or colons)
                if all(self._is_separator(p) for p in parts):
                    # This is just the separator, skip
                    continue
                if not in_table:
                    # First row is headers
                    headers = parts
                    in_table = True
                else:
                    rows.append(parts)
            elif in_table and not stripped.startswith("|"):
                # End of table
                if headers and rows:
                    tables.append(ExtractedTable(
                        table_id=f"md_table_{table_id}",
                        headers=headers,
                        rows=rows,
                        extraction_confidence=0.85,
                    ))
                    table_id += 1
                headers = []
                rows = []
                in_table = False

        # Handle table at end of content
        if headers and rows:
            tables.append(ExtractedTable(
                table_id=f"md_table_{table_id}",
                headers=headers,
                rows=rows,
                extraction_confidence=0.85,
            ))

        return tables

    def _is_separator(self, cell: str) -> bool:
        """Check if a cell is a markdown table separator."""
        return bool(re.match(r"^[-:]+$", cell))

    def extract_from_csv(self, content: str) -> List[ExtractedTable]:
        """Extract tables from CSV content.

        Args:
            content: CSV text

        Returns:
            List of extracted tables (usually just one)
        """
        import io
        import csv

        tables = []
        try:
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
            if rows:
                # First row as headers, rest as data
                tables.append(ExtractedTable(
                    table_id="csv_table_0",
                    headers=rows[0],
                    rows=rows[1:] if len(rows) > 1 else [],
                    extraction_confidence=0.95,
                ))
        except Exception:
            pass

        return tables

    def extract_from_html(self, content: str) -> List[ExtractedTable]:
        """Extract tables from HTML content.

        Args:
            content: HTML text potentially containing tables

        Returns:
            List of extracted tables
        """
        tables = []
        table_id = 0

        # Simple regex-based table extraction (stub)
        # Production would use BeautifulSoup or lxml
        table_pattern = r"<table[^>]*>(.*?)</table>"
        for match in re.finditer(table_pattern, content, re.DOTALL | re.IGNORECASE):
            table_content = match.group(1)

            # Extract headers
            header_pattern = r"<th[^>]*>(.*?)</th>"
            headers = [re.sub(r"<[^>]+>", "", h).strip() for h in re.findall(header_pattern, table_content, re.DOTALL)]

            # Extract data rows (only td, not th)
            row_pattern = r"<tr[^>]*>(.*?)</tr>"
            rows = []
            for row_match in re.finditer(row_pattern, table_content, re.DOTALL):
                row_content = row_match.group(1)
                # Only look for td cells, not th cells (which are headers)
                cell_pattern = r"<td[^>]*>(.*?)</td>"
                cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(cell_pattern, row_content, re.DOTALL)]
                if cells:
                    rows.append(cells)

            if headers or rows:
                # If we have headers, rows already contains only data rows
                # (header row had th, not td, so wasn't added)
                tables.append(ExtractedTable(
                    table_id=f"html_table_{table_id}",
                    headers=headers if headers else rows[0] if rows else [],
                    rows=rows if not headers else rows,  # Already filtered
                    source_location="html",
                    extraction_confidence=0.75,
                ))
                table_id += 1

        return tables

    def extract(self, content: str, format_hint: str = "auto") -> List[ExtractedTable]:
        """Extract tables from content with automatic format detection.

        Args:
            content: Content containing tables
            format_hint: Format hint (markdown, csv, html, auto)

        Returns:
            List of extracted tables
        """
        if format_hint == "auto":
            # Auto-detect format
            if content.strip().startswith("|"):
                format_hint = "markdown"
            elif content.strip().startswith("<"):
                format_hint = "html"
            elif "," in content.split("\n")[0] if "\n" in content else "":
                format_hint = "csv"

        if format_hint == "markdown":
            return self.extract_from_markdown(content)
        elif format_hint == "csv":
            return self.extract_from_csv(content)
        elif format_hint == "html":
            return self.extract_from_html(content)

        return []


class ImageOCRProcessor:
    """Processes images for OCR extraction.

    This is a stub implementation that provides the interface for OCR.
    In production, this would integrate with:
    - pytesseract (Tesseract OCR)
    - Google Cloud Vision API
    - AWS Textract
    - Azure Computer Vision
    """

    def __init__(self, language: str = "eng") -> None:
        """Initialize OCR processor.

        Args:
            language: Language code for OCR (default: eng)
        """
        self._language = language

    def process_image_bytes(self, image_data: bytes, image_id: str = "image") -> OCRResult:
        """Process image from bytes.

        Args:
            image_data: Raw image bytes
            image_id: Identifier for the image

        Returns:
            OCR result with extracted text
        """
        # Stub: In production, this would call actual OCR
        return OCRResult(
            image_id=image_id,
            text="[OCR stub: image content would be extracted here]",
            language=self._language,
            ocr_confidence=0.0,
            metadata={"stub": True, "note": "Requires OCR library integration"},
        )

    def process_image_path(self, image_path: str) -> OCRResult:
        """Process image from file path.

        Args:
            image_path: Path to image file

        Returns:
            OCR result with extracted text
        """
        # Stub: Would read file and call process_image_bytes
        return OCRResult(
            image_id=image_path,
            text="[OCR stub: image content would be extracted here]",
            language=self._language,
            ocr_confidence=0.0,
            metadata={"stub": True, "path": image_path},
        )

    def process_image_url(self, image_url: str) -> OCRResult:
        """Process image from URL.

        Args:
            image_url: URL to image

        Returns:
            OCR result with extracted text
        """
        # Stub: Would fetch and process
        return OCRResult(
            image_id=image_url,
            text="[OCR stub: image content would be extracted here]",
            language=self._language,
            ocr_confidence=0.0,
            metadata={"stub": True, "url": image_url},
        )


class MultimodalIngestor:
    """Processes mixed-modality content for ingestion into RAG pipeline.

    Combines text extraction, table extraction, and OCR processing
    to produce normalized chunks suitable for the retrieval engine.
    """

    def __init__(
        self,
        table_extractor: Optional[TableExtractor] = None,
        ocr_processor: Optional[ImageOCRProcessor] = None,
    ) -> None:
        """Initialize multimodal ingestor.

        Args:
            table_extractor: Table extraction handler
            ocr_processor: OCR processing handler
        """
        self._table_extractor = table_extractor or TableExtractor()
        self._ocr_processor = ocr_processor or ImageOCRProcessor()

    def process_artifact_bundle(
        self,
        artifact: "ArtifactBundle",
        include_tables: bool = True,
        include_images: bool = True,
    ) -> ProcessedArtifact:
        """Process an artifact bundle into normalized chunks.

        Args:
            artifact: The artifact bundle to process
            include_tables: Whether to extract tables
            include_images: Whether to process images

        Returns:
            Processed artifact with extracted content
        """
        result = ProcessedArtifact(
            source_id=artifact.source_id,
            source_type=artifact.source_type.value,
        )

        # Process text content
        if artifact.text:
            # Split text into chunks
            text_chunks = self._chunk_text(artifact.text)
            result.text_chunks.extend(text_chunks)

            # Extract tables from text if requested
            if include_tables:
                tables = self._table_extractor.extract(artifact.text)
                result.table_data.extend(tables)

        # Process images if requested
        if include_images and artifact.images:
            for idx, image_data in enumerate(artifact.images):
                if isinstance(image_data, bytes):
                    ocr_result = self._ocr_processor.process_image_bytes(
                        image_data,
                        f"{artifact.source_id}_img_{idx}",
                    )
                elif isinstance(image_data, str):
                    # Could be path or URL
                    if image_data.startswith(("http://", "https://")):
                        ocr_result = self._ocr_processor.process_image_url(image_data)
                    else:
                        ocr_result = self._ocr_processor.process_image_path(image_data)
                else:
                    ocr_result = OCRResult(
                        image_id=f"{artifact.source_id}_img_{idx}",
                        text="[Unsupported image format]",
                        ocr_confidence=0.0,
                    )
                result.ocr_results.append(ocr_result)

        # Store metadata
        result.metadata = dict(artifact.metadata)
        result.metadata["modality"] = self._determine_modality(result)

        return result

    def _chunk_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text] if text.strip() else []

        chunks = []
        paragraphs = text.split("\n\n")

        current_chunk = ""
        for para in paragraphs:
            # If single paragraph is too long, chunk by characters
            if len(para) > max_chunk_size:
                # First flush existing chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Chunk the long paragraph
                for i in range(0, len(para), max_chunk_size):
                    chunks.append(para[i:i + max_chunk_size])
            elif len(current_chunk) + len(para) + 2 <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _determine_modality(self, artifact: ProcessedArtifact) -> str:
        """Determine the primary modality of processed artifact.

        Args:
            artifact: Processed artifact

        Returns:
            Primary modality string
        """
        modalities = []
        if artifact.text_chunks:
            modalities.append(ModalityType.TEXT.value)
        if artifact.table_data:
            modalities.append(ModalityType.TABLE.value)
        if artifact.ocr_results:
            modalities.append(ModalityType.IMAGE.value)

        if len(modalities) > 1:
            return ModalityType.COMPOUND.value
        return modalities[0] if modalities else "unknown"

    def extract_chunks_for_ingestion(
        self,
        artifact: ArtifactBundle,
    ) -> List[Dict[str, Any]]:
        """Extract chunks ready for ingestion into retrieval engine.

        Args:
            artifact: Artifact bundle to process

        Returns:
            List of chunk dictionaries suitable for ingestion
        """
        processed = self.process_artifact_bundle(artifact)
        chunks = []

        # Add text chunks
        for idx, text in enumerate(processed.text_chunks):
            chunks.append({
                "chunk_id": f"{artifact.source_id}_text_{idx}",
                "source_id": artifact.source_id,
                "source_type": artifact.source_type.value,
                "modality": "text",
                "text": text,
                "token_count": len(text.split()),
            })

        # Add table chunks
        for table in processed.table_data:
            table_text = self._table_to_text(table)
            chunks.append({
                "chunk_id": table.table_id,
                "source_id": artifact.source_id,
                "source_type": artifact.source_type.value,
                "modality": "table",
                "text": table_text,
                "token_count": len(table_text.split()),
                "metadata": {"table_headers": table.headers},
            })

        # Add OCR chunks
        for ocr in processed.ocr_results:
            chunks.append({
                "chunk_id": ocr.image_id,
                "source_id": artifact.source_id,
                "source_type": artifact.source_type.value,
                "modality": "image",
                "text": ocr.text,
                "token_count": len(ocr.text.split()),
                "metadata": {"ocr_confidence": ocr.ocr_confidence},
            })

        return chunks

    def _table_to_text(self, table: ExtractedTable) -> str:
        """Convert table to searchable text.

        Args:
            table: Extracted table

        Returns:
            Text representation of table
        """
        lines = []

        if table.caption:
            lines.append(f"Table: {table.caption}")

        if table.headers:
            lines.append(" | ".join(table.headers))
            lines.append(" | ".join(["---"] * len(table.headers)))

        for row in table.rows:
            lines.append(" | ".join(row))

        return "\n".join(lines)


# Import ArtifactBundle for type hints
from .retrieval import ArtifactBundle
